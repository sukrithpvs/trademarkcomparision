import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import jellyfish
from PIL import Image
from sklearn.cluster import KMeans
import easyocr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import torchvision
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
import io
import tempfile
import glob
import pandas as pd
import warnings
import concurrent.futures
from functools import partial
import time
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import urllib.parse

# Enhanced CUDA imports
import cupy as cp
from numba import cuda
import math

# Add these new imports for enhanced color analysis
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI(
    title="CUDA-Accelerated Logo Similarity Ranker API",
    description="API for ranking logos based on similarity using GPU acceleration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="/"), name="static")

@app.get("/image/{file_path:path}")
async def get_image(file_path: str):
    """Serve images from file system"""
    try:
        decoded_path = urllib.parse.unquote(file_path)
        if not os.path.exists(decoded_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(decoded_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {str(e)}")

# Request/Response Models (unchanged)
class SimilarityRequest(BaseModel):
    reference_folder_path: str
    comparison_folder_path: str
    infringement_threshold: Optional[float] = 70.0
    batch_size: Optional[int] = 64  # Increased for GPU
    max_images: Optional[int] = 2000
    use_cuda: Optional[bool] = True

class SimilarityResult(BaseModel):
    logo_path: str
    logo_name: str
    text_similarity: float
    color_similarity: float
    vit_similarity: float
    image_score: float
    final_similarity: float
    infringement_detected: bool
    text1: str
    text2: str

class BatchResult(BaseModel):
    reference_logo: str
    results: List[SimilarityResult]
    processing_time: float
    infringement_count: int

class AnalysisResponse(BaseModel):
    total_reference_images: int
    processed_reference_images: int
    total_processing_time: float
    batch_results: List[BatchResult]
    summary: Dict[str, Any]
    reference_folder_path: str
    comparison_folder_path: str

# CUDA Kernels for color similarity computation
@cuda.jit
def cuda_color_distance_kernel(colors1, colors2, weights1, weights2, distances, n_colors):
    """CUDA kernel for computing color distances in parallel"""
    idx = cuda.grid(1)
    if idx < colors1.shape[0]:
        total_distance = 0.0
        total_weight = 0.0
        
        for i in range(n_colors):
            if weights1[idx, i] < 0.05:
                continue
                
            min_delta_e = 1000.0
            
            for j in range(n_colors):
                # Compute LAB distance (simplified)
                lab1_l, lab1_a, lab1_b = colors1[idx, i, 0], colors1[idx, i, 1], colors1[idx, i, 2]
                lab2_l, lab2_a, lab2_b = colors2[j, 0], colors2[j, 1], colors2[j, 2]
                
                delta_l = lab1_l - lab2_l
                delta_a = lab1_a - lab2_a
                delta_b = lab1_b - lab2_b
                
                delta_e = math.sqrt(delta_l*delta_l + delta_a*delta_a + delta_b*delta_b)
                min_delta_e = min(min_delta_e, delta_e)
            
            total_distance += min_delta_e * weights1[idx, i]
            total_weight += weights1[idx, i]
        
        if total_weight > 0:
            distances[idx] = total_distance / total_weight
        else:
            distances[idx] = 100.0

@cuda.jit
def cuda_vit_similarity_kernel(features1, features2, similarities):
    """CUDA kernel for computing ViT feature similarities"""
    idx = cuda.grid(1)
    if idx < features1.shape[0]:
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for i in range(features1.shape[1]):
            f1 = features1[idx, i]
            f2 = features2[i]
            
            dot_product += f1 * f2
            norm1 += f1 * f1
            norm2 += f2 * f2
        
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        
        if norm1 > 0 and norm2 > 0:
            similarities[idx] = dot_product / (norm1 * norm2)
        else:
            similarities[idx] = 0.0

class LogoSimilarityRanker:
    def __init__(self, use_cuda=True):
        """Initialize the Logo Similarity Ranker with CUDA support"""
        # Check CUDA availability
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            
            # Set CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        # Initialize OCR engine with GPU support if available
        print('Loading OCR engine...')
        try:
            self.reader = easyocr.Reader(['en'], gpu=self.use_cuda)
            print(f"EasyOCR initialized with GPU: {self.use_cuda}")
        except Exception as e:
            print(f"Failed to initialize EasyOCR with GPU, falling back to CPU: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)
            
        # Initialize Vision Transformer (ViT) model
        print('Loading Vision Transformer model...')
        self.vit_model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Remove the classification head to get features
        self.vit_model.heads = torch.nn.Identity()
        self.vit_model.eval()
        
        # Move model to device
        self.vit_model = self.vit_model.to(self.device)
        
        # Define image transformations for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define infringement threshold
        self.infringement_threshold = 70.0
        
        # CUDA memory management
        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def extract_text_batch_cuda(self, image_paths, max_workers=8):
        """Extract text from multiple images in parallel with GPU acceleration"""
        def extract_single_text(path):
            try:
                results = self.reader.readtext(path)
                return ' '.join([result[1] for result in results]).strip().lower()
            except:
                return ""
        
        # Use more workers for GPU-accelerated OCR
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            texts = list(executor.map(extract_single_text, image_paths))
        
        return texts
    
    def extract_text(self, image_path):
        """Extract text from logo image using EasyOCR with GPU acceleration"""
        try:
            results = self.reader.readtext(image_path)
            text = ' '.join([result[1] for result in results])
            return text.strip().lower()
        except Exception as e:
            return ""
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate text similarity using Levenshtein distance"""
        if not text1 or not text2:
            return 0.0
            
        distance = jellyfish.levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 0.0
            
        similarity = (1 - (distance / max_len)) * 100
        return similarity
    
    def extract_dominant_colors_cuda(self, image_path, n_colors=8):
        """CUDA-accelerated color extraction"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
            
            # Convert to RGB and resize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (200, 200))
            
            if self.use_cuda:
                # Use CuPy for GPU-accelerated image processing
                image_gpu = cp.asarray(image)
                
                # Apply bilateral filter on GPU
                image_gpu = cp.asarray(cv2.bilateralFilter(cp.asnumpy(image_gpu), 9, 75, 75))
                
                # Reshape for clustering
                pixels_gpu = image_gpu.reshape(-1, 3)
                
                # Filter pixels on GPU
                pixel_sums = cp.sum(pixels_gpu, axis=1)
                brightness_mask = (pixel_sums > 50) & (pixel_sums < 700)
                
                if cp.sum(brightness_mask) > n_colors * 15:
                    pixels_gpu = pixels_gpu[brightness_mask]
                
                # Convert back to CPU for KMeans (scikit-learn doesn't support GPU)
                pixels = cp.asnumpy(pixels_gpu)
            else:
                # CPU fallback
                image = cv2.bilateralFilter(image, 9, 75, 75)
                pixels = image.reshape(-1, 3)
                pixel_sums = pixels.sum(axis=1)
                brightness_mask = (pixel_sums > 50) & (pixel_sums < 700)
                
                if brightness_mask.sum() > n_colors * 15:
                    pixels = pixels[brightness_mask]
            
            # Perform clustering
            if len(pixels) > n_colors:
                kmeans = KMeans(
                    n_clusters=n_colors,
                    random_state=42,
                    n_init=15,
                    max_iter=300,
                    tol=1e-4
                )
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_
                
                labels = kmeans.labels_
                weights = np.bincount(labels, minlength=n_colors) / len(labels)
                
                # Sort by weight
                sorted_indices = np.argsort(weights)[::-1]
                colors = colors[sorted_indices]
                weights = weights[sorted_indices]
                
                # Filter out colors with very low weights
                significant_mask = weights > 0.03
                if significant_mask.sum() >= 3:
                    colors = colors[significant_mask]
                    weights = weights[significant_mask]
                    weights = weights / weights.sum()
                    
                    # Pad if necessary
                    if len(colors) < n_colors:
                        pad_size = n_colors - len(colors)
                        colors = np.vstack([colors, np.zeros((pad_size, 3))])
                        weights = np.concatenate([weights, np.zeros(pad_size)])
            else:
                colors = np.zeros((n_colors, 3))
                if len(pixels) > 0:
                    colors[:len(pixels)] = pixels[:n_colors]
                weights = np.ones(n_colors) / n_colors
            
            return colors.astype(np.uint8), weights
            
        except Exception as e:
            print(f"CUDA color extraction failed: {e}")
            return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
    
    def calculate_color_similarity_cuda_batch(self, ref_colors, ref_weights, batch_colors, batch_weights):
        """CUDA-accelerated batch color similarity calculation"""
        try:
            if not self.use_cuda:
                # Fallback to CPU implementation
                return self.calculate_color_similarity_cpu_batch(ref_colors, ref_weights, batch_colors, batch_weights)
            
            batch_size = len(batch_colors)
            n_colors = len(ref_colors)
            
            # Convert to LAB space (simplified for CUDA)
            ref_lab = self.rgb_to_lab_batch_cuda(ref_colors.reshape(1, -1, 3))
            batch_lab = self.rgb_to_lab_batch_cuda(np.array(batch_colors))
            
            # Prepare CUDA arrays
            colors1_gpu = cuda.to_device(batch_lab)
            colors2_gpu = cuda.to_device(ref_lab[0])
            weights1_gpu = cuda.to_device(np.array(batch_weights))
            weights2_gpu = cuda.to_device(ref_weights)
            distances_gpu = cuda.device_array(batch_size, dtype=np.float32)
            
            # Launch CUDA kernel
            threads_per_block = 256
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            
            cuda_color_distance_kernel[blocks_per_grid, threads_per_block](
                colors1_gpu, colors2_gpu, weights1_gpu, weights2_gpu, distances_gpu, n_colors
            )
            
            # Copy results back to CPU
            distances = distances_gpu.copy_to_host()
            
            # Convert distances to similarities
            similarities = []
            for dist in distances:
                if dist <= 1:
                    similarity = 95 - (dist * 5)
                elif dist <= 3:
                    similarity = 90 - ((dist - 1) * 20)
                elif dist <= 6:
                    similarity = 50 - ((dist - 3) * 10)
                elif dist <= 10:
                    similarity = 20 - ((dist - 6) * 4)
                else:
                    similarity = max(0, 4 - ((dist - 10) * 0.5))
                
                similarities.append(max(0, min(100, similarity)))
            
            return similarities
            
        except Exception as e:
            print(f"CUDA color similarity failed, falling back to CPU: {e}")
            return self.calculate_color_similarity_cpu_batch(ref_colors, ref_weights, batch_colors, batch_weights)
    
    def rgb_to_lab_batch_cuda(self, rgb_batch):
        """Convert RGB batch to LAB using CUDA acceleration"""
        if self.use_cuda:
            try:
                # Use CuPy for GPU-accelerated conversion
                rgb_gpu = cp.asarray(rgb_batch) / 255.0
                
                # Simplified RGB to LAB conversion on GPU
                # This is a simplified version - for production, use proper color space conversion
                lab_gpu = cp.zeros_like(rgb_gpu)
                lab_gpu[:, :, 0] = 0.299 * rgb_gpu[:, :, 0] + 0.587 * rgb_gpu[:, :, 1] + 0.114 * rgb_gpu[:, :, 2]  # L
                lab_gpu[:, :, 1] = rgb_gpu[:, :, 0] - rgb_gpu[:, :, 1]  # a
                lab_gpu[:, :, 2] = rgb_gpu[:, :, 1] - rgb_gpu[:, :, 2]  # b
                
                return cp.asnumpy(lab_gpu * 100)
            except:
                pass
        
        # CPU fallback
        lab_batch = []
        for rgb_image in rgb_batch:
            lab_colors = []
            for rgb_color in rgb_image:
                lab_colors.append(self.rgb_to_lab_accurate(rgb_color))
            lab_batch.append(lab_colors)
        
        return np.array(lab_batch)
    
    def extract_vit_features_batch_cuda(self, image_paths, batch_size=64):
        """CUDA-accelerated batch ViT feature extraction"""
        all_features = []
        
        # Use larger batch sizes for GPU
        effective_batch_size = batch_size if self.use_cuda else min(batch_size, 32)
        
        for i in range(0, len(image_paths), effective_batch_size):
            batch_paths = image_paths[i:i + effective_batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception:
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            if batch_tensors:
                # Stack into batch and move to device
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Extract features for entire batch
                with torch.no_grad():
                    if self.use_cuda:
                        # Use mixed precision for faster inference
                        with torch.cuda.amp.autocast():
                            features = self.vit_model(batch_tensor)
                    else:
                        features = self.vit_model(batch_tensor)
                
                # Convert to CPU and add to results
                all_features.extend(features.cpu().float().numpy())
                
                # Clear GPU cache periodically
                if self.use_cuda and i % (effective_batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        return all_features
    
    def calculate_vit_similarity_cuda_batch(self, ref_features, batch_features):
        """CUDA-accelerated batch ViT similarity calculation"""
        try:
            if not self.use_cuda:
                # CPU fallback
                similarities = []
                for features in batch_features:
                    sim = 1 - cosine(ref_features, features)
                    similarities.append(sim * 100)
                return similarities
            
            batch_size = len(batch_features)
            feature_dim = len(ref_features)
            
            # Prepare CUDA arrays
            features1_gpu = cuda.to_device(np.array(batch_features, dtype=np.float32))
            features2_gpu = cuda.to_device(ref_features.astype(np.float32))
            similarities_gpu = cuda.device_array(batch_size, dtype=np.float32)
            
            # Launch CUDA kernel
            threads_per_block = 256
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            
            cuda_vit_similarity_kernel[blocks_per_grid, threads_per_block](
                features1_gpu, features2_gpu, similarities_gpu
            )
            
            # Copy results back to CPU
            similarities = similarities_gpu.copy_to_host() * 100
            
            return similarities.tolist()
            
        except Exception as e:
            print(f"CUDA ViT similarity failed, falling back to CPU: {e}")
            # CPU fallback
            similarities = []
            for features in batch_features:
                sim = 1 - cosine(ref_features, features)
                similarities.append(sim * 100)
            return similarities
    
    def calculate_aggregated_scores(self, text_score, image_score):
        """Calculate final aggregated score based on the specified rules"""
        if image_score >= 75 and text_score >= 75:
            final_score = 0.5 * image_score + 0.5 * text_score
        elif image_score < 25 and text_score < 25:
            final_score = 0.5 * image_score + 0.5 * text_score
        elif image_score >= 75 and text_score < 30:
            final_score = 0.7 * image_score + 0.3 * text_score
        elif text_score >= 75 and image_score < 30:
            final_score = 0.3 * image_score + 0.7 * text_score
        else:
            final_score = 0.5 * image_score + 0.5 * text_score
        
        return final_score
    
    def process_logos_batch_cuda(self, reference_logo_path, comparison_paths, batch_size=64):
        """CUDA-accelerated batch logo processing"""
        
        # Extract reference features once
        ref_vit_features = self.extract_vit_features_batch_cuda([reference_logo_path], 1)[0]
        ref_colors, ref_weights = self.extract_dominant_colors_cuda(reference_logo_path)
        ref_text = self.extract_text(reference_logo_path)
        
        # Process comparison images in larger batches for GPU
        all_results = []
        effective_batch_size = batch_size if self.use_cuda else min(batch_size, 32)
        
        for i in range(0, len(comparison_paths), effective_batch_size):
            batch_paths = comparison_paths[i:i + effective_batch_size]
            
            # Batch extract features using CUDA acceleration
            batch_vit_features = self.extract_vit_features_batch_cuda(batch_paths, effective_batch_size)
            batch_colors = []
            batch_weights = []
            
            # Extract colors in parallel
            for path in batch_paths:
                colors, weights = self.extract_dominant_colors_cuda(path)
                batch_colors.append(colors)
                batch_weights.append(weights)
            
            batch_texts = self.extract_text_batch_cuda(batch_paths)
            
            # Calculate similarities using CUDA
            color_scores = self.calculate_color_similarity_cuda_batch(
                ref_colors, ref_weights, batch_colors, batch_weights
            )
            vit_scores = self.calculate_vit_similarity_cuda_batch(
                ref_vit_features, batch_vit_features
            )
            
            # Calculate final scores
            for j, path in enumerate(batch_paths):
                try:
                    # Text similarity
                    text_score = self.calculate_text_similarity(ref_text, batch_texts[j])
                    
                    # Get precomputed scores
                    color_score = color_scores[j] if j < len(color_scores) else 0.0
                    vit_score = vit_scores[j] if j < len(vit_scores) else 0.0
                    
                    # Calculate image score
                    image_score = 0.7 * vit_score + 0.3 * color_score
                    
                    # Calculate final aggregated score
                    final_score = self.calculate_aggregated_scores(text_score, image_score)
                    
                    result = {
                        'logo_path': path,
                        'logo_name': os.path.basename(path),
                        'text_similarity': text_score,
                        'color_similarity': color_score,
                        'vit_similarity': vit_score,
                        'image_score': image_score,
                        'final_similarity': final_score,
                        'infringement_detected': final_score >= self.infringement_threshold,
                        'text1': ref_text,
                        'text2': batch_texts[j]
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
            
            # Clear GPU cache periodically
            if self.use_cuda:
                torch.cuda.empty_cache()
        
        return all_results
    
    # CPU fallback methods
    def calculate_color_similarity_cpu_batch(self, ref_colors, ref_weights, batch_colors, batch_weights):
        """CPU fallback for color similarity calculation"""
        similarities = []
        for colors, weights in zip(batch_colors, batch_weights):
            sim = self.calculate_color_similarity_strict(ref_colors, ref_weights, colors, weights)
            similarities.append(sim)
        return similarities
    
    def rgb_to_lab_accurate(self, rgb_color):
        """Convert RGB to LAB using colormath library for accuracy"""
        try:
            rgb_normalized = rgb_color / 255.0
            rgb_color_obj = sRGBColor(
                rgb_normalized[0], 
                rgb_normalized[1], 
                rgb_normalized[2]
            )
            lab_color = convert_color(rgb_color_obj, LabColor)
            return np.array([lab_color.lab_l, lab_color.lab_a, lab_color.lab_b])
        except Exception as e:
            return self.rgb_to_lab_manual(rgb_color)
    
    def rgb_to_lab_manual(self, rgb_color):
        """Manual RGB to LAB conversion as fallback"""
        try:
            rgb_normalized = rgb_color / 255.0
            
            def gamma_correct(c):
                return np.where(c > 0.04045, np.power((c + 0.055) / 1.055, 2.4), c / 12.92)
            
            rgb_linear = gamma_correct(rgb_normalized)
            
            xyz = np.dot(rgb_linear, [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ])
            
            xyz = xyz / [95.047, 100.000, 108.883]
            
            def f(t):
                return np.where(t > 0.008856, np.power(t, 1/3), (7.787 * t) + (16/116))
            
            fx, fy, fz = f(xyz[0]), f(xyz[1]), f(xyz[2])
            
            L = (116 * fy) - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            
            return np.array([L, a, b])
        except:
            return np.array([50, 0, 0])
    
    def calculate_color_similarity_strict(self, colors1, weights1, colors2, weights2):
        """Strict color similarity calculation with proper Delta E thresholds"""
        try:
            avg_distance_cpu = self.calculate_color_similarity_cpu_strict(colors1, weights1, colors2, weights2)
            
            # Strict Delta E to similarity conversion
            if avg_distance_cpu <= 1:
                similarity = 95 - (avg_distance_cpu * 5)
            elif avg_distance_cpu <= 3:
                similarity = 90 - ((avg_distance_cpu - 1) * 20)
            elif avg_distance_cpu <= 6:
                similarity = 50 - ((avg_distance_cpu - 3) * 10)
            elif avg_distance_cpu <= 10:
                similarity = 20 - ((avg_distance_cpu - 6) * 4)
            else:
                similarity = max(0, 4 - ((avg_distance_cpu - 10) * 0.5))
            
            return max(0, min(100, similarity))
            
        except Exception as e:
            print(f"Strict color similarity calculation failed: {e}")
            return 0.0
    
    def calculate_color_similarity_cpu_strict(self, colors1, weights1, colors2, weights2):
        """Strict CPU color similarity calculation"""
        try:
            total_distance = 0
            total_weight = 0
            
            # Convert to LAB space
            lab_colors1 = []
            lab_colors2 = []
            
            for color in colors1:
                lab_colors1.append(self.rgb_to_lab_accurate(color))
            
            for color in colors2:
                lab_colors2.append(self.rgb_to_lab_accurate(color))
            
            lab_colors1 = np.array(lab_colors1)
            lab_colors2 = np.array(lab_colors2)
            
            # Calculate distances for significant colors only
            for i, (lab1, weight1) in enumerate(zip(lab_colors1, weights1)):
                if weight1 < 0.05:
                    continue
                
                min_delta_e = float('inf')
                
                for lab2 in lab_colors2:
                    try:
                        color1_lab = LabColor(lab1[0], lab1[1], lab1[2])
                        color2_lab = LabColor(lab2[0], lab2[1], lab2[2])
                        delta_e = delta_e_cie2000(color1_lab, color2_lab)
                        min_delta_e = min(min_delta_e, delta_e)
                    except:
                        euclidean_dist = np.linalg.norm(lab1 - lab2)
                        min_delta_e = min(min_delta_e, euclidean_dist)
                
                total_distance += min_delta_e * weight1
                total_weight += weight1
            
            if total_weight > 0:
                return total_distance / total_weight
            else:
                return 100.0
                
        except Exception as e:
            print(f"CPU strict color similarity failed: {e}")
            return 100.0

def get_image_files_from_folder(folder_path):
    """Get all image files from a folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    
    return image_files

# Global ranker instance
ranker = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ranker on startup"""
    global ranker
    print("Initializing CUDA-accelerated Logo Similarity Ranker...")
    ranker = LogoSimilarityRanker(use_cuda=True)
    print("Ranker initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name() if cuda_available else "CPU"
    
    return {
        "message": "CUDA-accelerated Logo Similarity Ranker API",
        "version": "2.0.0",
        "cuda_available": cuda_available,
        "device": device_name,
        "processing_mode": "GPU" if cuda_available else "CPU",
        "features": [
            "CUDA-accelerated color comparison",
            "GPU-optimized ViT feature extraction",
            "Parallel batch processing",
            "Mixed precision inference",
            "CuPy acceleration for image processing"
        ],
        "endpoints": {
            "/analyze": "POST - Analyze logo similarity with GPU acceleration",
            "/health": "GET - Health check with GPU status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU status"""
    cuda_available = torch.cuda.is_available()
    gpu_memory = None
    
    if cuda_available:
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "cached": torch.cuda.memory_reserved() / 1024**2,  # MB
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        }
    
    return {
        "status": "healthy",
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name() if cuda_available else "CPU",
        "gpu_memory_mb": gpu_memory,
        "ranker_initialized": ranker is not None,
        "processing_mode": "GPU" if cuda_available else "CPU"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logos(request: SimilarityRequest):
    """
    Analyze logo similarity with CUDA acceleration
    """
    global ranker
    
    if ranker is None:
        raise HTTPException(status_code=500, detail="Ranker not initialized")
    
    # Validate paths
    if not os.path.exists(request.reference_folder_path):
        raise HTTPException(status_code=400, detail=f"Reference folder path does not exist: {request.reference_folder_path}")
    
    if not os.path.exists(request.comparison_folder_path):
        raise HTTPException(status_code=400, detail=f"Comparison folder path does not exist: {request.comparison_folder_path}")
    
    # Get image files
    reference_images = get_image_files_from_folder(request.reference_folder_path)
    comparison_images = get_image_files_from_folder(request.comparison_folder_path)
    
    if not reference_images:
        raise HTTPException(status_code=400, detail="No image files found in reference folder")
    
    if not comparison_images:
        raise HTTPException(status_code=400, detail="No image files found in comparison folder")
    
    # Update ranker settings
    ranker.infringement_threshold = request.infringement_threshold
    
    # Limit images to process
    if len(reference_images) > request.max_images:
        print(f"Warning: Reference folder contains {len(reference_images)} images, limiting to {request.max_images}")
        reference_images = reference_images[:request.max_images]
    
    if len(comparison_images) > request.max_images:
        print(f"Warning: Comparison folder contains {len(comparison_images)} images, limiting to {request.max_images}")
        comparison_images = comparison_images[:request.max_images]
    
    processing_mode = "GPU" if ranker.use_cuda else "CPU"
    print(f"Processing {len(reference_images)} reference images against {len(comparison_images)} comparison images with {processing_mode}")
    
    # Process each reference image
    batch_results = []
    total_start_time = time.time()
    
    for i, ref_image_path in enumerate(reference_images):
        print(f"Processing reference image {i+1}/{len(reference_images)}: {os.path.basename(ref_image_path)}")
        
        batch_start_time = time.time()
        
        try:
            # Process this reference image against all comparison images using CUDA
            similarity_results = ranker.process_logos_batch_cuda(
                ref_image_path,
                comparison_images,
                batch_size=request.batch_size
            )
            
            # Sort results by final similarity
            similarity_results.sort(key=lambda x: x['final_similarity'], reverse=True)
            
            batch_processing_time = time.time() - batch_start_time
            
            # Convert to Pydantic models
            results = [
                SimilarityResult(**result) for result in similarity_results
            ]
            
            # Count infringements
            infringement_count = sum(1 for result in results if result.infringement_detected)
            
            batch_result = BatchResult(
                reference_logo=os.path.basename(ref_image_path),
                results=results,
                processing_time=batch_processing_time,
                infringement_count=infringement_count
            )
            
            batch_results.append(batch_result)
                
        except Exception as e:
            print(f"Error processing reference image {ref_image_path}: {str(e)}")
            continue
    
    total_processing_time = time.time() - total_start_time
    
    # Generate summary statistics
    total_comparisons = sum(len(batch.results) for batch in batch_results)
    total_infringements = sum(batch.infringement_count for batch in batch_results)
    avg_processing_time = total_processing_time / len(batch_results) if batch_results else 0
    
    summary = {
        "total_comparisons": total_comparisons,
        "total_infringements": total_infringements,
        "avg_processing_time_per_reference": round(avg_processing_time, 2),
        "images_per_second": round(total_comparisons / total_processing_time, 1) if total_processing_time > 0 else 0,
        "infringement_rate_percent": round((total_infringements / total_comparisons) * 100, 2) if total_comparisons > 0 else 0,
        "analysis_method": f"{processing_mode}-accelerated with CUDA kernels",
        "device_used": torch.cuda.get_device_name() if ranker.use_cuda else "CPU",
        "cuda_available": torch.cuda.is_available()
    }
    
    return AnalysisResponse(
        total_reference_images=len(reference_images),
        processed_reference_images=len(batch_results),
        total_processing_time=round(total_processing_time, 2),
        batch_results=batch_results,
        summary=summary,
        reference_folder_path=request.reference_folder_path,
        comparison_folder_path=request.comparison_folder_path
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
