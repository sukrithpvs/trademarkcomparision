import os
import torch
torch.classes.__path__ = []
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

# Add these new imports for enhanced color analysis
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")





app = FastAPI(
    title="CUDA-Accelerated Logo Similarity Ranker API",
    description="API for ranking logos based on similarity using CUDA acceleration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's default port
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.mount("/static", StaticFiles(directory="/"), name="static")

# Add this endpoint to serve images
@app.get("/image/{file_path:path}")
async def get_image(file_path: str):
    """Serve images from file system"""
    try:
        # Decode URL-encoded path
        decoded_path = urllib.parse.unquote(file_path)
        
        # Check if file exists
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
    batch_size: Optional[int] = 512
    max_images: Optional[int] = 2000

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

class LogoSimilarityRanker:
    def __init__(self):
        """Initialize the Logo Similarity Ranker with CUDA-optimized components"""
        # Set device for CUDA acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # GPU optimizations
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Initialize OCR engine with GPU support if available
        print('Loading OCR engine...')
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
        # Initialize Vision Transformer (ViT) model for deep feature extraction
        print('Loading Vision Transformer model...')
        # Load pretrained ViT model
        self.vit_model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Remove the classification head to get features
        self.vit_model.heads = torch.nn.Identity()
        self.vit_model.eval()
        
        # Move model to GPU
        self.vit_model = self.vit_model.to(self.device)
        
        # Enable mixed precision for faster inference
        if self.device.type == 'cuda':
            self.vit_model = self.vit_model.half()
        
        # Define image transformations for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Warm up GPU
        if torch.cuda.is_available():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device).half()
            with torch.no_grad():
                _ = self.vit_model(dummy_input)
            torch.cuda.empty_cache()
        
        # Define infringement threshold
        self.infringement_threshold = 70.0
        
        # Compile CUDA kernels for color processing
        self._compile_cuda_kernels()
    
    def _compile_cuda_kernels(self):
        """Pre-compile CUDA kernels for faster color processing"""
        if not torch.cuda.is_available():
            return
            
        try:
            dummy_colors1 = torch.randn(6, 3, device=self.device)
            dummy_colors2 = torch.randn(6, 3, device=self.device)
            _ = torch.cdist(dummy_colors1, dummy_colors2)
            
            dummy_matrix = torch.randn(1000, 3, device=self.device)
            _ = torch.norm(dummy_matrix, dim=1)
            
            torch.cuda.empty_cache()
            print("CUDA kernels pre-compiled successfully")
        except Exception as e:
            print(f"CUDA kernel compilation warning: {e}")
    
    def extract_text_batch(self, image_paths, max_workers=4):
        """Extract text from multiple images in parallel"""
        def extract_single_text(path):
            try:
                results = self.reader.readtext(path)
                return ' '.join([result[1] for result in results]).strip().lower()
            except:
                return ""
        
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
    
    # NEW: Enhanced strict color extraction
    def extract_dominant_colors_strict(self, image_path, n_colors=8):
        """Strict color extraction with better filtering and preprocessing"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to larger size for better color representation
            image = cv2.resize(image, (200, 200))
            
            # Apply bilateral filter to reduce noise while preserving edges
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Reshape for clustering
            pixels = image.reshape(-1, 3)
            
            # More aggressive filtering to remove background and noise
            # Remove very dark pixels (likely background)
            pixel_sums = pixels.sum(axis=1)
            brightness_mask = (pixel_sums > 50) & (pixel_sums < 700)
            
            if brightness_mask.sum() > n_colors * 15:
                pixels = pixels[brightness_mask]
            
            # Convert to HSV for better saturation filtering
            hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
            saturation = hsv_pixels[:, 0, 1]
            value = hsv_pixels[:, 0, 2]
            
            # Keep only pixels with reasonable saturation and value
            color_mask = (saturation > 40) & (value > 50)
            
            if color_mask.sum() > n_colors * 8:
                pixels = pixels[color_mask]
            
            # Remove grayscale-like colors
            rgb_std = np.std(pixels, axis=1)
            color_variation_mask = rgb_std > 15  # Colors with some variation between RGB channels
            
            if color_variation_mask.sum() > n_colors * 5:
                pixels = pixels[color_variation_mask]
            
            # Perform clustering with more iterations for better results
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
                
                # Sort by weight (most prominent colors first)
                sorted_indices = np.argsort(weights)[::-1]
                colors = colors[sorted_indices]
                weights = weights[sorted_indices]
                
                # Filter out colors with very low weights
                significant_mask = weights > 0.03  # At least 3% presence
                if significant_mask.sum() >= 3:  # Keep at least 3 colors
                    colors = colors[significant_mask]
                    weights = weights[significant_mask]
                    # Renormalize weights
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
            print(f"Strict color extraction failed: {e}")
            return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
    
    # NEW: Batch strict color extraction
    def extract_dominant_colors_batch_strict(self, image_paths, n_colors=8):
        """Strict batch color extraction"""
        def extract_colors_single(path):
            return self.extract_dominant_colors_strict(path, n_colors)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(extract_colors_single, image_paths))
        
        all_colors = [result[0] for result in results]
        all_weights = [result[1] for result in results]
        
        return all_colors, all_weights
    
    # NEW: RGB to LAB conversion methods
    def rgb_to_lab_gpu_batch(self, rgb_colors):
        """Batch RGB to LAB conversion on GPU using PyTorch"""
        try:
            # Normalize RGB to 0-1 range
            rgb_normalized = rgb_colors / 255.0
            
            # Gamma correction
            mask = rgb_normalized > 0.04045
            rgb_linear = torch.where(
                mask,
                torch.pow((rgb_normalized + 0.055) / 1.055, 2.4),
                rgb_normalized / 12.92
            )
            
            # RGB to XYZ transformation matrix
            transform_matrix = torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ], device=self.device, dtype=torch.float32)
            
            xyz = torch.matmul(rgb_linear, transform_matrix.T)
            
            # Normalize by D65 illuminant
            d65 = torch.tensor([95.047, 100.000, 108.883], device=self.device, dtype=torch.float32)
            xyz = xyz / d65
            
            # XYZ to LAB
            mask = xyz > 0.008856
            f_xyz = torch.where(
                mask,
                torch.pow(xyz, 1/3),
                (7.787 * xyz) + (16/116)
            )
            
            L = (116 * f_xyz[:, 1]) - 16
            a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
            b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
            
            return torch.stack([L, a, b], dim=1)
            
        except Exception as e:
            print(f"GPU LAB conversion failed: {e}")
            # Fallback to CPU
            lab_colors = []
            for color in rgb_colors.cpu().numpy():
                lab_colors.append(self.rgb_to_lab_accurate(color))
            return torch.tensor(lab_colors, device=self.device)
    
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
    
    # NEW: Color family analysis
    def calculate_color_family_penalty(self, colors1, weights1, colors2, weights2):
        """Calculate penalty for completely different color families"""
        try:
            # Define color families in HSV space
            def get_color_family(rgb_color):
                hsv = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
                hue = hsv[0]
                sat = hsv[1]
                val = hsv[2]
                
                # Low saturation = grayscale family
                if sat < 50:
                    return 'grayscale'
                
                # Color families based on hue
                if hue < 15 or hue > 165:
                    return 'red'
                elif hue < 35:
                    return 'orange'
                elif hue < 75:
                    return 'yellow'
                elif hue < 105:
                    return 'green'
                elif hue < 135:
                    return 'cyan'
                else:
                    return 'blue'
            
            # Get dominant color families for each image
            families1 = {}
            families2 = {}
            
            for color, weight in zip(colors1, weights1):
                if weight > 0.05:  # Only significant colors
                    family = get_color_family(color)
                    families1[family] = families1.get(family, 0) + weight
            
            for color, weight in zip(colors2, weights2):
                if weight > 0.05:  # Only significant colors
                    family = get_color_family(color)
                    families2[family] = families2.get(family, 0) + weight
            
            # Calculate overlap between color families
            all_families = set(families1.keys()) | set(families2.keys())
            overlap = 0
            
            for family in all_families:
                weight1 = families1.get(family, 0)
                weight2 = families2.get(family, 0)
                overlap += min(weight1, weight2)
            
            # Penalty is inverse of overlap
            penalty = max(0, 1 - overlap * 2)  # Strong penalty for no overlap
            
            return penalty
            
        except Exception as e:
            print(f"Color family penalty calculation failed: {e}")
            return 0.0
    
    # NEW: Strict color similarity calculation
    def calculate_color_similarity_strict(self, colors1, weights1, colors2, weights2):
        """Strict color similarity calculation with proper Delta E thresholds"""
        try:
            if torch.cuda.is_available():
                # GPU-accelerated calculation
                colors1_gpu = torch.tensor(colors1, dtype=torch.float32, device=self.device)
                colors2_gpu = torch.tensor(colors2, dtype=torch.float32, device=self.device)
                weights1_gpu = torch.tensor(weights1, dtype=torch.float32, device=self.device)
                
                # Convert to LAB space on GPU
                lab_colors1 = self.rgb_to_lab_gpu_batch(colors1_gpu)
                lab_colors2 = self.rgb_to_lab_gpu_batch(colors2_gpu)
                
                # Calculate pairwise distances
                distances = torch.cdist(lab_colors1.unsqueeze(0), lab_colors2.unsqueeze(0)).squeeze(0)
                
                # Find minimum distances for each significant color
                min_distances, _ = torch.min(distances, dim=1)
                
                # Only consider colors with significant presence
                significant_mask = weights1_gpu >= 0.05
                
                if torch.sum(significant_mask) > 0:
                    weighted_distances = min_distances[significant_mask] * weights1_gpu[significant_mask]
                    total_weight = torch.sum(weights1_gpu[significant_mask])
                    avg_distance = torch.sum(weighted_distances) / total_weight
                else:
                    avg_distance = torch.tensor(100.0, device=self.device)
                
                avg_distance_cpu = avg_distance.cpu().item()
            else:
                # CPU fallback
                avg_distance_cpu = self.calculate_color_similarity_cpu_strict(colors1, weights1, colors2, weights2)
            
            # Much stricter Delta E to similarity conversion
            if avg_distance_cpu <= 1:
                similarity = 95 - (avg_distance_cpu * 5)  # 95% to 90%
            elif avg_distance_cpu <= 3:
                similarity = 90 - ((avg_distance_cpu - 1) * 20)  # 90% to 50%
            elif avg_distance_cpu <= 6:
                similarity = 50 - ((avg_distance_cpu - 3) * 10)  # 50% to 20%
            elif avg_distance_cpu <= 10:
                similarity = 20 - ((avg_distance_cpu - 6) * 4)  # 20% to 4%
            else:
                similarity = max(0, 4 - ((avg_distance_cpu - 10) * 0.5))  # 4% to 0%
            
            # Additional penalty for completely different color families
            penalty = self.calculate_color_family_penalty(colors1, weights1, colors2, weights2)
            similarity = similarity * (1 - penalty)
            
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
                if weight1 < 0.05:  # Skip insignificant colors
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
    
    # Keep the old method for backward compatibility
    def extract_dominant_colors_fast(self, image_paths, n_colors=5):
        """Fast batch color extraction using optimized approach"""
        all_colors = []
        
        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is None:
                    all_colors.append(np.zeros((n_colors, 3)))
                    continue
                    
                # Resize for faster processing
                image = cv2.resize(image, (64, 64))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Use histogram-based approach instead of KMeans for speed
                pixels = image.reshape(-1, 3)
                
                # Simple quantization approach
                quantized = (pixels // 32) * 32  # Quantize to reduce unique colors
                unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
                
                # Get top colors by frequency
                sorted_indices = np.argsort(counts)[::-1]
                top_colors = unique_colors[sorted_indices[:n_colors]]
                
                # Pad if necessary
                colors = np.zeros((n_colors, 3))
                colors[:len(top_colors)] = top_colors
                all_colors.append(colors)
            except Exception:
                all_colors.append(np.zeros((n_colors, 3)))
        
        return all_colors
    
    def extract_dominant_colors(self, image_path, n_colors=5):
        """Extract dominant colors from the logo using optimized approach"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((n_colors, 3))
                
            # Resize for faster processing
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use histogram-based approach for speed
            pixels = image.reshape(-1, 3)
            quantized = (pixels // 32) * 32
            unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
            
            sorted_indices = np.argsort(counts)[::-1]
            top_colors = unique_colors[sorted_indices[:n_colors]]
            
            colors = np.zeros((n_colors, 3))
            colors[:len(top_colors)] = top_colors
            
            return colors
        except Exception:
            return np.zeros((n_colors, 3))
    
    def calculate_color_similarity(self, colors1, colors2):
        """Calculate similarity between two sets of dominant colors using vectorized operations"""
        try:
            colors1 = np.array(colors1)
            colors2 = np.array(colors2)
            
            distances = np.linalg.norm(colors1[:, np.newaxis] - colors2[np.newaxis, :], axis=2)
            min_distances = np.min(distances, axis=1)
            avg_distance = np.mean(min_distances)
            
            max_distance = 255 * np.sqrt(3)
            similarity = (1 - (avg_distance / max_distance)) * 100
            
            return similarity
        except Exception:
            return 0.0
    
    def extract_vit_features_batch(self, image_paths, batch_size=512):
        """Extract features from multiple images in batches for maximum GPU utilization"""
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch on CPU
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception:
                    # Use zero tensor for failed images
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            if batch_tensors:
                # Stack into batch and move to GPU
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                if self.device.type == 'cuda':
                    batch_tensor = batch_tensor.half()
                
                # Extract features for entire batch
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.amp.autocast("cuda"):
                            features = self.vit_model(batch_tensor)
                    else:
                        features = self.vit_model(batch_tensor)
                
                # Convert to CPU and add to results
                all_features.extend(features.cpu().float().numpy())
        
        return all_features
    
    def extract_vit_features(self, image_path):
        """Extract deep features from the logo using CUDA-accelerated Vision Transformer"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            if self.device.type == 'cuda':
                image_tensor = image_tensor.half()
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast("cuda"):
                        features = self.vit_model(image_tensor)
                else:
                    features = self.vit_model(image_tensor)
            
            return features.cpu().float().numpy().flatten()
        except Exception:
            return np.zeros(768)
    
    def calculate_aggregated_scores(self, text_score, image_score):
        """Calculate final aggregated score based on the specified rules"""
        # Rule 1: If both image score and text score are above 75%
        if image_score >= 75 and text_score >= 75:
            final_score = 0.5 * image_score + 0.5 * text_score
        # Rule 2: If both image score and text score are less than 25%
        elif image_score < 25 and text_score < 25:
            final_score = 0.5 * image_score + 0.5 * text_score
        # Rule 3: If image score is above 75 and text score is less than 30
        elif image_score >= 75 and text_score < 30:
            final_score = 0.7 * image_score + 0.3 * text_score
        # Rule 3 (vice versa): If text score is above 75 and image score is less than 30
        elif text_score >= 75 and image_score < 30:
            final_score = 0.3 * image_score + 0.7 * text_score
        # Default case: equal weighting
        else:
            final_score = 0.5 * image_score + 0.5 * text_score
        
        return final_score
    
    # UPDATED: Process logos batch with strict color analysis
    def process_logos_batch(self, reference_logo_path, comparison_paths, batch_size=512):
        """Process multiple logos efficiently using batch operations with strict color analysis"""
        
        # Extract reference features once using strict method
        ref_vit_features = self.extract_vit_features(reference_logo_path)
        ref_colors, ref_weights = self.extract_dominant_colors_strict(reference_logo_path)
        ref_text = self.extract_text(reference_logo_path)
        
        # Process comparison images in batches
        all_results = []
        
        for i in range(0, len(comparison_paths), batch_size):
            batch_paths = comparison_paths[i:i + batch_size]
            
            # Batch extract features using strict methods
            batch_vit_features = self.extract_vit_features_batch(batch_paths, batch_size)
            batch_colors, batch_weights = self.extract_dominant_colors_batch_strict(batch_paths)
            batch_texts = self.extract_text_batch(batch_paths)
            
            # Calculate similarities for batch
            for j, path in enumerate(batch_paths):
                try:
                    # Text similarity
                    text_score = self.calculate_text_similarity(ref_text, batch_texts[j])
                    
                    # Strict color similarity
                    color_score = self.calculate_color_similarity_strict(
                        ref_colors, ref_weights, batch_colors[j], batch_weights[j]
                    )
                    
                    # ViT similarity
                    vit_sim = 1 - cosine(ref_vit_features, batch_vit_features[j])
                    vit_score = vit_sim * 100
                    
                    # Calculate image score (60% ViT + 40% Color for better color influence)
                    image_score = 0.7 * vit_score + 0.3 * color_score
                    
                    # Calculate final aggregated score using the new rules
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
                    
                except Exception:
                    continue
        
        return all_results

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
    ranker = LogoSimilarityRanker()
    print("Ranker initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    cuda_status = "enabled" if torch.cuda.is_available() else "disabled"
    gpu_info = {}
    
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        }
    
    return {
        "message": "CUDA-Accelerated Logo Similarity Ranker API with Strict Color Analysis",
        "version": "1.0.0",
        "cuda_status": cuda_status,
        "gpu_info": gpu_info,
        "features": [
            "Strict Delta E color comparison",
            "Color family analysis",
            "GPU-accelerated processing",
            "Enhanced color filtering"
        ],
        "endpoints": {
            "/analyze": "POST - Analyze logo similarity for multiple reference images",
            "/health": "GET - Health check",
            "/cuda-stats": "GET - CUDA statistics (if available)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "ranker_initialized": ranker is not None
    }

@app.get("/cuda-stats")
async def cuda_stats():
    """Get CUDA statistics"""
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA not available")
    
    return {
        "gpu_utilization_percent": torch.cuda.utilization(),
        "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "memory_cached_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "device_name": torch.cuda.get_device_name(),
        "device_count": torch.cuda.device_count()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logos(request: SimilarityRequest):
    """
    Analyze logo similarity for multiple reference images against comparison logos
    
    This endpoint processes all images in the reference folder one by one,
    comparing each against all images in the comparison folder using strict color analysis.
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
    
    # Limit images to process with proper warnings
    if len(reference_images) > request.max_images:
        print(f"Warning: Reference folder contains {len(reference_images)} images, limiting to {request.max_images}")
        reference_images = reference_images[:request.max_images]
    
    if len(comparison_images) > request.max_images:
        print(f"Warning: Comparison folder contains {len(comparison_images)} images, limiting to {request.max_images}")
        comparison_images = comparison_images[:request.max_images]
    
    print(f"Processing {len(reference_images)} reference images against {len(comparison_images)} comparison images with strict color analysis")
    
    # Process each reference image
    batch_results = []
    total_start_time = time.time()
    
    for i, ref_image_path in enumerate(reference_images):
        print(f"Processing reference image {i+1}/{len(reference_images)}: {os.path.basename(ref_image_path)}")
        
        batch_start_time = time.time()
        
        try:
            # Process this reference image against all comparison images using strict analysis
            similarity_results = ranker.process_logos_batch(
                ref_image_path,
                comparison_images,
                batch_size=request.batch_size
            )
            
            # Sort results by final similarity (highest to lowest)
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
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and i % 2 == 0:
                torch.cuda.empty_cache()
                
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
        "analysis_method": "Strict Color Analysis with Delta E"
    }
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return AnalysisResponse(
        total_reference_images=len(reference_images),
        processed_reference_images=len(batch_results),
        total_processing_time=round(total_processing_time, 2),
        batch_results=batch_results,
        summary=summary,
        reference_folder_path=request.reference_folder_path,
        comparison_folder_path=request.comparison_folder_path
    )

@app.post("/clear-gpu-cache")
async def clear_gpu_cache():
    """Clear GPU cache"""
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA not available")
    
    torch.cuda.empty_cache()
    return {"message": "GPU cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

