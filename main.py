import os
import time
import glob
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import urllib.parse
import torch
import cv2
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from text_processor import TextProcessor
from color_processor import ColorProcessor
from final_ViT import (
    convert_to_grayscale,
    limited_flip_augment,
    extract_color_histogram,
    extract_edge_features,
    color_profile_sanity_check,
    structural_layout_check,
    normalize_features,
    conservative_similarity_calculation,  
    calculate_discriminative_features_similarity
)
from faiss_vit import MemoryEfficientFAISSIndex, AsyncBatchProcessor

# Request/Response Models
class SimilarityRequest(BaseModel):
    reference_folder_path: str
    comparison_folder_path: str
    infringement_threshold: Optional[float] = 70.0
    batch_size: Optional[int] = 64
    max_images: Optional[int] = 2000
    use_cuda: Optional[bool] = False

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

# FastAPI App Setup
app = FastAPI(
    title="Optimized CPU-Based Logo Similarity Ranker API with FAISS",
    description="High-performance API for ranking logos based on similarity using optimized CPU processing and FAISS indexing",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="/"), name="static")

class OptimizedLogoSimilarityRanker:
    def __init__(self, use_cuda=False):
        """Initialize the optimized Logo Similarity Ranker"""
        self.use_cuda = False
        self.device = torch.device("cpu")
        
        print("Using optimized CPU device for processing")
        
        # Initialize processors
        self.text_processor = TextProcessor(use_cuda=False)
        self.color_processor = ColorProcessor(use_cuda=False)
        
        # Initialize ViT model and feature extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)
        
        # Initialize optimized FAISS database
        self.faiss_db = MemoryEfficientFAISSIndex()
        
        # Initialize async batch processor
        self.batch_processor = AsyncBatchProcessor(max_workers=4)
        
        self.infringement_threshold = 70.0
        self.current_comparison_folder = None
        
        # Feature cache for avoiding recomputation
        self.feature_cache = {}
    
    def extract_vit_features_single(self, image_path):
        """Extract ViT features for a single image with caching"""
        # Check cache first
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]
        
        try:
            img = np.array(Image.open(image_path).convert("RGB"))
            img_gray = convert_to_grayscale(img)
            
            # Generate limited variants
            variants = limited_flip_augment(img_gray)
            
            # Extract features for all variants
            variant_features = []
            for variant in variants:
                inputs = self.feature_extractor(images=variant, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.vit_model(**inputs)
                variant_features.append(outputs.last_hidden_state[:,0].cpu().numpy())
            
            # Cache the result
            self.feature_cache[image_path] = variant_features
            return variant_features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    async def index_comparison_folder_async(self, comparison_folder_path):
        """Optimized indexing with pre-filtering"""
        if self.current_comparison_folder == comparison_folder_path:
            print("Comparison folder already indexed")
            return
        
        print(f"Indexing comparison folder: {comparison_folder_path}")
        
        # Step 1: Get all image files
        all_images = get_image_files_from_folder(comparison_folder_path)
        if not all_images:
            print("No images found in comparison folder")
            return
        
        print(f"Found {len(all_images)} total image files")
        
        # Step 2: Filter already indexed files (fastest check)
        new_images = self.faiss_db.batch_filter_new_files(all_images)
        if not new_images:
            print("All files already indexed")
            self.current_comparison_folder = comparison_folder_path
            return
        
        # Step 3: Deduplicate remaining files (before any feature extraction)
        unique_images = self.batch_processor.duplicate_detector.batch_deduplicate_files(new_images)
        if not unique_images:
            print("No unique files to process")
            return
        
        print(f"Processing {len(unique_images)} unique new images...")
        
        # Step 4: Process only unique, new files in batches
        batch_size = 64
        total_added = 0
        
        for i in range(0, len(unique_images), batch_size):
            batch_paths = unique_images[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(unique_images) + batch_size - 1)//batch_size}")
            
            # Extract features for batch
            batch_vectors = await self._extract_features_batch_optimized(batch_paths)
            
            if batch_vectors:
                added_count = self.faiss_db.add_vectors(batch_vectors, batch_paths)
                total_added += added_count
        
        # Save database
        self.faiss_db.save_database()
        self.current_comparison_folder = comparison_folder_path
        print(f"Indexing complete. Added {total_added} new unique images")
        print(f"Database stats: {self.faiss_db.get_stats()}")
    
    async def _extract_features_batch_optimized(self, image_paths):
        """Extract features without duplicate checking (already done)"""
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(
                self._extract_single_feature_direct(image_path)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if not isinstance(result, Exception) and result is not None and len(result) > 0:
                # Use the first variant as the representative vector
                valid_results.append(result[0].flatten())
        
        return valid_results
    
    async def _extract_single_feature_direct(self, image_path):
        """Direct feature extraction without duplicate checking"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.batch_processor.executor, 
            self.extract_vit_features_single, 
            image_path
        )
    
    def calculate_vit_similarity_with_faiss(self, ref_features, k=None):
        """Calculate ViT similarity using optimized FAISS search with conservative calculation"""
        if not ref_features or len(ref_features) == 0:
            return []
        
        # Use the first variant as query vector for initial FAISS search
        query_vector = ref_features[0].flatten()
        
        # Search in FAISS database with limited results for performance
        if k is None:
            k = min(10000, self.faiss_db.index.ntotal)
        
        search_results = self.faiss_db.search_similar(query_vector, k)
        
        # Now apply conservative similarity calculation for each result
        similarities = []
        for result in search_results:
            # Get the comparison image path
            comparison_path = result['path']
            
            # Extract features for the comparison image (this should use your existing feature extraction)
            comparison_features = self.extract_vit_features_single(comparison_path)
            
            if comparison_features and len(comparison_features) > 0:
                # Apply conservative similarity calculation
                conservative_result = conservative_similarity_calculation(
                    ref_features, 
                    comparison_features, 
                    min_consensus=3
                )
                
                # Convert to percentage (0-100)
                vit_score = max(0.0, min(100.0, conservative_result['vit_score'] * 100))
                
                similarities.append({
                    'path': result['path'],
                    'filename': result['filename'],
                    'vit_similarity': vit_score,
                    'conservative_metrics': {
                        'median_similarity': conservative_result['median_similarity'],
                        'mean_similarity': conservative_result['mean_similarity'],
                        'max_similarity': conservative_result['max_similarity'],
                        'consensus_ratio': conservative_result['consensus_ratio']
                    }
                })
            else:
                # Fallback to original FAISS score if feature extraction fails
                score = max(0.0, min(100.0, result['similarity_score'] * 100))
                similarities.append({
                    'path': result['path'],
                    'filename': result['filename'],
                    'vit_similarity': score
                })
        
        return similarities
    
    async def process_logos_batch_cpu_with_faiss_async(self, reference_logo_path, comparison_folder_path, batch_size=32):
        """Optimized processing with pre-filtering"""
        
        # Index comparison folder (now with pre-filtering)
        await self.index_comparison_folder_async(comparison_folder_path)
        
        # Extract reference features
        ref_vit_features = self.extract_vit_features_single(reference_logo_path)
        ref_colors, ref_weights = self.color_processor.extract_dominant_colors(reference_logo_path)
        ref_text = self.text_processor.extract_text(reference_logo_path)
        
        # Get ViT similarities using FAISS
        vit_results = self.calculate_vit_similarity_with_faiss(ref_vit_features)
        
        # Process each result
        all_results = []
        
        # Get all comparison image paths for color and text processing
        comparison_paths = [result['path'] for result in vit_results]
        
        # Process in batches for better performance
        effective_batch_size = min(batch_size, 32)
        
        for i in range(0, len(comparison_paths), effective_batch_size):
            batch_paths = comparison_paths[i:i + effective_batch_size]
            batch_vit_results = vit_results[i:i + effective_batch_size]
            
            # Extract colors for batch in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                color_futures = [
                    executor.submit(self.color_processor.extract_dominant_colors, path)
                    for path in batch_paths
                ]
                batch_colors_weights = [future.result() for future in color_futures]
            
            batch_colors = [cw[0] for cw in batch_colors_weights]
            batch_weights = [cw[1] for cw in batch_colors_weights]
            
            # Extract text for batch
            batch_texts = self.text_processor.extract_text_batch(batch_paths)
            
            # Calculate color similarities
            color_scores = self.color_processor.calculate_similarity_batch(
                ref_colors, ref_weights, batch_colors, batch_weights
            )
            
            # Calculate final scores
            for j, (path, vit_result) in enumerate(zip(batch_paths, batch_vit_results)):
                try:
                    # Get precomputed scores
                    vit_score = vit_result['vit_similarity']
                    color_score = color_scores[j] if j < len(color_scores) else 0.0
                    text_score = self.text_processor.calculate_similarity(ref_text, batch_texts[j])
                    
                    # Calculate image score
                    if color_score == 0:
                        image_score = vit_score
                    elif vit_score < color_score:
                        image_score = 0.6 * color_score + 0.4 * vit_score
                    else:
                        image_score = 0.75 * vit_score + 0.25 * color_score
                    
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
        
        return all_results
    
    def calculate_aggregated_scores(self, text_score, image_score):
        """Calculate final aggregated score"""
        if text_score == 0:
            final_score = image_score
        elif image_score >= 75 and text_score >= 75:
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
    """Initialize the optimized ranker on startup"""
    global ranker
    print("Initializing Optimized CPU-based Logo Similarity Ranker with FAISS...")
    ranker = OptimizedLogoSimilarityRanker(use_cuda=False)
    print("Optimized ranker initialized successfully!")

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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Optimized CPU-based Logo Similarity Ranker API with FAISS",
        "version": "4.0.0",
        "cuda_available": False,
        "device": "CPU",
        "processing_mode": "Optimized CPU + FAISS IVF + Pre-filtering",
        "optimizations": [
            "File system level duplicate detection",
            "Memory-mapped fast hashing",
            "Pre-processing duplicate filtering",
            "Async batch processing", 
            "Memory-efficient FAISS indexing",
            "Feature caching",
            "Parallel processing"
        ],
        "faiss_stats": ranker.faiss_db.get_stats() if ranker else None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": False,
        "device": "CPU",
        "gpu_memory_mb": None,
        "ranker_initialized": ranker is not None,
        "faiss_stats": ranker.faiss_db.get_stats() if ranker else None,
        "optimizations_active": True
    }

@app.get("/database/stats")
async def get_database_stats():
    """Get optimized FAISS database statistics"""
    global ranker
    if ranker is None:
        raise HTTPException(status_code=500, detail="Ranker not initialized")
    
    stats = ranker.faiss_db.get_stats()
    stats["cache_size"] = len(ranker.feature_cache)
    stats["duplicate_detector_stats"] = {
        "processed_files": len(ranker.batch_processor.duplicate_detector.processed_files),
        "size_cache_entries": len(ranker.batch_processor.duplicate_detector.file_size_cache)
    }
    return stats

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logos(request: SimilarityRequest):
    """Analyze logo similarity with optimized FAISS processing"""
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
    
    if not reference_images:
        raise HTTPException(status_code=400, detail="No image files found in reference folder")
    
    # Update ranker settings
    ranker.infringement_threshold = request.infringement_threshold
    
    # Limit images to process
    if len(reference_images) > request.max_images:
        reference_images = reference_images[:request.max_images]
    
    # Process each reference image using optimized FAISS
    batch_results = []
    total_start_time = time.time()
    
    for i, ref_image_path in enumerate(reference_images):
        batch_start_time = time.time()
        
        try:
            # Use optimized async FAISS processing
            similarity_results = await ranker.process_logos_batch_cpu_with_faiss_async(
                ref_image_path,
                request.comparison_folder_path,
                batch_size=request.batch_size
            )
            
            similarity_results.sort(key=lambda x: x['final_similarity'], reverse=True)
            
            batch_processing_time = time.time() - batch_start_time
            
            results = [SimilarityResult(**result) for result in similarity_results]
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
        "optimized_faiss_database_stats": ranker.faiss_db.get_stats(),
        "performance_improvements": {
            "file_system_dedup": "Inode + size + mtime signatures",
            "content_dedup": "Memory-mapped fast hashing",
            "pre_filtering": "Index check before processing",
            "batch_processing": "Async with parallel feature extraction",
            "faiss_indexing": "IVF with optimized threading",
            "memory_management": "Feature caching and chunked processing"
        }
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
