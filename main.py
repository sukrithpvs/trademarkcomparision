import os
import time
import glob
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import urllib.parse
import torch

from text_processor import TextProcessor
from color_processor import ColorProcessor
from vit_processor import ViTProcessor

# Request/Response Models
class SimilarityRequest(BaseModel):
    reference_folder_path: str
    comparison_folder_path: str
    infringement_threshold: Optional[float] = 70.0
    batch_size: Optional[int] = 32
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
    title="CPU-Based Logo Similarity Ranker API",
    description="API for ranking logos based on similarity using CPU processing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="/"), name="static")

class LogoSimilarityRanker:
    def __init__(self, use_cuda=False):
        """Initialize the Logo Similarity Ranker with CPU processing"""
        self.use_cuda = False
        self.device = torch.device("cpu")
        
        print("Using CPU device for processing")
        
        # Initialize processors
        self.text_processor = TextProcessor(use_cuda=False)
        self.color_processor = ColorProcessor(use_cuda=False)
        self.vit_processor = ViTProcessor(use_cuda=False, device=self.device)
        
        self.infringement_threshold = 70.0
    
    
    def calculate_aggregated_scores(self, text_score, image_score):
            """Calculate final aggregated score based on the specified rules"""
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

    
    def process_logos_batch_cpu(self, reference_logo_path, comparison_paths, batch_size=16):
        """CPU-based batch logo processing"""
        
        # Extract reference features once
        ref_vit_features = self.vit_processor.extract_features_batch([reference_logo_path], 1)[0]
        ref_colors, ref_weights = self.color_processor.extract_dominant_colors(reference_logo_path)
        ref_text = self.text_processor.extract_text(reference_logo_path)
        
        # Process comparison images in batches
        all_results = []
        effective_batch_size = min(batch_size, 16)  # Smaller batches for CPU
        
        for i in range(0, len(comparison_paths), effective_batch_size):
            batch_paths = comparison_paths[i:i + effective_batch_size]
            
            # Batch extract features
            batch_vit_features = self.vit_processor.extract_features_batch(batch_paths, effective_batch_size)
            batch_colors = []
            batch_weights = []
            
            # Extract colors
            for path in batch_paths:
                colors, weights = self.color_processor.extract_dominant_colors(path)
                batch_colors.append(colors)
                batch_weights.append(weights)
            
            batch_texts = self.text_processor.extract_text_batch(batch_paths)
            
            # Calculate similarities
            color_scores = self.color_processor.calculate_similarity_batch(
                ref_colors, ref_weights, batch_colors, batch_weights
            )
            vit_scores = self.vit_processor.calculate_similarity_batch(
                ref_vit_features, batch_vit_features
            )
            
            # Calculate final scores
            for j, path in enumerate(batch_paths):
                try:
                    # Text similarity
                    text_score = self.text_processor.calculate_similarity(ref_text, batch_texts[j])
                    
                    # Get precomputed scores
                    color_score = color_scores[j] if j < len(color_scores) else 0.0
                    vit_score = vit_scores[j] if j < len(vit_scores) else 0.0
                    
                    # Calculate image score
                    if color_score == 0:
                        image_score = vit_score
                    elif vit_score < color_score :
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
    print("Initializing CPU-based Logo Similarity Ranker...")
    ranker = LogoSimilarityRanker(use_cuda=False)
    print("Ranker initialized successfully!")

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
        "message": "CPU-based Logo Similarity Ranker API",
        "version": "2.0.0",
        "cuda_available": False,
        "device": "CPU",
        "processing_mode": "CPU"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": False,
        "device": "CPU",
        "gpu_memory_mb": None,
        "ranker_initialized": ranker is not None
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logos(request: SimilarityRequest):
    """Analyze logo similarity with CPU processing"""
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
        reference_images = reference_images[:request.max_images]
    
    if len(comparison_images) > request.max_images:
        comparison_images = comparison_images[:request.max_images]
    
    # Process each reference image
    batch_results = []
    total_start_time = time.time()
    
    for i, ref_image_path in enumerate(reference_images):
        batch_start_time = time.time()
        
        try:
            similarity_results = ranker.process_logos_batch_cpu(
                ref_image_path,
                comparison_images,
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
        "infringement_rate_percent": round((total_infringements / total_comparisons) * 100, 2) if total_comparisons > 0 else 0
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
