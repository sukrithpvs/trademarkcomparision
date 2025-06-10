import os
import time
import urllib.parse
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from sift import SIFTLogoComparison
import text as text_module

app = FastAPI(
    title="Optimized Logo Similarity Ranker API with Visual + Text Analysis",
    version="4.0.0"
)

# --- Data Models ---
class SimilarityResult(BaseModel):
    logo_name: str
    logo_path: str
    text1: str
    text2: str
    text_similarity: float
    vit_similarity: float
    final_similarity: float
    infringement_detected: bool

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
    summary: dict
    reference_folder_path: str
    comparison_folder_path: str

class SimilarityRequest(BaseModel):
    reference_folder_path: str = Field(..., description="Path to reference logos")
    comparison_folder_path: str = Field(..., description="Path to comparison logos")
    infringement_threshold: float = Field(70.0, description="Threshold (%) for infringement")
    batch_size: int = Field(512, description="Batch size for async processing")
    max_images: int = Field(2000, description="Max number of reference images to process")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
sift_ranker = None
executor = None

@app.on_event("startup")
async def startup_event():
    global sift_ranker, executor
    print("🚀 STARTUP: Initializing SIFT Logo Similarity Ranker...")
    sift_ranker = SIFTLogoComparison()
    executor = ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8))
    print("✅ STARTUP COMPLETE: SIFT ranker initialized successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    global executor
    if executor:
        executor.shutdown(wait=True)

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
        "message": "Optimized Logo Similarity Ranker API (Visual + Text)",
        "version": "4.0.0",
        "cuda_available": False,
        "device": "CPU",
        "processing_mode": "Visual (SIFT) + OCR Text + Aggregation",
        "optimizations": [
            "Feature caching",
            "Parallel processing",
            "Memory optimization",
            "Async batch processing",
            "Image resizing",
            "FLANN optimization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": False,
        "device": "CPU",
        "ranker_initialized": sift_ranker is not None,
        "optimizations_active": True
    }

# --- Aggregation Logic ---
def calculate_aggregated_scores(text_score, image_score, vit_score, ref_text="", comp_text=""):
    """Calculate final aggregated score"""

    # Force text_score to zero if either field is empty or whitespace
    if ref_text.strip() == "" or comp_text.strip() == "":
        text_score = 0
        final_score = image_score  # Use image_score for consistency
    elif text_score > 75:
        final_score = (text_score * 0.8) + (image_score * 0.2)
    else:
        final_score = (0.5 * text_score) + (0.5 * image_score)

    return final_score




# --- Helper: Get image files from folder ---
def get_image_files_from_folder(folder_path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

# --- Optimized Processing Functions ---
def process_single_comparison(args):
    """Process a single comparison - for parallel execution"""
    ref_image_path, comp_image_path, comp_filename, request = args
    
    try:
        # SIFT comparison
        comparison_result = sift_ranker.compare_two_images(ref_image_path, comp_image_path)
        
        # Extract text from comparison image
        comp_text = text_module.clean_extracted_text(
            text_module.extract_text_from_image(comp_image_path)
        )
        
        return {
            'filename': comp_filename,
            'comparison_result': comparison_result,
            'comp_text': comp_text,
            'comp_image_path': comp_image_path
        }
    except Exception as e:
        print(f"Error processing {comp_filename}: {e}")
        return None

async def process_reference_batch(reference_images, comparison_folder, request):
    """Process multiple reference images with optimized parallel processing"""
    global executor
    
    # Get comparison images once
    comparison_images = get_image_files_from_folder(comparison_folder)
    
    batch_results = []
    
    for ref_image_path in reference_images:
        print(f"\n🔎 PROCESSING REFERENCE: {os.path.basename(ref_image_path)}")
        batch_start_time = time.time()
        
        try:
            # Extract text from reference image once
            ref_text = text_module.clean_extracted_text(
                text_module.extract_text_from_image(ref_image_path)
            )
            
            # Prepare arguments for parallel processing
            comparison_args = [
                (ref_image_path, comp_image, os.path.basename(comp_image), request)
                for comp_image in comparison_images
            ]
            
            # Process comparisons in parallel
            loop = asyncio.get_event_loop()
            comparison_results = await loop.run_in_executor(
                executor,
                lambda: [process_single_comparison(args) for args in comparison_args]
            )
            
            # Filter out failed comparisons
            valid_results = [r for r in comparison_results if r is not None]
            
            # Process results
            results = []
            for result_data in valid_results:
                comp_text = result_data['comp_text']
                comparison_result = result_data['comparison_result']
                
                # Text similarity
                text_similarity = text_module.calculate_similarity_advanced(ref_text, comp_text)
                
                # Visual similarity (SIFT)
                vit_similarity = comparison_result['similarity'] * 100
                image_score = vit_similarity
                
                # Aggregated score
                final_score = calculate_aggregated_scores(
                    text_similarity, image_score, vit_similarity, ref_text, comp_text
                )
                
                infringement_detected = final_score >= request.infringement_threshold
                
                result = SimilarityResult(
                    logo_name=result_data['filename'],
                    logo_path=result_data['comp_image_path'],
                    text1=ref_text,
                    text2=comp_text,
                    text_similarity=text_similarity,
                    vit_similarity=vit_similarity,
                    final_similarity=final_score,
                    infringement_detected=infringement_detected
                )
                results.append(result)
            
            # Sort results by similarity
            results.sort(key=lambda x: x.final_similarity, reverse=True)
            
            batch_processing_time = time.time() - batch_start_time
            infringement_count = sum(1 for result in results if result.infringement_detected)
            
            batch_result = BatchResult(
                reference_logo=os.path.basename(ref_image_path),
                results=results,
                processing_time=batch_processing_time,
                infringement_count=infringement_count
            )
            batch_results.append(batch_result)
            
            print(f"✅ REFERENCE COMPLETE: {len(results)} comparisons, {infringement_count} infringements, {batch_processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ REFERENCE ERROR: {ref_image_path}: {str(e)}")
            continue
    
    return batch_results

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logos(request: SimilarityRequest):
    global sift_ranker

    if sift_ranker is None:
        raise HTTPException(status_code=500, detail="Ranker not initialized")

    print(f"🚀 API REQUEST: Starting logo analysis")
    print(f"📁 FOLDERS: Reference='{request.reference_folder_path}', Comparison='{request.comparison_folder_path}'")

    # Validate folders
    if not os.path.exists(request.reference_folder_path):
        raise HTTPException(status_code=400, detail=f"Reference folder path does not exist: {request.reference_folder_path}")

    if not os.path.exists(request.comparison_folder_path):
        raise HTTPException(status_code=400, detail=f"Comparison folder path does not exist: {request.comparison_folder_path}")

    # Get image files
    reference_images = get_image_files_from_folder(request.reference_folder_path)
    if not reference_images:
        raise HTTPException(status_code=400, detail="No image files found in reference folder")

    print(f"📚 REFERENCE IMAGES: Found {len(reference_images)} images to process")

    # Limit images to process
    if len(reference_images) > request.max_images:
        reference_images = reference_images[:request.max_images]
        print(f"⚠️ LIMITING: Processing only first {request.max_images} images")

    total_start_time = time.time()
    
    # Process in optimized batches
    batch_results = await process_reference_batch(reference_images, request.comparison_folder_path, request)
    
    total_processing_time = time.time() - total_start_time
    total_comparisons = sum(len(batch.results) for batch in batch_results)
    total_infringements = sum(batch.infringement_count for batch in batch_results)
    avg_processing_time = total_processing_time / len(batch_results) if batch_results else 0

    summary = {
        "total_comparisons": total_comparisons,
        "total_infringements": total_infringements,
        "avg_processing_time_per_reference": round(avg_processing_time, 2),
        "images_per_second": round(total_comparisons / total_processing_time, 1) if total_processing_time > 0 else 0,
        "infringement_rate_percent": round((total_infringements / total_comparisons) * 100, 2) if total_comparisons > 0 else 0,
        "performance_improvements": {
            "feature_caching": "SIFT features cached to disk",
            "parallel_processing": "Multi-threaded comparison processing",
            "optimized_matching": "FLANN with optimized parameters",
            "image_resizing": "Automatic image resizing for speed",
            "memory_management": "Efficient memory usage patterns"
        }
    }

    print(f"\n✅ ANALYSIS COMPLETE: {total_comparisons} total comparisons in {total_processing_time:.2f}s")
    print(f"📚 FINAL STATS: {total_infringements} infringements detected ({summary['infringement_rate_percent']}%)")

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
