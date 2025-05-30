import os
import time
import glob
import hashlib
import xxhash
import pickle
import asyncio
import mmap
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
from scipy.stats import wasserstein_distance
import faiss
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from sklearn.decomposition import PCA

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

class FileSystemOptimizer:
    def __init__(self):
        self.metadata_cache = {}
        self.inode_cache = {}
    
    def get_file_signature(self, file_path):
        """Get unique file signature without reading content"""
        try:
            stat = os.stat(file_path)
            signature = f"{stat.st_ino}_{stat.st_size}_{stat.st_mtime}"
            return signature
        except OSError:
            return None
    
    def batch_get_signatures(self, file_paths):
        """Get signatures for batch of files and deduplicate"""
        signatures = {}
        for path in file_paths:
            sig = self.get_file_signature(path)
            if sig:
                if sig not in signatures:
                    signatures[sig] = []
                signatures[sig].append(path)
        
        # Return only first occurrence of each signature
        unique_files = []
        duplicate_count = 0
        for sig, paths in signatures.items():
            unique_files.append(paths[0])
            duplicate_count += len(paths) - 1
        
        print(f"FileSystem dedup: removed {duplicate_count} duplicates, kept {len(unique_files)} unique files")
        return unique_files

def fast_file_hash(file_path, chunk_size=8192):
    """Memory-mapped file hashing for speed"""
    try:
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'rb') as f:
            # For small files, read directly
            if file_size <= chunk_size:
                return xxhash.xxh64(f.read()).hexdigest()
            
            # For larger files, use memory mapping
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                hasher = xxhash.xxh64()
                
                # Hash first chunk
                hasher.update(mm[:chunk_size])
                
                # Hash middle chunk
                if file_size > chunk_size * 2:
                    mid_pos = file_size // 2
                    hasher.update(mm[mid_pos:mid_pos + chunk_size])
                
                # Hash last chunk
                if file_size > chunk_size:
                    hasher.update(mm[-chunk_size:])
                
                return hasher.hexdigest()
    except Exception:
        return None

class OptimizedDuplicateDetector:
    def __init__(self):
        self.file_size_cache = {}
        self.quick_hash_cache = {}
        self.processed_files = set()
        self.hash_to_path = {}
        self.fs_optimizer = FileSystemOptimizer()
        
    def batch_deduplicate_files(self, file_paths):
        """Deduplicate entire batch before any processing"""
        print(f"Starting deduplication of {len(file_paths)} files...")
        
        # Stage 1: File system level deduplication (fastest)
        unique_files = self.fs_optimizer.batch_get_signatures(file_paths)
        
        if len(unique_files) == len(file_paths):
            print("No file system duplicates found")
            return unique_files
        
        # Stage 2: Group by file size for remaining files
        size_groups = {}
        for path in unique_files:
            try:
                size = os.path.getsize(path)
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(path)
            except OSError:
                continue
        
        # Stage 3: Process each size group
        final_unique = []
        total_duplicates = 0
        
        for size, paths in size_groups.items():
            if len(paths) == 1:
                final_unique.extend(paths)
            else:
                unique_in_group = self._deduplicate_size_group(paths)
                final_unique.extend(unique_in_group)
                total_duplicates += len(paths) - len(unique_in_group)
        
        print(f"Content dedup: removed {total_duplicates} additional duplicates")
        print(f"Final result: {len(final_unique)} unique files from {len(file_paths)} original files")
        return final_unique
    
    def _deduplicate_size_group(self, paths):
        """Deduplicate files with same size using progressive hashing"""
        unique_paths = []
        seen_hashes = set()
        
        for path in paths:
            # Use fast file hash
            file_hash = fast_file_hash(path)
            if file_hash and file_hash not in seen_hashes:
                unique_paths.append(path)
                seen_hashes.add(file_hash)
        
        return unique_paths

class MemoryEfficientFAISSIndex:
    def __init__(self, dimension=768, chunk_size=1000):
        self.dimension = dimension
        self.chunk_size = chunk_size
        self.index = None
        self.image_metadata = {}
        self.image_hashes = {}
        self.next_id = 0
        self.database_path = "faiss_logo_database"
        self.processed_paths = set()
        self.path_to_id = {}
        
        # Start with a simple flat index, upgrade to IVF when we have enough data
        self.index = faiss.IndexFlatIP(dimension)
        self.use_ivf = False
        self.min_vectors_for_ivf = 1000
        
        # Optimize FAISS threading
        self.optimize_faiss_threading()
        self.load_database()
    
    def optimize_faiss_threading(self):
        """Set optimal FAISS threading based on system"""
        cpu_count = os.cpu_count()
        optimal_threads = min(4, cpu_count // 2)
        
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
        faiss.omp_set_num_threads(optimal_threads)
        
        print(f"FAISS threading optimized: using {optimal_threads} threads")
    
    def is_already_indexed(self, file_path):
        """Quick check if file is already in index"""
        normalized_path = os.path.normpath(file_path)
        return normalized_path in self.processed_paths
    
    def batch_filter_new_files(self, file_paths):
        """Filter out already indexed files - fastest possible check"""
        new_files = []
        already_indexed = 0
        
        for path in file_paths:
            normalized_path = os.path.normpath(path)
            if normalized_path not in self.processed_paths:
                new_files.append(path)
            else:
                already_indexed += 1
        
        print(f"FAISS filter: {already_indexed} already indexed, {len(new_files)} new files to process")
        return new_files
    
    def get_file_hash(self, file_path):
        """Generate hash for file to detect duplicates"""
        return fast_file_hash(file_path)
    
    def should_upgrade_to_ivf(self):
        """Check if we should upgrade to IVF index"""
        return (not self.use_ivf and 
                self.index.ntotal >= self.min_vectors_for_ivf)
    
    def upgrade_to_ivf_index(self):
        """Upgrade from flat index to IVF index when we have enough data"""
        if self.use_ivf or self.index.ntotal < self.min_vectors_for_ivf:
            return
        
        print(f"Upgrading to IVF index with {self.index.ntotal} vectors...")
        
        n_vectors = self.index.ntotal
        nlist = min(
            max(10, int(np.sqrt(n_vectors))),
            n_vectors // 4,
            256
        )
        
        # Get all vectors from current index
        all_vectors = np.zeros((n_vectors, self.dimension), dtype=np.float32)
        self.index.reconstruct_n(0, n_vectors, all_vectors)
        
        # Create new IVF index
        quantizer = faiss.IndexFlatIP(self.dimension)
        new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Train with all available data
        new_index.train(all_vectors)
        new_index.add(all_vectors)
        
        # Replace old index
        self.index = new_index
        self.use_ivf = True
        
        print(f"Successfully upgraded to IVF index with nlist={nlist}")
    
    def add_vectors(self, vectors, image_paths):
        """Add vectors to FAISS index with metadata and track processed paths"""
        new_vectors = []
        new_metadata = []
        
        for vector, path in zip(vectors, image_paths):
            # Normalize vector for cosine similarity
            normalized_vector = vector / np.linalg.norm(vector)
            new_vectors.append(normalized_vector)
            
            # Store metadata
            metadata = {
                'path': path,
                'filename': os.path.basename(path),
                'index_id': self.next_id
            }
            
            self.image_metadata[self.next_id] = metadata
            self.processed_paths.add(os.path.normpath(path))
            self.path_to_id[os.path.normpath(path)] = self.next_id
            new_metadata.append(metadata)
            self.next_id += 1
        
        if new_vectors:
            vectors_array = np.array(new_vectors).astype('float32')
            
            # For IVF indices, train only if not already trained
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if len(vectors_array) >= getattr(self.index, 'nlist', 1):
                    self.index.train(vectors_array)
                else:
                    print(f"Not enough vectors ({len(vectors_array)}) to train IVF index")
                    return 0
            
            self.index.add(vectors_array)
            print(f"Added {len(new_vectors)} new vectors to FAISS index")
            
            # Check if we should upgrade to IVF
            if self.should_upgrade_to_ivf():
                self.upgrade_to_ivf_index()
        
        return len(new_vectors)
    
    def search_similar(self, query_vector, k=10):
        """Search for similar vectors in FAISS index"""
        if self.index.ntotal == 0:
            return []
        
        # Set nprobe for IVF indices
        if self.use_ivf and hasattr(self.index, 'nprobe'):
            nlist = getattr(self.index, 'nlist', 32)
            self.index.nprobe = min(max(1, nlist // 4), 32)
        
        # Normalize query vector
        normalized_query = query_vector / np.linalg.norm(query_vector)
        query_array = np.array([normalized_query]).astype('float32')
        
        # Limit search results for performance
        effective_k = min(k, self.index.ntotal, 100)
        
        # Search
        scores, indices = self.index.search(query_array, effective_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.image_metadata:
                metadata = self.image_metadata[idx]
                results.append({
                    'path': metadata['path'],
                    'filename': metadata['filename'],
                    'similarity_score': float(score),
                    'index_id': idx
                })
        
        return results
    
    def save_database(self):
        """Save FAISS index and metadata to disk"""
        os.makedirs(self.database_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.database_path, "faiss_index.bin"))
        
        # Save metadata
        metadata_file = os.path.join(self.database_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'image_metadata': self.image_metadata,
                'image_hashes': self.image_hashes,
                'next_id': self.next_id,
                'use_ivf': self.use_ivf,
                'processed_paths': self.processed_paths,
                'path_to_id': self.path_to_id
            }, f)
        
        print(f"Database saved to {self.database_path}")
    
    def load_database(self):
        """Load FAISS index and metadata from disk"""
        index_file = os.path.join(self.database_path, "faiss_index.bin")
        metadata_file = os.path.join(self.database_path, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.image_metadata = data['image_metadata']
                    self.image_hashes = data.get('image_hashes', {})
                    self.next_id = data['next_id']
                    self.use_ivf = data.get('use_ivf', hasattr(self.index, 'nlist'))
                    self.processed_paths = data.get('processed_paths', set())
                    self.path_to_id = data.get('path_to_id', {})
                
                index_type = "IVF" if self.use_ivf else "Flat"
                print(f"Loaded {index_type} database with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading database: {e}")
                # Fallback to flat index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.use_ivf = False
    
    def get_stats(self):
        """Get database statistics"""
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'IndexIVFFlat' if self.use_ivf else 'IndexFlatIP',
            'unique_images': len(self.image_metadata),
            'processed_paths': len(self.processed_paths)
        }
        
        if self.use_ivf and hasattr(self.index, 'nlist'):
            stats['nlist'] = self.index.nlist
            stats['nprobe'] = getattr(self.index, 'nprobe', 'default')
        
        return stats

class AsyncBatchProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.duplicate_detector = OptimizedDuplicateDetector()

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