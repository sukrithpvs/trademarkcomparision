import os
import time
import pickle
import numpy as np
import faiss
import xxhash
import mmap
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio

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
