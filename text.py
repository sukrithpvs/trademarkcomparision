import argparse
import requests
import base64
from PIL import Image
import io
import jellyfish
import os
import json
import hashlib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# Cache directory for storing OCR results
CACHE_DIR = "ocr_cache"

# Global executor for async operations
text_executor = None

def get_text_executor():
    """Get or create the text processing executor"""
    global text_executor
    if text_executor is None:
        text_executor = ThreadPoolExecutor(max_workers=4)
    return text_executor

def ensure_cache_directory():
    """Create cache directory if it doesn't exist"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_file_hash(image_path):
    """Generate a unique hash for the image file based on path and modification time"""
    stat = os.stat(image_path)
    file_info = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(file_info.encode()).hexdigest()

def get_cache_path(image_path):
    """Get the cache file path for an image"""
    file_hash = get_file_hash(image_path)
    return os.path.join(CACHE_DIR, f"{file_hash}.json")

def load_cached_text(image_path):
    """Load cached OCR result if it exists and is valid"""
    cache_path = get_cache_path(image_path)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Verify the cached data is for the same file
        if cache_data.get('image_path') == image_path:
            return cache_data.get('extracted_text', '')
    except (json.JSONDecodeError, KeyError):
        # If cache file is corrupted, remove it
        try:
            os.remove(cache_path)
        except:
            pass
    
    return None

def save_cached_text(image_path, extracted_text):
    """Save OCR result to cache"""
    ensure_cache_directory()
    cache_path = get_cache_path(image_path)
    
    cache_data = {
        'image_path': image_path,
        'extracted_text': extracted_text,
        'timestamp': datetime.now().isoformat(),
        'file_hash': get_file_hash(image_path)
    }
    
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Warning: Could not save cache for {image_path}: {e}")

# --- CLEANING FUNCTION ---
def clean_extracted_text(text):
    """Clean extracted text and handle null/empty cases"""
    # Handle None values
    if text is None:
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Return empty string if text is empty, just quotes, or whitespace
    if not text or text in ['""', "''", '""', "''", '""""', "''''"] or text.isspace():
        return ""
    
    # Remove common OCR prefixes
    for prefix in ["The text in the image reads:", "The text in the image is:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    
    # Final check after cleaning
    if not text or text.isspace():
        return ""
    
    return text

# --- SIMILARITY FUNCTION ---
def calculate_similarity_advanced(text1, text2):
    """
    Enhanced similarity calculation with proper null/empty handling
    Returns 0.0 if either text is null, empty, or contains only whitespace
    """
    def longest_common_suffix(w1, w2):
        i = 0
        min_len = min(len(w1), len(w2))
        while i < min_len and w1[-(i+1)] == w2[-(i+1)]:
            i += 1
        return i

    def longest_common_prefix(w1, w2):
        i = 0
        min_len = min(len(w1), len(w2))
        while i < min_len and w1[i] == w2[i]:
            i += 1
        return i

    def preprocess_text(text):
        if text is None:
            return ""
        text = str(text).strip()
        return ' '.join(text.split())

    def calculate_character_overlap(w1, w2):
        """Calculate character overlap percentage"""
        if not w1 or not w2:
            return 0.0
        
        chars1 = set(w1.lower())
        chars2 = set(w2.lower())
        
        if not chars1 or not chars2:
            return 0.0
            
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return (intersection / union) * 100 if union > 0 else 0.0

    # Check for null/empty texts first - CRITICAL CHECK
    if text1 is None or text2 is None:
        return 0.0
    
    # Clean and preprocess both texts
    cleaned_text1 = preprocess_text(clean_extracted_text(text1))
    cleaned_text2 = preprocess_text(clean_extracted_text(text2))

    # Return 0 if either text is empty after cleaning - CRITICAL CHECK
    if not cleaned_text1 or not cleaned_text2:
        return 0.0
    
    # Additional check for texts that are just whitespace or meaningless
    if len(cleaned_text1.strip()) == 0 or len(cleaned_text2.strip()) == 0:
        return 0.0

    # Split into words and check again
    words1 = cleaned_text1.split()
    words2 = cleaned_text2.split()

    if not words1 or not words2:
        return 0.0

    # Calculate overall text similarity first
    overall_jaro = jellyfish.jaro_winkler_similarity(cleaned_text1, cleaned_text2) * 100
    
    # Word-level similarities
    word_similarities = []
    
    for w1 in words1:
        best_match_for_w1 = 0.0
        
        for w2 in words2:
            current_sim = 0.0
            
            # Exact match gets high score but not 100% unless texts are very similar
            if w1 == w2:
                if len(words1) == 1 and len(words2) == 1:
                    current_sim = 95.0  # Single word exact match
                elif len(set(words1).intersection(set(words2))) / max(len(set(words1)), len(set(words2))) > 0.7:
                    current_sim = 90.0  # Most words match
                else:
                    current_sim = 70.0  # Some words match but overall different
            
            # Jaro-Winkler similarity for longer words
            elif len(w1) >= 4 and len(w2) >= 4:
                jaro_sim = jellyfish.jaro_winkler_similarity(w1, w2)
                if jaro_sim >= 0.8:
                    current_sim = max(current_sim, jaro_sim * 85)
                elif jaro_sim >= 0.6:
                    current_sim = max(current_sim, jaro_sim * 70)
            
            # Enhanced character overlap for all word lengths
            char_overlap = calculate_character_overlap(w1, w2)
            if char_overlap >= 60:
                current_sim = max(current_sim, char_overlap * 0.8)
            elif char_overlap >= 40:
                current_sim = max(current_sim, char_overlap * 0.6)
            
            # Suffix similarity
            suffix_len = longest_common_suffix(w1, w2)
            if suffix_len >= 3:
                min_word_len = min(len(w1), len(w2))
                suffix_sim = (suffix_len / min_word_len) * 70
                current_sim = max(current_sim, suffix_sim)

            # Prefix similarity
            prefix_len = longest_common_prefix(w1, w2)
            if prefix_len >= 3:
                min_word_len = min(len(w1), len(w2))
                prefix_sim = (prefix_len / min_word_len) * 70
                current_sim = max(current_sim, prefix_sim)

            # Enhanced Levenshtein for shorter words
            if 3 <= len(w1) <= 10 and 3 <= len(w2) <= 10:
                lev_dist = jellyfish.levenshtein_distance(w1, w2)
                max_len = max(len(w1), len(w2))
                
                if lev_dist <= 2 and max_len >= 5:
                    lev_sim = ((max_len - lev_dist) / max_len) * 75
                    current_sim = max(current_sim, lev_sim)
                elif lev_dist <= 1:
                    lev_sim = ((max_len - lev_dist) / max_len) * 80
                    current_sim = max(current_sim, lev_sim)
            
            # Containment check (one word contains another)
            if len(w1) >= 4 and len(w2) >= 4:
                if w1 in w2 or w2 in w1:
                    containment_sim = (min(len(w1), len(w2)) / max(len(w1), len(w2))) * 60
                    current_sim = max(current_sim, containment_sim)
            
            best_match_for_w1 = max(best_match_for_w1, current_sim)
        
        word_similarities.append(best_match_for_w1)
    
    # Calculate final similarity
    if word_similarities:
        avg_word_sim = sum(word_similarities) / len(word_similarities)
        
        # Weight the final score considering both word-level and overall text similarity
        final_similarity = (avg_word_sim * 0.7) + (overall_jaro * 0.3)
        
        # Apply length penalty for very different text lengths
        len_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
        if len_ratio < 0.5:
            final_similarity *= (0.5 + len_ratio * 0.5)
        
        return round(min(final_similarity, 99.5), 2)  # Cap at 99.5% to avoid false 100%
    
    return 0.0

# --- DEBUG FUNCTION ---
def debug_similarity_calculation(text1, text2):
    """Debug function to see exactly what values are being compared"""
    print(f"🔍 DEBUG: Original text1: {repr(text1)}")
    print(f"🔍 DEBUG: Original text2: {repr(text2)}")
    
    cleaned1 = clean_extracted_text(text1)
    cleaned2 = clean_extracted_text(text2)
    
    print(f"🔍 DEBUG: Cleaned text1: {repr(cleaned1)}")
    print(f"🔍 DEBUG: Cleaned text2: {repr(cleaned2)}")
    print(f"🔍 DEBUG: Text1 is empty: {not cleaned1}")
    print(f"🔍 DEBUG: Text2 is empty: {not cleaned2}")
    
    similarity = calculate_similarity_advanced(text1, text2)
    print(f"🔍 DEBUG: Final similarity: {similarity}")
    
    return similarity

# --- IMAGE TO TEXT EXTRACTION ---
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_text_from_image(image_path):
    """Extract text from image with optimized caching support"""
    # First, try to load from cache
    cached_text = load_cached_text(image_path)
    if cached_text is not None:
        return cached_text
    
    try:
        # Load and encode image
        image = Image.open(image_path)
        
        # Optimize image size for faster API processing
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        image_b64 = encode_image_to_base64(image)
        
        # Prepare API request
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image and just give in a single line avoid any other info or explanation strictly. If there is no text in the image, just return empty string,example = \"\""},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0
        }
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make API call with timeout
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            extracted_text = data['choices'][0]['message']['content']
            
            # Save to cache
            save_cached_text(image_path, extracted_text)
            
            return extracted_text
        else:
            error_msg = f"Error: {response.status_code}\n{response.text}"
            print(f"❌ API Error: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error processing image {image_path}: {str(e)}"
        print(f"❌ Processing Error: {error_msg}")
        return error_msg

# --- ASYNC BATCH PROCESSING ---
async def extract_text_batch(image_paths, max_workers=4):
    """Extract text from multiple images concurrently"""
    executor = get_text_executor()
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(executor, extract_text_from_image, img_path)
        for img_path in image_paths
    ]
    
    return await asyncio.gather(*tasks)

def clear_cache():
    """Clear all cached OCR results"""
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"🗑️ Cleared OCR cache directory: {CACHE_DIR}")
    else:
        print(f"📁 Cache directory {CACHE_DIR} doesn't exist")

def cache_stats():
    """Display cache statistics"""
    if not os.path.exists(CACHE_DIR):
        print(f"📁 Cache directory {CACHE_DIR} doesn't exist")
        return
    
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
    
    print(f"📊 OCR Cache Statistics:")
    print(f"   📁 Cache directory: {CACHE_DIR}")
    print(f"   📄 Cached files: {len(cache_files)}")
    print(f"   💾 Total size: {total_size / 1024:.2f} KB")

# --- MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser(
        description='Compare text similarity between two images using Groq OCR with caching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text.py image1.jpg image2.png
  python text.py /path/to/first.jpeg /path/to/second.jpg --verbose
  python text.py --clear-cache
  python text.py --cache-stats
        """
    )
    parser.add_argument('image1', nargs='?', help='Path to the first image')
    parser.add_argument('image2', nargs='?', help='Path to the second image')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed output including raw extracted text')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Show debug information for similarity calculation')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached OCR results')
    parser.add_argument('--cache-stats', action='store_true',
                       help='Show cache statistics')
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.clear_cache:
        clear_cache()
        return
    
    if args.cache_stats:
        cache_stats()
        return
    
    # Validate that image arguments are provided
    if not args.image1 or not args.image2:
        parser.print_help()
        return
    
    # Validate image files exist
    for img_path in [args.image1, args.image2]:
        if not os.path.exists(img_path):
            print(f"❌ Error: Image file '{img_path}' not found!")
            return
        if not os.path.isfile(img_path):
            print(f"❌ Error: '{img_path}' is not a file!")
            return
    
    print("🔍 Image Text Similarity Analyzer (with OCR Caching)")
    print("=" * 60)
    print(f"📷 Image 1: {args.image1}")
    print(f"📷 Image 2: {args.image2}")
    print()
    
    print("⏳ Extracting text from first image...")
    text1 = extract_text_from_image(args.image1)
    clean_text1 = clean_extracted_text(text1)
    
    print("⏳ Extracting text from second image...")
    text2 = extract_text_from_image(args.image2)
    clean_text2 = clean_extracted_text(text2)
    
    print("🧮 Calculating similarity...")
    
    # Use debug function if requested
    if args.debug:
        similarity = debug_similarity_calculation(text1, text2)
    else:
        similarity = calculate_similarity_advanced(text1, text2)
    
    # Display results
    print("\n" + "=" * 60)
    print("📄 EXTRACTED TEXT FROM FIRST IMAGE:")
    print("-" * 30)
    print(clean_text1 if clean_text1 else "(No text detected)")
    
    print("\n📄 EXTRACTED TEXT FROM SECOND IMAGE:")
    print("-" * 30)
    print(clean_text2 if clean_text2 else "(No text detected)")
    
    print(f"\n✅ TEXT SIMILARITY SCORE: {similarity:.2f}%")
    
    # Similarity interpretation
    if similarity >= 90:
        print("🟢 Very High Similarity - Texts are nearly identical")
    elif similarity >= 70:
        print("🟡 High Similarity - Texts are quite similar")
    elif similarity >= 50:
        print("🟠 Moderate Similarity - Some similarities found")
    elif similarity >= 20:
        print("🔴 Low Similarity - Few similarities found")
    else:
        print("⚫ Very Low Similarity - Texts appear different")
    
    # Verbose output
    if args.verbose:
        print("\n" + "=" * 60)
        print("🔍 VERBOSE OUTPUT:")
        print("-" * 20)
        print("Raw text from image 1:")
        print(repr(text1))
        print("\nRaw text from image 2:")
        print(repr(text2))
    
    # Show cache stats
    print(f"\n📊 Cache info: OCR results saved in '{CACHE_DIR}' directory")

if __name__ == "__main__":
    main()
