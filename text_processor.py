import easyocr
import jellyfish
import concurrent.futures

class TextProcessor:
    def __init__(self, use_cuda=False):
        """Initialize text processor with CPU-only OCR"""
        self.use_cuda = False
        
        print('Loading OCR engine...')
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized with CPU processing")
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def extract_text(self, image_path):
        """Extract text from logo image using EasyOCR"""
        if self.reader is None:
            return ""
            
        try:
            results = self.reader.readtext(image_path)
            text = ' '.join([result[1] for result in results])
            return text.strip().lower()
        except Exception as e:
            print(f"Text extraction failed for {image_path}: {e}")
            return ""
    
    def extract_text_batch(self, image_paths, max_workers=4):
        """Extract text from multiple images in parallel (CPU optimized)"""
        if self.reader is None:
            return [""] * len(image_paths)
            
        def extract_single_text(path):
            try:
                results = self.reader.readtext(path)
                return ' '.join([result[1] for result in results]).strip().lower()
            except:
                return ""
        
        # Use fewer workers for CPU processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            texts = list(executor.map(extract_single_text, image_paths))
        
        return texts
    
    def calculate_similarity(self, text1, text2):
        """Calculate text similarity using Levenshtein distance"""
        if not text1 or not text2:
            return 0.0
            
        distance = jellyfish.levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 0.0
            
        similarity = (1 - (distance / max_len)) * 100
        return similarity
    
    def preprocess_text(self, text):
        """Preprocess text for better similarity calculation"""
        if not text:
            return ""
        
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def calculate_similarity_advanced(self, text1, text2):
        """Advanced text similarity with preprocessing"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        if not text1 or not text2:
            return 0.0
        
        # Use multiple similarity metrics
        levenshtein_sim = self.calculate_similarity(text1, text2)
        
        # Jaro-Winkler similarity
        try:
            jaro_sim = jellyfish.jaro_winkler_similarity(text1, text2) * 100
        except:
            jaro_sim = 0.0
        
        # Combine similarities (weighted average)
        final_similarity = 0.7 * levenshtein_sim + 0.3 * jaro_sim
        
        return final_similarity
