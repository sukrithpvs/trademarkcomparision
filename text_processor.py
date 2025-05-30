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
        """Calculate text similarity using Levenshtein distance with substring bonus"""
        if not text1 or not text2:
            return 0.0
        
        # Check for substring relationships
        shorter_text = text1 if len(text1) <= len(text2) else text2
        longer_text = text2 if len(text1) <= len(text2) else text1
        
        # If one string is contained in the other, give high similarity
        if shorter_text in longer_text:
            # Calculate similarity based on how much of the longer string is the shorter one
            containment_ratio = len(shorter_text) / len(longer_text)
            return 85 + (containment_ratio * 15)  # 85-100% similarity for containment
        
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
            """Advanced text similarity with preprocessing and word-level analysis"""
            text1 = self.preprocess_text(text1)
            text2 = self.preprocess_text(text2)
            
            if not text1 or not text2:
                return 0.0
            
            # Split into words for better analysis
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            # Calculate word-level Jaccard similarity
            if words1 and words2:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                jaccard_sim = (intersection / union) * 100 if union > 0 else 0
            else:
                jaccard_sim = 0
            
            # Check for substring relationships at word level
            word_containment_bonus = 0
            shorter_words = words1 if len(words1) <= len(words2) else words2
            longer_words = words2 if len(words1) <= len(words2) else words1
            
            if shorter_words.issubset(longer_words) and shorter_words:
                # All words from shorter text are in longer text
                word_containment_bonus = 30
            
            # Character-level similarity
            levenshtein_sim = self.calculate_similarity(text1, text2)
            
            # Jaro-Winkler similarity
            try:
                jaro_sim = jellyfish.jaro_winkler_similarity(text1, text2) * 100
            except:
                jaro_sim = 0.0
            
            # Combine similarities with word containment bonus
            final_similarity = (0.4 * levenshtein_sim + 
                            0.3 * jaro_sim + 
                            0.3 * jaccard_sim + 
                            word_containment_bonus)
            
            # Cap at 100%
            return min(final_similarity, 100.0)