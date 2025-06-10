import argparse
import torch
from PIL import Image
import jellyfish
import os
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- CONFIGURATION ---
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"  # 2B parameter model optimized for efficiency

# --- INITIALIZE MODEL ---
def initialize_smolvlm_model():
    """Initialize SmolVLM model for CPU inference"""
    print("🔄 Loading SmolVLM (2B) for CPU inference...")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        trust_remote_code=True
    )
    
    print("✅ SmolVLM model loaded on CPU")
    return processor, model

# --- CLEANING FUNCTION ---
def clean_extracted_text(text):
    if text is None:
        return ""
    text = text.strip()
    
    prefixes_to_remove = [
        "The text in the image reads:",
        "The text in the image is:",
        "The image contains the text:",
        "I can see the text:",
        "The text shown is:",
        "Text in image:",
        "Image text:",
        "The text visible in the image is:",
        "Looking at the image, the text says:"
    ]
    
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    
    return text

# --- SIMILARITY FUNCTION (same as before) ---
def calculate_similarity_advanced(text1, text2):
    """Calculate advanced similarity between two texts"""
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
        if not text:
            return ""
        text = text.lower().strip()
        return ' '.join(text.split())

    text1 = preprocess_text(clean_extracted_text(text1))
    text2 = preprocess_text(clean_extracted_text(text2))

    words1 = set(text1.split())
    words2 = set(text2.split())

    if words1 and words2 and words1.intersection(words2):
        return 100.0

    max_sim = 0.0

    for w1 in words1:
        for w2 in words2:
            if len(w1) >= 5 and len(w2) >= 5:
                sim = jellyfish.jaro_winkler_similarity(w1, w2)
                min_len = min(len(w1), len(w2))
                common_letters = sum((c in w2) for c in w1)
                if sim >= 0.85 and common_letters >= min_len // 2:
                    max_sim = max(max_sim, sim * 100)

            suffix_len = longest_common_suffix(w1, w2)
            if suffix_len >= 3:
                min_word_len = min(len(w1), len(w2))
                suffix_sim = (suffix_len / min_word_len) * 100
                max_sim = max(max_sim, suffix_sim)

            prefix_len = longest_common_prefix(w1, w2)
            if prefix_len >= 3:
                min_word_len = min(len(w1), len(w2))
                prefix_sim = (prefix_len / min_word_len) * 100
                max_sim = max(max_sim, prefix_sim)

            if 5 <= len(w1) <= 8 and 5 <= len(w2) <= 8:
                lev_dist = jellyfish.levenshtein_distance(w1, w2)
                if lev_dist <= 2:
                    lev_sim = ((min(len(w1), len(w2)) - lev_dist) / min(len(w1), len(w2))) * 100
                    max_sim = max(max_sim, lev_sim)

    return round(max_sim, 2) if max_sim > 0 else 0.0

# --- IMAGE TO TEXT EXTRACTION ---
def extract_text_from_image_smolvlm(image_path, processor, model):
    """Extract text using SmolVLM model"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Create conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract all visible text from this image. Return only the text content without any explanations or descriptions."}
                ]
            }
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        return generated_text
        
    except Exception as e:
        return f"Error processing image {image_path}: {str(e)}"

# --- MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser(
        description='Compare text similarity between two images using SmolVLM (CPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text2.py image1.jpg image2.png
  python text2.py /path/to/first.jpeg /path/to/second.jpg
        """
    )
    parser.add_argument('image1', help='Path to the first image')
    parser.add_argument('image2', help='Path to the second image')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed output including raw extracted text')
    
    args = parser.parse_args()
    
    # Validate image files exist
    for img_path in [args.image1, args.image2]:
        if not os.path.exists(img_path):
            print(f"❌ Error: Image file '{img_path}' not found!")
            return
        if not os.path.isfile(img_path):
            print(f"❌ Error: '{img_path}' is not a file!")
            return
    
    # Initialize model
    processor, model = initialize_smolvlm_model()
    
    print("🔍 Image Text Similarity Analyzer (SmolVLM CPU)")
    print("=" * 50)
    print(f"📷 Image 1: {args.image1}")
    print(f"📷 Image 2: {args.image2}")
    print()
    
    print("⏳ Extracting text from first image...")
    text1 = extract_text_from_image_smolvlm(args.image1, processor, model)
    clean_text1 = clean_extracted_text(text1)
    
    print("⏳ Extracting text from second image...")
    text2 = extract_text_from_image_smolvlm(args.image2, processor, model)
    clean_text2 = clean_extracted_text(text2)
    
    print("🧮 Calculating similarity...")
    similarity = calculate_similarity_advanced(text1, text2)
    
    # Display results
    print("\n" + "=" * 50)
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
        print("\n" + "=" * 50)
        print("🔍 VERBOSE OUTPUT:")
        print("-" * 20)
        print("Raw text from image 1:")
        print(repr(text1))
        print("\nRaw text from image 2:")
        print(repr(text2))

if __name__ == "__main__":
    main()
