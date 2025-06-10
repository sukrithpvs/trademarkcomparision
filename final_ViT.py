

import torch
import cv2
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import argparse
from scipy.stats import wasserstein_distance

def convert_to_grayscale(image):
    """Convert RGB image to grayscale and back to RGB format for ViT compatibility"""
    # Convert to grayscale using the standard luminance formula
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    
    # Convert back to 3-channel format by duplicating the grayscale channel
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    
    return gray_rgb.astype(np.uint8)

def limited_flip_augment(image):
    """Generate limited, logical augmentations for logo comparison"""
    flipped = [image]  # Original image
    
    # Only include transformations that make sense for logos
    flipped.append(cv2.flip(image, 1))   # Horizontal flip (mirror) - common for logos
    
    # Small rotations (logos might be slightly rotated)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    for angle in [90, 180, 270]:  # Only 90-degree rotations
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        flipped.append(rotated)
    
    return flipped

def extract_color_histogram(image):
    """Extract color histogram features"""
    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
    
    # Normalize histograms
    hist_h = hist_h.flatten() / np.sum(hist_h)
    hist_s = hist_s.flatten() / np.sum(hist_s)
    hist_v = hist_v.flatten() / np.sum(hist_v)
    
    return np.concatenate([hist_h, hist_s, hist_v])

def extract_edge_features(image):
    """Extract edge-based features"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate edge density and distribution
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Edge orientation histogram
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(sobely, sobelx)
    
    # Create orientation histogram
    hist, _ = np.histogram(orientation.flatten(), bins=8, range=(-np.pi, np.pi))
    hist = hist / np.sum(hist)
    
    return np.concatenate([[edge_density], hist])

def color_profile_sanity_check(img1, img2, threshold=0.3):
    """Check if images have fundamentally different color profiles"""
    hist1 = extract_color_histogram(img1)
    hist2 = extract_color_histogram(img2)
    
    # Calculate Wasserstein distance between color histograms
    color_distance = wasserstein_distance(hist1, hist2)
    
    # If color profiles are too different, flag as dissimilar
    if color_distance > threshold:
        print(f"Color profile check: Images have significantly different color profiles (distance: {color_distance:.4f})")
        return False
    
    return True

def structural_layout_check(img1, img2, threshold=0.4):
    """Check if images have fundamentally different structural layouts"""
    edge1 = extract_edge_features(img1)
    edge2 = extract_edge_features(img2)
    
    # Calculate structural similarity based on edge features
    structural_sim = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
    
    if structural_sim < threshold:
        print(f"Structural layout check: Images have significantly different layouts (similarity: {structural_sim:.4f})")
        return False
    
    return True

def normalize_features(features):
    """Normalize feature vectors to unit length"""
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm

def conservative_similarity_calculation(features1_list, features2_list, min_consensus=3):
    """
    Conservative similarity calculation using median and consensus approaches
    """
    # Normalize all feature vectors
    normalized_features1 = [normalize_features(f.flatten()) for f in features1_list]
    normalized_features2 = [normalize_features(f.flatten()) for f in features2_list]
    
    similarities = []
    
    for f1 in normalized_features1:
        for f2 in normalized_features2:
            # Cosine similarity
            cosine_sim = np.dot(f1, f2)
            similarities.append(cosine_sim)
    
    similarities = np.array(similarities)
    
    # Conservative metrics
    median_sim = np.median(similarities)
    mean_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    
    # Count how many similarities exceed a reasonable threshold
    high_sim_count = np.sum(similarities > 0.6)
    consensus_ratio = high_sim_count / len(similarities)
    
    # Require consensus: multiple variants should show high similarity
    if high_sim_count < min_consensus:
        print(f"Consensus check: Only {high_sim_count} out of {len(similarities)} comparisons show high similarity")
        vit_score = median_sim  # Use conservative median
    else:
        vit_score = max_sim  # Use max only if there's consensus
    
    return {
        'median_similarity': median_sim,
        'mean_similarity': mean_sim,
        'max_similarity': max_sim,
        'consensus_ratio': consensus_ratio,
        'vit_score': vit_score
    }

def calculate_discriminative_features_similarity(img1, img2):
    """Calculate similarity using discriminative features (color + edge)"""
    # Color histogram similarity
    hist1 = extract_color_histogram(img1)
    hist2 = extract_color_histogram(img2)
    color_sim = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    
    # Edge feature similarity
    edge1 = extract_edge_features(img1)
    edge2 = extract_edge_features(img2)
    edge_sim = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
    
    # Combined discriminative score
    discriminative_score = (color_sim + edge_sim) / 2
    
    return {
        'color_similarity': color_sim,
        'edge_similarity': edge_sim,
        'discriminative_score': discriminative_score
    }

def main(img1_path, img2_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained ViT
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    
    # Process images
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    img2 = np.array(Image.open(img2_path).convert("RGB"))
    
    print("=== SANITY CHECKS ===")
    
    # Sanity Check 1: Color profile compatibility
    color_compatible = color_profile_sanity_check(img1, img2)
    
    # Sanity Check 2: Structural layout compatibility
    structure_compatible = structural_layout_check(img1, img2)
    
    if not color_compatible or not structure_compatible:
        print("WARNING: Images failed sanity checks - they may be fundamentally different")
    
    # Calculate discriminative features similarity
    print("\n=== DISCRIMINATIVE FEATURES ANALYSIS ===")
    discriminative_results = calculate_discriminative_features_similarity(img1, img2)
    print(f"Color Histogram Similarity: {discriminative_results['color_similarity']:.4f}")
    print(f"Edge Features Similarity: {discriminative_results['edge_similarity']:.4f}")
    print(f"Combined Discriminative Score: {discriminative_results['discriminative_score']:.4f}")
    
    # Convert to grayscale for ViT processing
    img1_gray = convert_to_grayscale(img1)
    img2_gray = convert_to_grayscale(img2)
    
    # Generate LIMITED variants (no color inversion, limited transformations)
    print("\n=== LIMITED AUGMENTATION ANALYSIS ===")
    print("Generating limited augmented variants...")
    img1_variants = limited_flip_augment(img1_gray)
    img2_variants = limited_flip_augment(img2_gray)
    print(f"Generated {len(img1_variants)} variants for comparison image")
    print(f"Generated {len(img2_variants)} variants for reference image")
    
    # Extract features
    print("Extracting ViT features...")
    features1 = []
    for variant in img1_variants:
        inputs = feature_extractor(images=variant, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        features1.append(outputs.last_hidden_state[:,0].cpu().numpy())
    
    features2 = []
    for variant in img2_variants:
        inputs = feature_extractor(images=variant, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        features2.append(outputs.last_hidden_state[:,0].cpu().numpy())
    
    # Conservative similarity calculation
    print("Calculating conservative similarity scores...")
    vit_results = conservative_similarity_calculation(features1, features2)
    
    # Original similarity (no augmentation)
    f1_orig = normalize_features(features1[0].flatten())
    f2_orig = normalize_features(features2[0].flatten())
    original_sim = np.dot(f1_orig, f2_orig)
    
    
    print(f"ViT Final Score: {vit_results['vit_score']:.4f}")
    
  
    
  
    
    
    
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", type=str, help="Path to comparison image")
    parser.add_argument("img2", type=str, help="Path to reference image")
    args = parser.parse_args()
    main(args.img1, args.img2)

