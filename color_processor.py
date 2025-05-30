import cv2
import numpy as np
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

class ColorProcessor:
    def __init__(self, use_cuda=False):
        """Initialize color processor with human perception focus"""
        self.use_cuda = False
        print("Color processor initialized with human perception alignment")
    
    def extract_perceptual_colors(self, image_path, n_colors=6):
        """Extract colors with focus on human perception"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = image.shape
            
            # Resize for processing but keep aspect ratio consideration
            image = cv2.resize(image, (150, 150))
            
            # Apply edge-preserving filter to maintain important color boundaries
            image = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=0.4)
            
            # Convert to LAB for perceptual uniformity
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Reshape for clustering
            pixels = lab_image.reshape(-1, 3)
            
            # Remove extreme values (pure black/white) unless dominant
            l_channel = pixels[:, 0]
            valid_mask = (l_channel > 10) & (l_channel < 240)
            
            if valid_mask.sum() > len(pixels) * 0.3:  # If enough valid pixels
                pixels = pixels[valid_mask]
            
            # Perform clustering in LAB space
            if len(pixels) > n_colors:
                kmeans = KMeans(
                    n_clusters=n_colors,
                    random_state=42,
                    n_init=10,
                    max_iter=200
                )
                kmeans.fit(pixels)
                lab_colors = kmeans.cluster_centers_
                
                labels = kmeans.labels_
                weights = np.bincount(labels, minlength=n_colors) / len(labels)
                
                # Convert back to RGB
                rgb_colors = []
                for lab_color in lab_colors:
                    lab_reshaped = lab_color.reshape(1, 1, 3).astype(np.uint8)
                    rgb_color = cv2.cvtColor(lab_reshaped, cv2.COLOR_LAB2RGB)[0, 0]
                    rgb_colors.append(rgb_color)
                
                colors = np.array(rgb_colors)
                
                # Sort by perceptual importance (combination of weight and distinctiveness)
                importance_scores = []
                for i, (color, weight) in enumerate(zip(colors, weights)):
                    # Calculate distinctiveness from other colors
                    distinctiveness = 0
                    for j, other_color in enumerate(colors):
                        if i != j:
                            dist = np.linalg.norm(color.astype(float) - other_color.astype(float))
                            distinctiveness += dist
                    
                    # Combine weight and distinctiveness
                    importance = weight * 0.7 + (distinctiveness / len(colors)) * 0.3
                    importance_scores.append(importance)
                
                # Sort by importance
                sorted_indices = np.argsort(importance_scores)[::-1]
                colors = colors[sorted_indices]
                weights = weights[sorted_indices]
                
                # Filter out very similar colors
                filtered_colors = [colors[0]]
                filtered_weights = [weights[0]]
                
                for i in range(1, len(colors)):
                    is_distinct = True
                    for existing_color in filtered_colors:
                        color_dist = np.linalg.norm(colors[i].astype(float) - existing_color.astype(float))
                        if color_dist < 30:  # Too similar
                            is_distinct = False
                            break
                    
                    if is_distinct and len(filtered_colors) < n_colors:
                        filtered_colors.append(colors[i])
                        filtered_weights.append(weights[i])
                
                # Pad if necessary
                while len(filtered_colors) < n_colors:
                    filtered_colors.append(np.zeros(3))
                    filtered_weights.append(0.0)
                
                colors = np.array(filtered_colors[:n_colors])
                weights = np.array(filtered_weights[:n_colors])
                
                # Renormalize weights
                if weights.sum() > 0:
                    weights = weights / weights.sum()
            else:
                colors = np.zeros((n_colors, 3))
                weights = np.ones(n_colors) / n_colors
            
            return colors.astype(np.uint8), weights
            
        except Exception as e:
            print(f"Color extraction failed for {image_path}: {e}")
            return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
    
    def extract_dominant_colors(self, image_path, n_colors=6):
        """Wrapper for backward compatibility"""
        return self.extract_perceptual_colors(image_path, n_colors)
    
    def calculate_perceptual_similarity(self, colors1, weights1, colors2, weights2):
        """Calculate color similarity based on human perception research"""
        try:
            # Filter out zero-weight colors
            valid_colors1 = [(c, w) for c, w in zip(colors1, weights1) if w > 0.01 and not np.all(c == 0)]
            valid_colors2 = [(c, w) for c, w in zip(colors2, weights2) if w > 0.01 and not np.all(c == 0)]
            
            if not valid_colors1 or not valid_colors2:
                return 0.0
            
            # Convert to LAB for perceptual calculations
            lab_colors1 = []
            lab_weights1 = []
            for color, weight in valid_colors1:
                lab_color = self._rgb_to_lab_safe(color)
                lab_colors1.append(lab_color)
                lab_weights1.append(weight)
            
            lab_colors2 = []
            lab_weights2 = []
            for color, weight in valid_colors2:
                lab_color = self._rgb_to_lab_safe(color)
                lab_colors2.append(lab_color)
                lab_weights2.append(weight)
            
            # Normalize weights
            lab_weights1 = np.array(lab_weights1)
            lab_weights2 = np.array(lab_weights2)
            lab_weights1 = lab_weights1 / lab_weights1.sum()
            lab_weights2 = lab_weights2 / lab_weights2.sum()
            
            # Calculate weighted color distances using human perception model
            total_similarity = 0
            total_weight = 0
            
            for i, (lab1, weight1) in enumerate(zip(lab_colors1, lab_weights1)):
                best_match_similarity = 0
                
                for lab2 in lab_colors2:
                    # Calculate Delta E (perceptual color difference)
                    delta_e = self._calculate_delta_e_safe(lab1, lab2)
                    
                    # Convert Delta E to similarity using human perception curve
                    if delta_e <= 1:
                        similarity = 100  # Imperceptible difference
                    elif delta_e <= 2:
                        similarity = 95 - (delta_e - 1) * 10  # Just noticeable
                    elif delta_e <= 5:
                        similarity = 85 - (delta_e - 2) * 15  # Noticeable difference
                    elif delta_e <= 10:
                        similarity = 40 - (delta_e - 5) * 6   # Clear difference
                    elif delta_e <= 20:
                        similarity = 10 - (delta_e - 10) * 0.8  # Very different
                    else:
                        similarity = 0  # Completely different
                    
                    best_match_similarity = max(best_match_similarity, similarity)
                
                total_similarity += best_match_similarity * weight1
                total_weight += weight1
            
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
            else:
                final_similarity = 0.0
            
            return max(0, min(100, final_similarity))
            
        except Exception as e:
            print(f"Color similarity calculation failed: {e}")
            return 0.0
    
    def calculate_similarity_batch(self, ref_colors, ref_weights, batch_colors, batch_weights):
        """Calculate batch similarities with human perception"""
        similarities = []
        for colors, weights in zip(batch_colors, batch_weights):
            sim = self.calculate_perceptual_similarity(ref_colors, ref_weights, colors, weights)
            similarities.append(sim)
        return similarities
    
    def _rgb_to_lab_safe(self, rgb_color):
        """Safe RGB to LAB conversion"""
        try:
            rgb_color = np.clip(rgb_color, 0, 255)
            rgb_normalized = rgb_color / 255.0
            
            if np.all(rgb_normalized == 0):
                return np.array([0, 0, 0])
            
            rgb_color_obj = sRGBColor(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
            lab_color = convert_color(rgb_color_obj, LabColor)
            return np.array([lab_color.lab_l, lab_color.lab_a, lab_color.lab_b])
        except:
            # Fallback conversion
            rgb_norm = rgb_color / 255.0
            L = 0.299 * rgb_norm[0] + 0.587 * rgb_norm[1] + 0.114 * rgb_norm[2]
            a = (rgb_norm[0] - rgb_norm[1]) * 128
            b = (rgb_norm[1] - rgb_norm[2]) * 128
            return np.array([L * 100, a, b])
    
    def _calculate_delta_e_safe(self, lab1, lab2):
        """Safe Delta E calculation"""
        try:
            color1_lab = LabColor(lab1[0], lab1[1], lab1[2])
            color2_lab = LabColor(lab2[0], lab2[1], lab2[2])
            return delta_e_cie2000(color1_lab, color2_lab)
        except:
            # Fallback to Euclidean distance
            return np.linalg.norm(lab1 - lab2)
    
    def calculate_single_similarity(self, colors1, weights1, colors2, weights2):
        """Calculate similarity between two color sets"""
        return self.calculate_perceptual_similarity(colors1, weights1, colors2, weights2)
