import cv2
import numpy as np
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

class ColorProcessor:
    def __init__(self, use_cuda=False):
        """Initialize color processor (CPU only)"""
        self.use_cuda = False
        print("Color processor initialized with CPU processing")
    
    def extract_dominant_colors(self, image_path, n_colors=8):
        """Extract dominant colors using CPU processing"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
            
            # Convert to RGB and resize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (200, 200))
            
            # Apply bilateral filter
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Reshape for clustering
            pixels = image.reshape(-1, 3)
            
            # Filter pixels
            pixel_sums = pixels.sum(axis=1)
            brightness_mask = (pixel_sums > 50) & (pixel_sums < 700)
            
            if brightness_mask.sum() > n_colors * 15:
                pixels = pixels[brightness_mask]
            
            # Perform clustering
            if len(pixels) > n_colors:
                kmeans = KMeans(
                    n_clusters=n_colors,
                    random_state=42,
                    n_init=15,
                    max_iter=300,
                    tol=1e-4
                )
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_
                
                labels = kmeans.labels_
                weights = np.bincount(labels, minlength=n_colors) / len(labels)
                
                # Sort by weight
                sorted_indices = np.argsort(weights)[::-1]
                colors = colors[sorted_indices]
                weights = weights[sorted_indices]
                
                # Filter out colors with very low weights
                significant_mask = weights > 0.03
                if significant_mask.sum() >= 3:
                    colors = colors[significant_mask]
                    weights = weights[significant_mask]
                    weights = weights / weights.sum()
                    
                    # Pad if necessary
                    if len(colors) < n_colors:
                        pad_size = n_colors - len(colors)
                        colors = np.vstack([colors, np.zeros((pad_size, 3))])
                        weights = np.concatenate([weights, np.zeros(pad_size)])
            else:
                colors = np.zeros((n_colors, 3))
                if len(pixels) > 0:
                    colors[:len(pixels)] = pixels[:n_colors]
                weights = np.ones(n_colors) / n_colors
            
            return colors.astype(np.uint8), weights
            
        except Exception as e:
            print(f"Color extraction failed for {image_path}: {e}")
            return np.zeros((n_colors, 3)), np.ones(n_colors) / n_colors
    
    def calculate_similarity_batch(self, ref_colors, ref_weights, batch_colors, batch_weights):
        """CPU-based batch color similarity calculation"""
        similarities = []
        for colors, weights in zip(batch_colors, batch_weights):
            sim = self._calculate_similarity_strict(ref_colors, ref_weights, colors, weights)
            similarities.append(sim)
        return similarities
    
    def _rgb_to_lab_accurate(self, rgb_color):
        """Convert RGB to LAB using colormath library"""
        try:
            rgb_normalized = rgb_color / 255.0
            rgb_color_obj = sRGBColor(
                rgb_normalized[0], 
                rgb_normalized[1], 
                rgb_normalized[2]
            )
            lab_color = convert_color(rgb_color_obj, LabColor)
            return np.array([lab_color.lab_l, lab_color.lab_a, lab_color.lab_b])
        except Exception:
            return np.array([50, 0, 0])
    
    def _calculate_similarity_strict(self, colors1, weights1, colors2, weights2):
        """Strict color similarity calculation with proper Delta E thresholds"""
        try:
            total_distance = 0
            total_weight = 0
            
            # Convert to LAB space
            lab_colors1 = [self._rgb_to_lab_accurate(color) for color in colors1]
            lab_colors2 = [self._rgb_to_lab_accurate(color) for color in colors2]
            
            # Calculate distances for significant colors only
            for i, (lab1, weight1) in enumerate(zip(lab_colors1, weights1)):
                if weight1 < 0.05:
                    continue
                
                min_delta_e = float('inf')
                
                for lab2 in lab_colors2:
                    try:
                        color1_lab = LabColor(lab1[0], lab1[1], lab1[2])
                        color2_lab = LabColor(lab2[0], lab2[1], lab2[2])
                        delta_e = delta_e_cie2000(color1_lab, color2_lab)
                        min_delta_e = min(min_delta_e, delta_e)
                    except:
                        euclidean_dist = np.linalg.norm(lab1 - lab2)
                        min_delta_e = min(min_delta_e, euclidean_dist)
                
                total_distance += min_delta_e * weight1
                total_weight += weight1
            
            if total_weight > 0:
                avg_distance = total_distance / total_weight
            else:
                return 0.0
            
            # Convert distance to similarity
            if avg_distance <= 1:
                similarity = 95 - (avg_distance * 5)
            elif avg_distance <= 3:
                similarity = 90 - ((avg_distance - 1) * 20)
            elif avg_distance <= 6:
                similarity = 50 - ((avg_distance - 3) * 10)
            elif avg_distance <= 10:
                similarity = 20 - ((avg_distance - 6) * 4)
            else:
                similarity = max(0, 4 - ((avg_distance - 10) * 0.5))
            
            return max(0, min(100, similarity))
                
        except Exception as e:
            print(f"Color similarity calculation failed: {e}")
            return 0.0
    
    def calculate_single_similarity(self, colors1, weights1, colors2, weights2):
        """Calculate similarity between two sets of colors"""
        return self._calculate_similarity_strict(colors1, weights1, colors2, weights2)
