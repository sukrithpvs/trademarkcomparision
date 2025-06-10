import os
import numpy as np
import cv2
from pathlib import Path
from rembg import remove
from PIL import Image
import io
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
from colorspacious import cspace_convert
from skimage.color import rgb2lab
import faiss
import pickle
import json
from tqdm import tqdm

class MainColorVectorProcessor:
    def __init__(self, num_main_colors=3, min_percentage=5.0):
        self.num_main_colors = num_main_colors
        self.min_percentage = min_percentage
        
    def remove_logo_background(self, image_path):
        """Remove background from logo using rembg library"""
        try:
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()
            
            output_data = remove(input_data)
            image = Image.open(io.BytesIO(output_data))
            image_array = np.array(image)
            
            return image_array
        except Exception as e:
            print(f"Error removing background from {image_path}: {e}")
            return None

    def calculate_color_distinctiveness(self, rgb_color):
        """Calculate how distinctive/vibrant a color is"""
        r, g, b = rgb_color
        
        # Calculate saturation in HSV space
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        # Calculate color variance
        color_variance = np.std([r, g, b]) / 255.0
        
        # Combine saturation and variance for distinctiveness score
        distinctiveness = (s * 0.7) + (color_variance * 0.3)
        
        return distinctiveness

    def rgb_to_lab(self, rgb_color):
        """Convert RGB to CIELAB color space"""
        try:
            rgb_normalized = np.array(rgb_color).reshape(1, 1, 3) / 255.0
            lab = rgb2lab(rgb_normalized)
            return lab[0, 0, :]
        except:
            # Fallback method
            rgb_normalized = np.array(rgb_color) / 255.0
            lab = cspace_convert(rgb_normalized, "sRGB1", "CIELab")
            return lab

    def rgb_to_hsv(self, rgb_color):
        """Convert RGB to HSV color space"""
        r, g, b = rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return np.array([h * 360, s * 100, v * 100])

    def extract_main_colors_only(self, image_array):
        """Extract only the main/dominant colors from the logo"""
        
        # Handle alpha channel if present
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            alpha_channel = image_array[:, :, 3]
            logo_mask = alpha_channel > 100
            
            if np.sum(logo_mask) == 0:
                return []
            
            logo_pixels = image_array[logo_mask][:, :3]
        else:
            # Enhanced background detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Multiple background detection methods
            white_mask = gray > 240
            black_mask = gray < 15
            edge_detected = cv2.Canny(gray, 50, 150)
            edge_mask = edge_detected > 0
            
            # Create composite mask for logo pixels
            background_mask = white_mask | black_mask
            logo_mask = ~background_mask | edge_mask
            
            if np.sum(logo_mask) == 0:
                return []
            
            logo_pixels = image_array[logo_mask]
        
        # Filter out near-white, near-black, and low-saturation colors
        color_filter = (
            (np.max(logo_pixels, axis=1) - np.min(logo_pixels, axis=1) > 30) &
            (np.mean(logo_pixels, axis=1) < 230) &
            (np.mean(logo_pixels, axis=1) > 25) &
            (np.std(logo_pixels, axis=1) > 10)
        )
        
        logo_pixels = logo_pixels[color_filter]
        
        if len(logo_pixels) == 0:
            return []
        
        # Use fixed number of clusters to get main colors only
        n_clusters = min(self.num_main_colors, len(logo_pixels) // 100)
        
        if n_clusters < 1:
            return []
        
        # Perform clustering to find main colors
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=20,
            max_iter=300
        )
        kmeans.fit(logo_pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        main_colors = []
        for i, color in enumerate(colors):
            percentage = (label_counts[i] / total_pixels) * 100
            
            if percentage >= self.min_percentage:
                lab_color = self.rgb_to_lab(color)
                hsv_color = self.rgb_to_hsv(color)
                
                main_colors.append({
                    'rgb': tuple(color),
                    'percentage': percentage,
                    'lab': lab_color,
                    'hsv': hsv_color,
                    'prominence_score': percentage * self.calculate_color_distinctiveness(color)
                })
        
        # Sort by prominence
        main_colors.sort(key=lambda x: x['prominence_score'], reverse=True)
        
        return main_colors[:self.num_main_colors]

    def create_color_feature_vector(self, main_colors):
        """Create a feature vector from main colors for FAISS storage"""
        # Fixed-size vector to store color features (13 features per color)
        feature_vector = np.zeros(self.num_main_colors * 13)
        
        for i, color in enumerate(main_colors):
            if i >= self.num_main_colors:
                break
                
            start_idx = i * 13
            
            # RGB values (normalized)
            feature_vector[start_idx:start_idx+3] = np.array(color['rgb']) / 255.0
            
            # LAB values (normalized)
            lab_normalized = color['lab'] / np.array([100, 127, 127])  # Typical LAB ranges
            feature_vector[start_idx+3:start_idx+6] = lab_normalized
            
            # HSV values (normalized)
            hsv_normalized = color['hsv'] / np.array([360, 100, 100])
            feature_vector[start_idx+6:start_idx+9] = hsv_normalized
            
            # Additional features
            feature_vector[start_idx+9] = color['percentage'] / 100.0  # Normalized percentage
            feature_vector[start_idx+10] = color['prominence_score'] / 100.0  # Normalized prominence
            feature_vector[start_idx+11] = self.calculate_color_distinctiveness(color['rgb'])
            feature_vector[start_idx+12] = 1.0  # Color presence indicator
        
        return feature_vector.astype(np.float32)

    def process_folder_images(self, folder_path, output_dir="vector_db"):
        """Process all images in folder and create FAISS index"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        image_files = [str(p) for p in Path(folder_path).rglob('*') 
                      if p.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process")
        
        # Storage for vectors and metadata
        vectors = []
        metadata = []
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Remove background and extract main colors
                image_array = self.remove_logo_background(img_path)
                if image_array is None:
                    continue
                
                main_colors = self.extract_main_colors_only(image_array)
                if not main_colors:
                    continue
                
                # Create feature vector
                feature_vector = self.create_color_feature_vector(main_colors)
                vectors.append(feature_vector)
                
                # Store metadata
                metadata.append({
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'main_colors': main_colors,
                    'num_main_colors': len(main_colors)
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not vectors:
            print("No valid vectors created!")
            return
        
        # Convert to numpy array
        vectors_array = np.array(vectors)
        
        # Create FAISS index
        dimension = vectors_array.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance for color similarity
        index.add(vectors_array)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(output_dir, "color_vectors.index"))
        
        # Save metadata
        with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save configuration
        config = {
            'num_main_colors': self.num_main_colors,
            'min_percentage': self.min_percentage,
            'total_images': len(metadata),
            'vector_dimension': dimension
        }
        
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Successfully processed {len(metadata)} images")
        print(f"FAISS index saved with {len(metadata)} vectors of dimension {dimension}")
        print(f"Files saved in: {output_dir}")

def main():
    print("Main Color Vector Preprocessing")
    print("="*50)
    
    # Get folder path from user
    folder_path = input("Enter the path to the folder containing images to process: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder path does not exist: {folder_path}")
        return
    
    # Get output directory
    output_dir = input("Enter output directory for vector database (default: vector_db): ").strip()
    if not output_dir:
        output_dir = "vector_db"
    
    # Get parameters
    try:
        num_main_colors = int(input("Number of main colors to analyze (default: 3): ") or "3")
        min_percentage = float(input("Minimum color coverage percentage (default: 5.0): ") or "5.0")
    except ValueError:
        print("Using default parameters: num_main_colors=3, min_percentage=5.0")
        num_main_colors = 3
        min_percentage = 5.0
    
    # Create processor and process images
    processor = MainColorVectorProcessor(num_main_colors, min_percentage)
    processor.process_folder_images(folder_path, output_dir)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()