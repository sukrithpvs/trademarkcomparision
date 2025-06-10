import cv2
import numpy as np
import os
import pickle
import hashlib
from PIL import Image
import matplotlib.pyplot as plt

class SIFTLogoComparison:
    def __init__(self):
        """Initialize the SIFT-based logo comparison system with caching"""
        self.sift = cv2.SIFT_create()
        self.feature_cache = {}
        self.cache_dir = "sift_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Optimized FLANN parameters
        FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        self.search_params = dict(checks=32)

    def get_image_hash(self, image_path):
        """Generate hash for image caching"""
        stat = os.stat(image_path)
        return hashlib.md5(f"{image_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()

    def get_cached_features(self, image_path):
        """Load cached SIFT features if available"""
        image_hash = self.get_image_hash(image_path)
        cache_file = os.path.join(self.cache_dir, f"{image_hash}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                # Remove corrupted cache file
                os.remove(cache_file)
        return None

    def save_features_to_cache(self, image_path, keypoints, descriptors):
        """Save SIFT features to cache"""
        image_hash = self.get_image_hash(image_path)
        cache_file = os.path.join(self.cache_dir, f"{image_hash}.pkl")
        
        # Convert keypoints to serializable format
        kp_data = [(kp.pt, kp.angle, kp.response, kp.octave, kp.class_id, kp.size) for kp in keypoints]
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((kp_data, descriptors), f)
        except Exception as e:
            print(f"Warning: Could not cache features for {image_path}: {e}")

    def load_keypoints_from_cache(self, kp_data):
        """Convert cached keypoint data back to cv2.KeyPoint objects"""
        keypoints = []
        for pt, angle, response, octave, class_id, size in kp_data:
            kp = cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle, 
                             response=response, octave=octave, class_id=class_id)
            keypoints.append(kp)
        return keypoints

    def get_sift_features(self, image_path):
        """Get SIFT features with caching"""
        cached = self.get_cached_features(image_path)
        if cached:
            kp_data, descriptors = cached
            keypoints = self.load_keypoints_from_cache(kp_data)
            return keypoints, descriptors
        
        # Compute features if not cached
        _, gray = self.load_and_preprocess(image_path)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is not None and keypoints:
            self.save_features_to_cache(image_path, keypoints, descriptors)
        
        return keypoints, descriptors

    def load_and_preprocess(self, image_input, max_size=800):
        """Load and preprocess image with size optimization"""
        try:
            image = None
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image path does not exist: {image_input}")
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from {image_input}")
            else:
                if hasattr(image_input, 'seek'):
                    image_input.seek(0)
                image_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not decode image.")

            # Resize large images for faster processing
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to RGB for display/output and grayscale for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image_rgb, image_gray

        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise

    def sift_similarity(self, img1, img2):
        """Optimized SIFT similarity calculation with caching"""
        try:
            # Use cached features if available
            kp1, desc1 = self.get_sift_features(img1) if isinstance(img1, str) else self.get_sift_features_direct(img1)
            kp2, desc2 = self.get_sift_features(img2) if isinstance(img2, str) else self.get_sift_features_direct(img2)

            if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
                return 0.0, len(kp1) if kp1 is not None else 0, len(kp2) if kp2 is not None else 0, 0

            # Optimized FLANN matching
            flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test with early termination
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                        # Early termination for very high similarity
                        if len(good_matches) > 100:
                            break

            # Geometric verification using RANSAC
            MIN_MATCH_COUNT = 10
            if len(good_matches) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if mask is not None:
                    num_inliers = np.sum(mask)
                    normalization_factor = 50.0
                    similarity_score = num_inliers / normalization_factor
                    return min(1.0, similarity_score), len(kp1), len(kp2), int(num_inliers)

            return 0.0, len(kp1), len(kp2), 0

        except cv2.error as e:
            print(f"Warning: OpenCV error during SIFT similarity: {e}")
            return 0.0, 0, 0, 0
        except Exception as e:
            print(f"Error in SIFT similarity calculation: {str(e)}")
            raise

    def get_sift_features_direct(self, image_input):
        """Get SIFT features directly from image data (for non-file inputs)"""
        _, gray = self.load_and_preprocess(image_input)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def compare_two_images(self, img1_path, img2_path):
        """Compare two images and return detailed results dictionary"""
        similarity, kp1_count, kp2_count, inlier_matches = self.sift_similarity(img1_path, img2_path)
        return {
            'similarity': similarity,
            'keypoints_img1': kp1_count,
            'keypoints_img2': kp2_count,
            'inlier_matches': inlier_matches,
        }

    def compare_one_to_many(self, reference_image_path, image_folder_path):
        """Compare one reference image to all images in a folder with optimizations"""
        if not os.path.exists(reference_image_path):
            print(f"Error: Reference image path does not exist: {reference_image_path}")
            return []
        if not os.path.exists(image_folder_path):
            print(f"Error: Target folder path does not exist: {image_folder_path}")
            return []

        results = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(image_extensions)]

        if not image_files:
            print("Warning: No image files found in the specified folder!")
            return []
        
        total_files = len(image_files)
        print(f"Found {total_files} images to compare against '{os.path.basename(reference_image_path)}'.")

        # Pre-extract features for reference image
        ref_kp, ref_desc = self.get_sift_features(reference_image_path)
        if ref_desc is None:
            print("Warning: Could not extract features from reference image")
            return []

        for i, filename in enumerate(image_files):
            try:
                image_path = os.path.join(image_folder_path, filename)
                print(f'Processing [{i+1}/{total_files}]: {filename}...')
                
                comparison_result = self.compare_two_images(reference_image_path, image_path)
                
                result = {
                    'filename': filename,
                    'similarity_score': comparison_result['similarity'],
                    'keypoints_ref': comparison_result['keypoints_img1'],
                    'keypoints_target': comparison_result['keypoints_img2'],
                    'inlier_matches': comparison_result['inlier_matches'],
                }
                results.append(result)
                
            except Exception as e:
                print(f"Could not process {filename}. Reason: {str(e)}")
                continue
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print('Processing complete!')
        return results

def display_top_results(results, folder_path, ref_image_path):
    """Displays the top 20 matching images in a grid."""
    top_n = min(20, len(results))
    if top_n == 0:
        print("No results to display.")
        return

    cols = 5
    rows = (top_n + cols - 1) // cols 
    
    fig = plt.figure(figsize=(15, 3 * rows + 3))
    gs = fig.add_gridspec(rows + 1, cols)

    # Reference image subplot spanning the top row
    ref_img = cv2.imread(ref_image_path)
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ax_ref = fig.add_subplot(gs[0, :])
    ax_ref.imshow(ref_img_rgb)
    ax_ref.set_title(f"Reference Image: {os.path.basename(ref_image_path)}", fontsize=14, color='blue')
    ax_ref.axis('off')
    
    print(f"\nDisplaying top {top_n} matching images...")

    for i in range(top_n):
        res = results[i]
        img_path = os.path.join(folder_path, res['filename'])
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        row_idx = i // cols + 1
        col_idx = i % cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        ax.imshow(img_rgb)
        title = f"{res['filename']}\nScore: {res['similarity_score']:.4f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    logo_comparator = SIFTLogoComparison()

    print("--- Comparing One Image to a Folder ---")
    ref_image_path_folder = '/Users/aaa/Downloads/test_images/amazon.png'
    target_folder_path = '/Users/aaa/Downloads/test_images'

    if os.path.exists(ref_image_path_folder) and os.path.isdir(target_folder_path):
        try:
            batch_results = logo_comparator.compare_one_to_many(ref_image_path_folder, target_folder_path)
            
            print(f"\nTop 10 Text Results for '{os.path.basename(ref_image_path_folder)}':")
            if not batch_results:
                print("No matches found.")
            else:
                for i, res in enumerate(batch_results[:10]):
                    print(f"  {i+1}. Filename: {res['filename']:<25} | Score: {res['similarity_score']:.4f} | Consistent Matches: {res['inlier_matches']}")
                
                display_top_results(batch_results, target_folder_path, ref_image_path_folder)

        except Exception as e:
            print(f"An error occurred during batch comparison: {e}")
    else:
        print(f"Warning: Could not find reference image '{ref_image_path_folder}' or folder '{target_folder_path}'.")
        print("Please update the paths in the script to run the comparison.")
