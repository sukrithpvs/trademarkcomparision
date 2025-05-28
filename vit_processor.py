import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

class ViTProcessor:
    def __init__(self, use_cuda=False, device=None):
        """Initialize ViT processor (CPU only)"""
        self.use_cuda = False
        self.device = torch.device("cpu")
        
        print('Loading Vision Transformer model...')
        self.vit_model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Remove the classification head to get features
        self.vit_model.heads = torch.nn.Identity()
        self.vit_model.eval()
        
        # Model stays on CPU
        self.vit_model = self.vit_model.to(self.device)
        
        # Define image transformations for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"ViT model loaded on device: {self.device}")
    
    def extract_features_batch(self, image_paths, batch_size=32):
        """CPU-based batch ViT feature extraction"""
        all_features = []
        
        # Use smaller batch sizes for CPU
        effective_batch_size = min(batch_size, 16)
        
        for i in range(0, len(image_paths), effective_batch_size):
            batch_paths = image_paths[i:i + effective_batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Failed to process image {path}: {e}")
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            if batch_tensors:
                # Stack into batch and keep on CPU
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Extract features for entire batch
                with torch.no_grad():
                    features = self.vit_model(batch_tensor)
                
                # Convert to numpy and add to results
                all_features.extend(features.float().numpy())
        
        return all_features
    
    def calculate_similarity_batch(self, ref_features, batch_features):
        """CPU-based batch ViT similarity calculation"""
        similarities = []
        for features in batch_features:
            try:
                sim = 1 - cosine(ref_features, features)
                similarities.append(sim * 100)
            except Exception as e:
                print(f"Failed to calculate similarity: {e}")
                similarities.append(0.0)
        return similarities
    
    def extract_single_features(self, image_path):
        """Extract features from a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.vit_model(tensor)
            
            return features.float().numpy()[0]
            
        except Exception as e:
            print(f"Failed to extract features from {image_path}: {e}")
            return np.zeros(768)  # Default ViT feature size
    
    def calculate_single_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        try:
            similarity = 1 - cosine(features1, features2)
            return similarity * 100
        except Exception as e:
            print(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for ViT"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Failed to preprocess image {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def extract_features_from_tensor(self, tensor):
        """Extract features from preprocessed tensor"""
        try:
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            tensor = tensor.to(self.device)
            
            with torch.no_grad():
                features = self.vit_model(tensor)
            
            return features.float().numpy()[0] if features.dim() == 2 else features.float().numpy()
            
        except Exception as e:
            print(f"Failed to extract features from tensor: {e}")
            return np.zeros(768)
