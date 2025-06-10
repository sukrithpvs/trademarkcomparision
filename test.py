import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageTk
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import warnings
import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar, Frame, Label, Button, filedialog, messagebox
import pandas as pd
import json
from datetime import datetime
import time

warnings.filterwarnings("ignore")

class BalancedLogoComparison:
    def _init_(self):
        print("Initializing Balanced Logo Comparison System...")
        
        # BALANCED feature detectors - not too strict, not too lenient
        self.sift = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=3,
            contrastThreshold=0.03,  # Balanced sensitivity
            edgeThreshold=10,        # Balanced edge detection
            sigma=1.6
        )
        
        self.orb = cv2.ORB_create(
            nfeatures=1500,          # Reasonable count
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=18,        # Balanced sensitivity
            scaleFactor=1.2,
            nlevels=8
        )
        
        self.akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.0008,        # Balanced sensitivity
            nOctaves=4,
            nOctaveLayers=4
        )
        
        # BALANCED scoring configuration
        self.scoring_config = {
            'feature_weight': 0.45,       # Primary focus
            'template_weight': 0.25,      # Pattern matching
            'ssim_weight': 0.20,          # Structural similarity
            'shape_weight': 0.10,         # Shape analysis
            
            # BALANCED multipliers - not too conservative, not too generous
            'confidence_boost_high': 1.25,    # Meaningful boost
            'confidence_boost_medium': 1.15,  # Moderate boost
            'confidence_boost_low': 1.08,     # Light boost
            
            # Balanced requirements
            'min_inliers_for_quality': 5,     # Reasonable requirement
            'geometric_consistency_threshold': 0.5,  # Balanced threshold
        }

    def load_and_preprocess(self, image_path):
        """Balanced preprocessing for optimal feature detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Balanced enhancement
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Moderate contrast and sharpness enhancement
            contrast_enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = contrast_enhancer.enhance(1.4)  # Balanced
            
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.5)  # Balanced
            
            img_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            img_cv = self.smart_resize(img_cv)
            
            # Balanced grayscale processing
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Balanced CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Balanced filtering
            gray = cv2.bilateralFilter(gray, 9, 80, 80)
            
            # Light morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            return img_cv, gray
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise

    def smart_resize(self, image, target_size=550):  # Balanced target size
        """Smart resizing for optimal feature detection"""
        h, w = image.shape[:2]
        if max(h, w) == 0:
            return image
        
        # Balanced scaling
        if max(h, w) < 120:
            scale_factor = 350 / max(h, w)  # Moderate upscaling
            interpolation = cv2.INTER_CUBIC
        elif max(h, w) > target_size:
            scale_factor = target_size / max(h, w)
            interpolation = cv2.INTER_AREA
        else:
            return image
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        if new_w > 0 and new_h > 0:
            return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return image

    def balanced_feature_scoring(self, num_inliers, num_good_matches, kp_count1, kp_count2, method_name=""):
        """BALANCED feature scoring - reasonable requirements"""
        if num_good_matches == 0 or num_inliers == 0:
            return 0.0
        
        # BALANCED requirements - based on logo detection research
        min_matches_required = 4 if method_name == "SIFT" else 5
        min_inliers_required = 3 if method_name == "SIFT" else 3
        
        if num_good_matches < min_matches_required or num_inliers < min_inliers_required:
            return 0.0
        
        # Balanced scoring components
        inlier_ratio = num_inliers / num_good_matches
        match_confidence = min(1.0, num_good_matches / 20.0)  # Balanced requirement
        inlier_confidence = min(1.0, num_inliers / 15.0)      # Balanced requirement
        
        # Balanced geometric consistency requirement
        if inlier_ratio < self.scoring_config['geometric_consistency_threshold']:
            inlier_ratio *= 0.8  # Penalty, not rejection
        
        # Balanced base score calculation
        base_score = (inlier_ratio * 0.4 + match_confidence * 0.35 + inlier_confidence * 0.25)
        
        # BALANCED quality bonuses
        if inlier_ratio > 0.8 and num_inliers > 10:
            base_score *= 1.3  # Good bonus
        elif inlier_ratio > 0.7 and num_inliers > 8:
            base_score *= 1.25 # Moderate bonus
        elif inlier_ratio > 0.6 and num_inliers > 5:
            base_score *= 1.2  # Light bonus
        elif inlier_ratio > 0.5 and num_inliers > 3:
            base_score *= 1.15 # Minimal bonus
        elif inlier_ratio < 0.4:
            base_score *= 0.9  # Light penalty
        
        # Keypoint density bonus
        avg_keypoints = (kp_count1 + kp_count2) / 2
        if avg_keypoints > 150:
            base_score *= 1.1
        elif avg_keypoints > 100:
            base_score *= 1.05
        
        return min(0.95, max(0.0, base_score))

    def enhanced_sift_similarity(self, img1_path, img2_path):
        """Enhanced SIFT with balanced requirements"""
        try:
            _, gray1 = self.load_and_preprocess(img1_path)
            _, gray2 = self.load_and_preprocess(img2_path)
            
            kp1, desc1 = self.sift.detectAndCompute(gray1, None)
            kp2, desc2 = self.sift.detectAndCompute(gray2, None)

            # BALANCED requirements - not too strict
            if desc1 is None or desc2 is None or len(kp1) < 5 or len(kp2) < 5:
                return 0.0, len(kp1) if kp1 is not None else 0, len(kp2) if kp2 is not None else 0, 0

            # FLANN matching with error handling
            try:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc1, desc2, k=2)
            except:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(desc1, desc2, k=2)
            
            if not matches:
                return 0.0, len(kp1), len(kp2), 0

            # BALANCED Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Balanced ratio
                        good_matches.append(m)

            MIN_MATCH_COUNT = 4  # BALANCED requirement
            if len(good_matches) < MIN_MATCH_COUNT:
                return 0.0, len(kp1), len(kp2), 0

            # BALANCED homography validation
            score = 0.0
            num_inliers = 0
            
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Balanced threshold
                    
                    if M is not None and mask is not None:
                        num_inliers = np.sum(mask)
                        if num_inliers >= 3:  # BALANCED minimum
                            score = self.balanced_feature_scoring(
                                num_inliers, len(good_matches), len(kp1), len(kp2), "SIFT"
                            )
                except:
                    # Fallback scoring for edge cases
                    if len(good_matches) >= 6:
                        score = min(0.6, len(good_matches) / 15.0)

            return score, len(kp1), len(kp2), int(num_inliers)
            
        except Exception as e:
            print(f"Warning: SIFT error: {e}")
            return 0.0, 0, 0, 0

    def enhanced_orb_similarity(self, img1_path, img2_path):
        """Enhanced ORB with balanced requirements"""
        try:
            _, gray1 = self.load_and_preprocess(img1_path)
            _, gray2 = self.load_and_preprocess(img2_path)
            
            kp1, desc1 = self.orb.detectAndCompute(gray1, None)
            kp2, desc2 = self.orb.detectAndCompute(gray2, None)

            # BALANCED requirements
            if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
                return 0.0, len(kp1) if kp1 is not None else 0, len(kp2) if kp2 is not None else 0, 0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            
            if not matches:
                return 0.0, len(kp1), len(kp2), 0
                
            matches = sorted(matches, key=lambda x: x.distance)
            # BALANCED distance threshold
            good_matches = [m for m in matches if m.distance < 40]
            
            MIN_ORB_MATCHES = 5  # BALANCED requirement
            if len(good_matches) < MIN_ORB_MATCHES:
                return 0.0, len(kp1), len(kp2), 0

            score = 0.0
            num_inliers = 0

            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Balanced
                    
                    if M is not None and mask is not None:
                        num_inliers = np.sum(mask)
                        if num_inliers >= 3:  # BALANCED minimum
                            score = self.balanced_feature_scoring(
                                num_inliers, len(good_matches), len(kp1), len(kp2), "ORB"
                            )
                except:
                    # Fallback scoring
                    if len(good_matches) >= 8:
                        score = min(0.7, len(good_matches) / 20.0)

            return score, len(kp1), len(kp2), int(num_inliers)
            
        except Exception as e:
            print(f"Warning: ORB error: {e}")
            return 0.0, 0, 0, 0

    def enhanced_akaze_similarity(self, img1_path, img2_path):
        """Enhanced AKAZE with balanced requirements"""
        try:
            _, gray1 = self.load_and_preprocess(img1_path)
            _, gray2 = self.load_and_preprocess(img2_path)
            
            kp1, desc1 = self.akaze.detectAndCompute(gray1, None)
            kp2, desc2 = self.akaze.detectAndCompute(gray2, None)

            # BALANCED requirements
            if desc1 is None or desc2 is None or len(kp1) < 6 or len(kp2) < 6:
                return 0.0, len(kp1) if kp1 is not None else 0, len(kp2) if kp2 is not None else 0, 0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            
            if not matches:
                return 0.0, len(kp1), len(kp2), 0
                
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 45]  # BALANCED threshold
            
            MIN_AKAZE_MATCHES = 4
            if len(good_matches) < MIN_AKAZE_MATCHES:
                return 0.0, len(kp1), len(kp2), 0

            score = 0.0
            num_inliers = 0

            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None and mask is not None:
                        num_inliers = np.sum(mask)
                        if num_inliers >= 3:
                            score = self.balanced_feature_scoring(
                                num_inliers, len(good_matches), len(kp1), len(kp2), "AKAZE"
                            )
                except:
                    # Fallback scoring
                    if len(good_matches) >= 6:
                        score = min(0.65, len(good_matches) / 12.0)

            return score, len(kp1), len(kp2), int(num_inliers)
            
        except Exception as e:
            print(f"Warning: AKAZE error: {e}")
            return 0.0, 0, 0, 0

    def balanced_template_similarity(self, img1_path, img2_path):
        """Balanced template matching"""
        try:
            _, gray1 = self.load_and_preprocess(img1_path)
            _, gray2 = self.load_and_preprocess(img2_path)
            
            area1 = gray1.shape[0] * gray1.shape[1]
            area2 = gray2.shape[0] * gray2.shape[1]
            
            if area1 == 0 or area2 == 0:
                return 0.0

            if area1 < area2:
                template, target = gray1, gray2
            else:
                template, target = gray2, gray1
            
            # BALANCED size validation
            if (template.shape[0] * template.shape[1]) < 0.015 * (target.shape[0] * target.shape[1]):
                return 0.0
            
            max_score = 0.0
            
            # Balanced scale range
            for scale in np.linspace(0.7, 1.3, 5)[::-1]:
                resized_h = int(template.shape[0] * scale)
                resized_w = int(template.shape[1] * scale)
                
                if resized_h <= 8 or resized_w <= 8:
                    continue
                if resized_h > target.shape[0] or resized_w > target.shape[1]:
                    continue
                
                scaled_template = cv2.resize(template, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
                
                try:
                    result = cv2.matchTemplate(target, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    # BALANCED threshold
                    if max_val > 0.25:
                        # BALANCED validation
                        x, y = max_loc
                        matched_region = target[y:y+resized_h, x:x+resized_w]
                        
                        # Edge validation
                        template_edges = cv2.Canny(scaled_template, 40, 120)
                        matched_edges = cv2.Canny(matched_region, 40, 120)
                        
                        edge_corr = cv2.matchTemplate(matched_edges, template_edges, cv2.TM_CCORR_NORMED)
                        _, edge_max, _, _ = cv2.minMaxLoc(edge_corr)
                        
                        # BALANCED validation requirement
                        if edge_max > 0.3:  # Balanced threshold
                            combined_score = (max_val * 0.7) + (edge_max * 0.3)
                            max_score = max(max_score, combined_score)
                        elif max_val > 0.6:  # High template match can stand alone
                            max_score = max(max_score, max_val * 0.9)
                        
                except:
                    continue
            
            # Balanced final scoring
            if max_score < 0.3:
                return 0.0
            elif max_score < 0.6:
                return (max_score - 0.3) * (0.5 / 0.3)  # Map [0.3-0.6] to [0-0.5]
            else:
                return 0.5 + (max_score - 0.6) * (0.45 / 0.4)  # Map [0.6-1.0] to [0.5-0.95]
            
        except Exception as e:
            print(f"Warning: Template matching error: {e}")
            return 0.0

    def balanced_ssim_similarity(self, img1_path, img2_path):
        """Balanced SSIM similarity"""
        try:
            _, gray1 = self.load_and_preprocess(img1_path)
            _, gray2 = self.load_and_preprocess(img2_path)
            
            # Multi-resolution SSIM
            target_sizes = [(64, 64), (128, 128), (256, 256)]
            ssim_scores = []
            
            for target_size in target_sizes:
                try:
                    gray1_resized = cv2.resize(gray1, target_size)
                    gray2_resized = cv2.resize(gray2, target_size)
                    
                    data_range = max(gray1_resized.max() - gray1_resized.min(),
                                   gray2_resized.max() - gray2_resized.min())
                    
                    if data_range == 0:
                        ssim_val = 1.0 if np.array_equal(gray1_resized, gray2_resized) else 0.0
                    else:
                        score, _ = ssim(gray1_resized, gray2_resized, full=True, data_range=data_range, win_size=7)
                        ssim_val = max(0.0, score)
                    
                    ssim_scores.append(ssim_val)
                except:
                    continue
            
            if ssim_scores:
                # Weighted average - emphasize medium resolution
                if len(ssim_scores) >= 3:
                    weighted_ssim = (ssim_scores[0] * 0.2 + ssim_scores[1] * 0.6 + ssim_scores[2] * 0.2)
                else:
                    weighted_ssim = np.mean(ssim_scores)
                
                # BALANCED SSIM scoring
                if weighted_ssim < 0.3:
                    return 0.0
                else:
                    return min(0.9, (weighted_ssim - 0.3) / 0.7 * 0.9)
            
            return 0.0
                
        except Exception as e:
            print(f"Warning: SSIM error: {e}")
            return 0.0

    def balanced_shape_analysis(self, img1_path, img2_path):
        """Balanced shape analysis"""
        try:
            _, gray1 = self.load_and_preprocess(img1_path)
            _, gray2 = self.load_and_preprocess(img2_path)
            
            # Edge detection
            edges1 = cv2.Canny(gray1, 40, 120)
            edges2 = cv2.Canny(gray2, 40, 120)
            
            # Find contours
            contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours1 or not contours2:
                return 0.0
            
            # BALANCED contour filtering
            min_area = 100  # Balanced threshold
            contours1 = [c for c in contours1 if cv2.contourArea(c) > min_area]
            contours2 = [c for c in contours2 if cv2.contourArea(c) > min_area]
            
            if not contours1 or not contours2:
                return 0.0
            
            best_similarity = 0.0
            
            # Compare top contours
            contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)[:3]
            contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:3]
            
            for c1 in contours1:
                for c2 in contours2:
                    try:
                        similarities = []
                        
                        # Hu moments
                        moments1 = cv2.moments(c1)
                        moments2 = cv2.moments(c2)
                        
                        if moments1['m00'] > 0 and moments2['m00'] > 0:
                            hu1 = cv2.HuMoments(moments1).flatten()
                            hu2 = cv2.HuMoments(moments2).flatten()
                            
                            hu_distance = np.sum(np.abs(np.log(np.abs(hu1) + 1e-10) - np.log(np.abs(hu2) + 1e-10)))
                            hu_similarity = 1.0 / (1.0 + hu_distance)
                            similarities.append(hu_similarity)
                        
                        # Shape matching
                        match_score = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                        shape_similarity = 1.0 / (1.0 + match_score)
                        similarities.append(shape_similarity)
                        
                        # Area ratio
                        area1 = cv2.contourArea(c1)
                        area2 = cv2.contourArea(c2)
                        if area1 > 0 and area2 > 0:
                            area_ratio = min(area1, area2) / max(area1, area2)
                            similarities.append(area_ratio)
                        
                        if similarities:
                            avg_similarity = np.mean(similarities)
                            best_similarity = max(best_similarity, avg_similarity)
                            
                    except:
                        continue
            
            # Balanced shape scoring
            if best_similarity > 0.6:
                best_similarity = min(0.85, best_similarity * 1.2)
            elif best_similarity > 0.4:
                best_similarity = min(0.7, best_similarity * 1.1)
            
            return best_similarity
            
        except Exception as e:
            print(f"Warning: Shape analysis error: {e}")
            return 0.0

    def balanced_score_fusion(self, scores, method_details):
        """BALANCED score fusion with proper discrimination"""
        try:
            feature_score = scores.get('feature', 0.0)
            template_score = scores.get('template', 0.0)
            ssim_score = scores.get('ssim', 0.0)
            shape_score = scores.get('shape', 0.0)
            
            # Apply balanced weights
            weighted_score = (
                feature_score * self.scoring_config['feature_weight'] +
                template_score * self.scoring_config['template_weight'] +
                ssim_score * self.scoring_config['ssim_weight'] +
                shape_score * self.scoring_config['shape_weight']
            )
            
            # BALANCED confidence analysis
            strong_methods = sum(1 for s in [feature_score, template_score, ssim_score, shape_score] if s > 0.6)
            good_methods = sum(1 for s in [feature_score, template_score, ssim_score, shape_score] if s > 0.35)
            weak_methods = sum(1 for s in [feature_score, template_score, ssim_score, shape_score] if s > 0.1)
            
            # BALANCED confidence multipliers
            if strong_methods >= 2:
                confidence_multiplier = self.scoring_config['confidence_boost_high']  # 1.25
            elif strong_methods >= 1 or good_methods >= 2:
                confidence_multiplier = self.scoring_config['confidence_boost_medium']  # 1.15
            elif good_methods >= 1 or weak_methods >= 2:
                confidence_multiplier = self.scoring_config['confidence_boost_low']  # 1.08
            else:
                confidence_multiplier = 1.0  # No boost for very weak signals
            
            # Apply balanced multiplier
            enhanced_score = weighted_score * confidence_multiplier
            
            # Method agreement bonuses
            if feature_score > 0.4 and template_score > 0.3:
                enhanced_score *= 1.12  # Feature + template agreement
            elif feature_score > 0.5 and ssim_score > 0.4:
                enhanced_score *= 1.08  # Feature + structure agreement
            
            # Geometric validation consideration
            inliers = method_details.get('inliers', 0)
            if inliers >= self.scoring_config['min_inliers_for_quality']:
                enhanced_score *= 1.1  # Geometric validation bonus
            elif inliers > 0:
                enhanced_score *= 1.05  # Light bonus for some geometric consistency
            
            # BALANCED final score capping
            final_score = min(0.95, max(0.0, enhanced_score))
            
            return final_score
            
        except Exception as e:
            print(f"Score fusion error: {e}")
            return 0.0

    def comprehensive_logo_analysis(self, img1_path, img2_path):
        """Comprehensive analysis with balanced scoring"""
        try:
            print(f"    🔍 Analyzing {os.path.basename(img2_path)}...")
            
            # Run all methods with balanced requirements
            sift_score, kp1_s, kp2_s, inliers_s = self.enhanced_sift_similarity(img1_path, img2_path)
            orb_score, kp1_o, kp2_o, inliers_o = self.enhanced_orb_similarity(img1_path, img2_path)
            akaze_score, kp1_a, kp2_a, inliers_a = self.enhanced_akaze_similarity(img1_path, img2_path)
            template_score = self.balanced_template_similarity(img1_path, img2_path)
            ssim_score = self.balanced_ssim_similarity(img1_path, img2_path)
            shape_score = self.balanced_shape_analysis(img1_path, img2_path)

            print(f"        SIFT: {sift_score:.4f} | ORB: {orb_score:.4f} | AKAZE: {akaze_score:.4f}")
            print(f"        Template: {template_score:.4f} | SSIM: {ssim_score:.4f} | Shape: {shape_score:.4f}")

            # Select best feature method
            feature_methods = [
                (sift_score, inliers_s, kp1_s, kp2_s, 'SIFT'),
                (orb_score, inliers_o, kp1_o, kp2_o, 'ORB'),
                (akaze_score, inliers_a, kp1_a, kp2_a, 'AKAZE')
            ]
            
            best_feature = max(feature_methods, key=lambda x: x[0])
            feature_score = best_feature[0]
            feature_inliers = best_feature[1]
            kp1_count, kp2_count = best_feature[2], best_feature[3]
            best_method = best_feature[4]

            # Method details for balanced fusion
            method_details = {
                'best_method': best_method,
                'inliers': feature_inliers,
                'total_matches': (kp1_count + kp2_count) / 2
            }

            # BALANCED score fusion
            final_score = self.balanced_score_fusion({
                'feature': feature_score,
                'template': template_score,
                'ssim': ssim_score,
                'shape': shape_score
            }, method_details)

            print(f"        🎯 Final Score: {final_score:.4f} (Method: {best_method})")

            return final_score, kp1_count, kp2_count, feature_inliers, best_method
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return 0.0, 0, 0, 0, "Error"

    def compare_two_images(self, img1_path, img2_path):
        """Compare two images with balanced scoring"""
        similarity_score, kp1_count, kp2_count, matches, best_method = self.comprehensive_logo_analysis(img1_path, img2_path)
        
        # BALANCED confidence levels
        if similarity_score >= 0.75:
            confidence_level = "Very High"
        elif similarity_score >= 0.55:
            confidence_level = "High"
        elif similarity_score >= 0.35:
            confidence_level = "Medium"
        elif similarity_score >= 0.15:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        return {
            'similarity_score': similarity_score, 
            'keypoints_img1': kp1_count, 
            'keypoints_img2': kp2_count, 
            'inlier_matches': matches,
            'best_method': best_method,
            'confidence_level': confidence_level
        }

    def compare_one_to_many(self, reference_image_path, image_folder_path):
        """Balanced batch comparison"""
        if not os.path.exists(reference_image_path):
            print(f"Error: Reference image not found: {reference_image_path}")
            return []
            
        if not os.path.exists(image_folder_path):
            print(f"Error: Target folder not found: {image_folder_path}")
            return []

        results = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
        ref_basename = os.path.basename(reference_image_path)
        
        image_files = [f for f in os.listdir(image_folder_path) 
                      if (os.path.isfile(os.path.join(image_folder_path, f)) and 
                          f.lower().endswith(image_extensions) and 
                          f != ref_basename)]

        if not image_files:
            print("Warning: No suitable image files found!")
            return []
            
        total_files = len(image_files)
        print(f"🚀 Found {total_files} images to compare against '{ref_basename}'.")

        start_time = time.time()
        
        for i, filename in enumerate(image_files):
            try:
                image_path = os.path.join(image_folder_path, filename)
                print(f'📸 Processing [{i+1}/{total_files}]: {filename}...')
                
                comparison_result = self.compare_two_images(reference_image_path, image_path)
                comparison_result['filename'] = filename
                results.append(comparison_result)
                
                score = comparison_result['similarity_score']
                confidence = comparison_result['confidence_level']
                method = comparison_result['best_method']
                
                print(f"    ✅ Score: {score:.4f} | Confidence: {confidence} | Method: {method}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                results.append({
                    'filename': filename, 
                    'similarity_score': 0.0, 
                    'keypoints_img1': 0, 
                    'keypoints_img2': 0, 
                    'inlier_matches': 0,
                    'best_method': 'Error',
                    'confidence_level': 'Very Low',
                    'error': str(e)
                })
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        elapsed_time = time.time() - start_time
        print(f'🎉 Processing complete! Total time: {elapsed_time:.2f} seconds')
        
        return results


def display_balanced_results(results, folder_path, ref_image_path):
    """Display results with balanced visualization"""
    top_n = min(15, len(results))
    if top_n == 0:
        print("No results to display.")
        return

    cols = 5
    rows = (top_n + cols - 1) // cols
    fig = plt.figure(figsize=(18, 4 * rows + 2))
    gs = fig.add_gridspec(rows + 1, cols, hspace=0.4, wspace=0.3)

    # Reference image
    try:
        ref_img = cv2.imread(ref_image_path)
        ax_ref = fig.add_subplot(gs[0, :])
        if ref_img is not None:
            ax_ref.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            ax_ref.set_title(f"🎯 REFERENCE: {os.path.basename(ref_image_path)}", 
                           fontsize=16, color='darkblue', weight='bold', pad=15)
        ax_ref.axis('off')
    except Exception as e:
        print(f"Error loading reference image: {e}")

    # Results with balanced color coding
    for i in range(top_n):
        try:
            res = results[i]
            img_path = os.path.join(folder_path, res['filename'])
            
            row_idx = i // cols + 1
            col_idx = i % cols
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            img = cv2.imread(img_path)
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.text(0.5, 0.5, "Missing", ha='center', va='center')
            
            score = res.get('similarity_score', 0.0)
            confidence = res.get('confidence_level', 'Unknown')
            method = res.get('best_method', 'Unknown')
            inliers = res.get('inlier_matches', 0)
            
            # BALANCED color coding
            if score >= 0.75:
                color = 'darkgreen'
            elif score >= 0.55:
                color = 'green'
            elif score >= 0.35:
                color = 'orange'
            elif score >= 0.15:
                color = 'darkorange'
            else:
                color = 'red'
            
            # Rank indicators
            if i == 0:
                rank = "🥇"
            elif i == 1:
                rank = "🥈"  
            elif i == 2:
                rank = "🥉"
            else:
                rank = f"#{i+1}"
            
            title = f"{rank} {res['filename']}\n🎯 {score:.4f} ({confidence})\n⚙ {method}"
            if inliers > 0:
                title += f" | I:{inliers}"
            
            ax.set_title(title, fontsize=9, color=color, weight='bold')
            ax.axis('off')
            
        except Exception as e:
            print(f"Error displaying result {i}: {e}")

    plt.suptitle("⚖ BALANCED Logo Similarity Results - Proper Discrimination", 
                fontsize=18, y=0.98, weight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("⚖ Initializing BALANCED Logo Comparison System...")
    
    logo_comparator = BalancedLogoComparison()
    
    print("=" * 80)
    print("⚖ BALANCED LOGO COMPARISON SYSTEM")
    print("=" * 80)
    print("✨ Balanced Fixes Applied:")
    print("   • REASONABLE confidence multipliers (1.08-1.25x)")
    print("   • BALANCED geometric validation requirements")
    print("   • MODERATE score ranges (Nike logos: 0.6-0.9)")
    print("   • PROPER discrimination without complete rejection")
    print("   • BALANCED thresholds for meaningful results")
    print("=" * 80)
    
    # Your paths
    ref_image_path = r"C:\Users\sukri\Desktop\TradeMarl_Comparison\extracted_logos\client\5736778_page_870_img_0.png"
    target_folder_path = r"C:\Users\sukri\Desktop\TradeMarl_Comparison\extracted_logos\client"
    
    if os.path.exists(ref_image_path) and os.path.isdir(target_folder_path):
        try:
            print(f"📁 Reference: {os.path.basename(ref_image_path)}")
            print(f"📁 Target Folder: {os.path.basename(target_folder_path)}")
            print("")
            
            batch_results = logo_comparator.compare_one_to_many(ref_image_path, target_folder_path)
            
            if not batch_results:
                print("❌ No matches found.")
            else:
                print(f"\n🏆 BALANCED RESULTS for '{os.path.basename(ref_image_path)}':")
                print("=" * 100)
                
                for i, res in enumerate(batch_results):
                    score = res.get('similarity_score', 0.0)
                    confidence = res.get('confidence_level', 'Unknown')
                    method = res.get('best_method', 'Unknown')
                    filename = res.get('filename', 'N/A')
                    inliers = res.get('inlier_matches', 0)
                    
                    if i == 0:
                        rank_emoji = "🥇"
                    elif i == 1:
                        rank_emoji = "🥈"
                    elif i == 2:
                        rank_emoji = "🥉"
                    else:
                        rank_emoji = f"{i+1:2d}."
                    
                    print(f"  {rank_emoji} {filename:<35} | 🎯 {score:.4f} | 🔍 {confidence:<10} | ⚙ {method:<6} | 🔗 I:{inliers}")
                
                print("=" * 100)
                
                # BALANCED statistics
                scores = [res['similarity_score'] for res in batch_results]
                avg_score = np.mean(scores)
                very_high_matches = len([s for s in scores if s >= 0.75])
                high_matches = len([s for s in scores if 0.55 <= s < 0.75])
                medium_matches = len([s for s in scores if 0.35 <= s < 0.55])
                low_matches = len([s for s in scores if 0.15 <= s < 0.35])
                very_low_matches = len([s for s in scores if s < 0.15])
                
                print(f"📊 BALANCED SUMMARY:")
                print(f"   • Average Score: {avg_score:.4f}")
                print(f"   • Very High (≥0.75): {very_high_matches}/{len(scores)}")
                print(f"   • High (0.55-0.75): {high_matches}/{len(scores)}")
                print(f"   • Medium (0.35-0.55): {medium_matches}/{len(scores)}")
                print(f"   • Low (0.15-0.35): {low_matches}/{len(scores)}")
                print(f"   • Very Low (<0.15): {very_low_matches}/{len(scores)}")
                print(f"   • BALANCED discrimination - meaningful score ranges!")
                
                print(f"\n🖥  Launching Balanced Results Viewer...")
                display_balanced_results(batch_results, target_folder_path, ref_image_path)
                
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Could not find paths.")