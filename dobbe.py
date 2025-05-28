#!/usr/bin/env python3
"""
Adaptive Image Preprocessing Pipeline for IOPA X-rays
Dobbe AI - Data Science Intern Assignment

This module implements an adaptive image preprocessing pipeline that intelligently
adjusts image parameters based on the characteristics of input IOPA X-ray images.
"""

import os
import numpy as np
import cv2
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import filters, restoration, measure, feature
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings('ignore')

class DICOMHandler:
    """Handler for DICOM file operations"""
    
    @staticmethod
    def read_dicom(filepath):
        """
        Read DICOM file and extract pixel data and metadata
        
        Args:
            filepath (str): Path to DICOM file
            
        Returns:
            tuple: (pixel_array, metadata_dict)
        """
        try:
            dicom_data = pydicom.dcmread(filepath)
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array
            
            # Extract relevant metadata
            metadata = {
                'StudyDate': getattr(dicom_data, 'StudyDate', 'Unknown'),
                'Modality': getattr(dicom_data, 'Modality', 'Unknown'),
                'PhotometricInterpretation': getattr(dicom_data, 'PhotometricInterpretation', 'Unknown'),
                'PixelSpacing': getattr(dicom_data, 'PixelSpacing', None),
                'WindowCenter': getattr(dicom_data, 'WindowCenter', None),
                'WindowWidth': getattr(dicom_data, 'WindowWidth', None),
                'RescaleIntercept': getattr(dicom_data, 'RescaleIntercept', 0),
                'RescaleSlope': getattr(dicom_data, 'RescaleSlope', 1),
            }
            
            # Apply rescale if available
            if metadata['RescaleSlope'] != 1 or metadata['RescaleIntercept'] != 0:
                pixel_array = pixel_array * metadata['RescaleSlope'] + metadata['RescaleIntercept']
            
            return pixel_array, metadata
            
        except Exception as e:
            print(f"Error reading DICOM file {filepath}: {e}")
            return None, None
    
    @staticmethod
    def visualize_dicom(pixel_array, title="DICOM Image"):
        """Visualize DICOM image"""
        plt.figure(figsize=(8, 6))
        plt.imshow(pixel_array, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.show()

class ImageQualityAnalyzer:
    """Analyzer for image quality metrics"""
    
    @staticmethod
    def calculate_brightness(image):
        """Calculate brightness as mean pixel intensity"""
        return np.mean(image)
    
    @staticmethod
    def calculate_contrast_std(image):
        """Calculate contrast as standard deviation of pixel intensities"""
        return np.std(image)
    
    @staticmethod
    def calculate_rms_contrast(image):
        """Calculate RMS contrast"""
        mean_intensity = np.mean(image)
        return np.sqrt(np.mean((image - mean_intensity) ** 2))
    
    @staticmethod
    def calculate_michelson_contrast(image):
        """Calculate Michelson contrast"""
        max_intensity = np.max(image)
        min_intensity = np.min(image)
        if max_intensity + min_intensity == 0:
            return 0
        return (max_intensity - min_intensity) / (max_intensity + min_intensity)
    
    @staticmethod
    def calculate_sharpness_laplacian(image):
        if len(image.shape) == 3:
          image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Ensure image is uint8 for OpenCV Laplacian
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
        return cv2.Laplacian(image, cv2.CV_64F).var()

    
    @staticmethod
    def calculate_sharpness_tenengrad(image):
        """Calculate sharpness using Tenengrad method"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        sobel_x = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(sobel_x**2 + sobel_y**2)
    
    @staticmethod
    def estimate_noise_std(image):
        """Estimate noise using standard deviation in flat regions"""
        # Use median filter to estimate noise
        median_filtered = cv2.medianBlur(image.astype(np.uint8), 5)
        noise = image.astype(np.float32) - median_filtered.astype(np.float32)
        return np.std(noise)
    
    @staticmethod
    def estimate_noise_wavelet(image):
        """Estimate noise using wavelet-based method"""
        try:
            from skimage.restoration import estimate_sigma
            return estimate_sigma(image, average_sigmas=True)
        except ImportError:
            # Fallback to standard deviation method
            return ImageQualityAnalyzer.estimate_noise_std(image)
    
    @staticmethod
    def analyze_image_quality(image):
        """
        Comprehensive image quality analysis
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Dictionary containing all quality metrics
        """
        # Normalize image to 0-255 range for consistent analysis
        if image.dtype != np.uint8:
            image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            image_norm = image
        
        metrics = {
            'brightness': ImageQualityAnalyzer.calculate_brightness(image_norm),
            'contrast_std': ImageQualityAnalyzer.calculate_contrast_std(image_norm),
            'contrast_rms': ImageQualityAnalyzer.calculate_rms_contrast(image_norm),
            'contrast_michelson': ImageQualityAnalyzer.calculate_michelson_contrast(image_norm),
            'sharpness_laplacian': ImageQualityAnalyzer.calculate_sharpness_laplacian(image_norm),
            'sharpness_tenengrad': ImageQualityAnalyzer.calculate_sharpness_tenengrad(image_norm),
            'noise_std': ImageQualityAnalyzer.estimate_noise_std(image_norm),
            'noise_wavelet': ImageQualityAnalyzer.estimate_noise_wavelet(image_norm)
        }
        
        return metrics

class StaticPreprocessor:
    """Static preprocessing pipeline (baseline)"""
    
    @staticmethod
    def preprocess(image):
        """
        Apply static preprocessing steps
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Normalize to 0-255
        if image.dtype != np.uint8:
            processed = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            processed = image.copy()
        
        # Apply histogram equalization
        processed = cv2.equalizeHist(processed)
        
        # Apply fixed sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
        
        # Apply basic denoising
        processed = cv2.bilateralFilter(processed, 5, 50, 50)
        
        return processed

class AdaptivePreprocessor:
    """Adaptive preprocessing pipeline"""
    
    def __init__(self):
        self.quality_analyzer = ImageQualityAnalyzer()
    
    def _adaptive_contrast_enhancement(self, image, contrast_metrics):
        """Apply adaptive contrast enhancement based on image characteristics"""
        contrast_std = contrast_metrics['contrast_std']
        contrast_rms = contrast_metrics['contrast_rms']
        
        if contrast_std < 30:  # Low contrast
            # Apply CLAHE with higher clip limit
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        elif contrast_std > 80:  # High contrast
            # Apply mild histogram equalization
            enhanced = cv2.equalizeHist(image)
            enhanced = cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
        else:  # Normal contrast
            # Apply standard CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _adaptive_sharpening(self, image, sharpness_metrics):
        """Apply adaptive sharpening based on image sharpness"""
        sharpness_laplacian = sharpness_metrics['sharpness_laplacian']
        
        if sharpness_laplacian < 100:  # Blurry image
            # Strong sharpening
            kernel = np.array([[-1,-1,-1], [-1,12,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            # Blend with original to avoid over-sharpening
            result = cv2.addWeighted(image, 0.3, sharpened, 0.7, 0)
        elif sharpness_laplacian > 500:  # Already sharp
            # Mild sharpening or no sharpening
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            result = cv2.filter2D(image, -1, kernel)
            result = cv2.addWeighted(image, 0.8, result, 0.2, 0)
        else:  # Moderate sharpness
            # Standard sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(image, -1, kernel)
        
        return result
    
    def _adaptive_denoising(self, image, noise_metrics):
        """Apply adaptive denoising based on estimated noise level"""
        noise_std = noise_metrics['noise_std']
        
        if noise_std > 15:  # High noise
            # Strong denoising
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            # Also apply non-local means denoising
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        elif noise_std > 8:  # Moderate noise
            # Medium denoising
            denoised = cv2.bilateralFilter(image, 7, 50, 50)
        elif noise_std > 3:  # Low noise
            # Light denoising
            denoised = cv2.bilateralFilter(image, 5, 25, 25)
        else:  # Very low noise
            # Minimal or no denoising
            denoised = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return denoised
    
    def _adaptive_brightness_adjustment(self, image, brightness_metric):
        """Apply adaptive brightness adjustment"""
        brightness = brightness_metric['brightness']
        
        if brightness < 80:  # Too dark
            # Increase brightness
            adjusted = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        elif brightness > 180:  # Too bright
            # Decrease brightness
            adjusted = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        else:  # Acceptable brightness
            adjusted = image
        
        return adjusted
    
    def preprocess(self, image):
        """
        Apply adaptive preprocessing pipeline
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (preprocessed_image, quality_metrics)
        """
        # Normalize to 0-255
        if image.dtype != np.uint8:
            processed = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            processed = image.copy()
        
        # Analyze image quality
        quality_metrics = self.quality_analyzer.analyze_image_quality(processed)
        
        # Apply adaptive adjustments in sequence
        processed = self._adaptive_brightness_adjustment(processed, quality_metrics)
        processed = self._adaptive_contrast_enhancement(processed, quality_metrics)
        processed = self._adaptive_denoising(processed, quality_metrics)
        processed = self._adaptive_sharpening(processed, quality_metrics)
        
        return processed, quality_metrics

class PipelineEvaluator:
    """Evaluator for preprocessing pipeline performance"""
    
    @staticmethod
    def calculate_psnr(original, processed, data_range=255):
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(original, processed, data_range=data_range)
    
    @staticmethod
    def calculate_ssim(original, processed, data_range=255):
        """Calculate Structural Similarity Index"""
        return ssim(original, processed, data_range=data_range)
    
    @staticmethod
    def calculate_edge_enhancement(original, processed):
        """Calculate improvement in edge content"""
        # Calculate edge content using Sobel operator
        original_edges = cv2.Sobel(original, cv2.CV_64F, 1, 1, ksize=3)
        processed_edges = cv2.Sobel(processed, cv2.CV_64F, 1, 1, ksize=3)
        
        original_edge_strength = np.mean(np.abs(original_edges))
        processed_edge_strength = np.mean(np.abs(processed_edges))
        
        return processed_edge_strength / original_edge_strength if original_edge_strength > 0 else 1
    
    @staticmethod
    def evaluate_pipeline(original, processed):
        """
        Comprehensive evaluation of preprocessing pipeline
        
        Args:
            original (numpy.ndarray): Original image
            processed (numpy.ndarray): Processed image
            
        Returns:
            dict: Evaluation metrics
        """
        # Ensure both images are the same type and range
        if original.dtype != processed.dtype:
            original = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        metrics = {
            'psnr': PipelineEvaluator.calculate_psnr(original, processed),
            'ssim': PipelineEvaluator.calculate_ssim(original, processed),
            'edge_enhancement': PipelineEvaluator.calculate_edge_enhancement(original, processed)
        }
        
        return metrics

class AdaptivePreprocessingPipeline:
    """Main pipeline class orchestrating the entire preprocessing workflow"""
    
    def __init__(self):
        self.dicom_handler = DICOMHandler()
        self.quality_analyzer = ImageQualityAnalyzer()
        self.static_preprocessor = StaticPreprocessor()
        self.adaptive_preprocessor = AdaptivePreprocessor()
        self.evaluator = PipelineEvaluator()
    
    def process_single_image(self, image_path):
        """
        Process a single image through both static and adaptive pipelines
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Results containing original image, processed images, and metrics
        """
        results = {}
        
        # Read image
        if image_path.lower().endswith('.dcm'):
            original_image, metadata = self.dicom_handler.read_dicom(image_path)
            if original_image is None:
                return None
            results['metadata'] = metadata
        else:
            # Handle other formats (jpg, png, etc.)
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                return None
            results['metadata'] = {}
        
        results['original'] = original_image
        
        # Analyze original image quality
        original_quality = self.quality_analyzer.analyze_image_quality(original_image)
        results['original_quality'] = original_quality
        
        # Static preprocessing
        static_processed = self.static_preprocessor.preprocess(original_image)
        results['static_processed'] = static_processed
        
        # Adaptive preprocessing
        adaptive_processed, adaptive_quality = self.adaptive_preprocessor.preprocess(original_image)
        results['adaptive_processed'] = adaptive_processed
        results['adaptive_quality'] = adaptive_quality
        
        # Evaluate both pipelines
        static_evaluation = self.evaluator.evaluate_pipeline(original_image, static_processed)
        adaptive_evaluation = self.evaluator.evaluate_pipeline(original_image, adaptive_processed)
        
        results['static_evaluation'] = static_evaluation
        results['adaptive_evaluation'] = adaptive_evaluation
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """
        Visualize comparison between original, static, and adaptive preprocessing
        
        Args:
            results (dict): Results from process_single_image
            save_path (str, optional): Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Static processed
        axes[0, 1].imshow(results['static_processed'], cmap='gray')
        axes[0, 1].set_title('Static Preprocessing')
        axes[0, 1].axis('off')
        
        # Adaptive processed
        axes[0, 2].imshow(results['adaptive_processed'], cmap='gray')
        axes[0, 2].set_title('Adaptive Preprocessing')
        axes[0, 2].axis('off')
        
        # Histograms
        axes[1, 0].hist(results['original'].flatten(), bins=50, alpha=0.7, color='blue')
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(results['static_processed'].flatten(), bins=50, alpha=0.7, color='green')
        axes[1, 1].set_title('Static Processed Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        axes[1, 2].hist(results['adaptive_processed'].flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 2].set_title('Adaptive Processed Histogram')
        axes[1, 2].set_xlabel('Pixel Intensity')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_metrics_comparison(self, results):
        """Print detailed comparison of metrics"""
        print("="*60)
        print("IMAGE QUALITY ANALYSIS RESULTS")
        print("="*60)
        
        print("\nOriginal Image Quality Metrics:")
        for key, value in results['original_quality'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nEvaluation Metrics Comparison:")
        print(f"Static Pipeline:")
        for key, value in results['static_evaluation'].items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nAdaptive Pipeline:")
        for key, value in results['adaptive_evaluation'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nImprovement (Adaptive vs Static):")
        for key in results['static_evaluation'].keys():
            static_val = results['static_evaluation'][key]
            adaptive_val = results['adaptive_evaluation'][key]
            improvement = ((adaptive_val - static_val) / static_val) * 100 if static_val != 0 else 0
            print(f"  {key}: {improvement:.2f}%")

def main():
    """Main function demonstrating the pipeline usage"""
    pipeline = AdaptivePreprocessingPipeline()
    
    # Example usage (you would replace with actual image paths)
    example_images = [
    r"IS20250115_171841_9465_61003253.dcm"
   ]

    
    print("Adaptive Image Preprocessing Pipeline for IOPA X-rays")
    print("="*60)
    
    for image_path in example_images:
        if os.path.exists(image_path):
            print(f"\nProcessing: {image_path}")
            results = pipeline.process_single_image(image_path)
            
            if results:
                pipeline.print_metrics_comparison(results)
                pipeline.visualize_results(results, 
                                         save_path=f"results_{os.path.basename(image_path)}.png")
            else:
                print(f"Failed to process {image_path}")
        else:
            print(f"File not found: {image_path}")
    
    print("\nPipeline execution completed!")

if __name__ == "__main__":
    main()