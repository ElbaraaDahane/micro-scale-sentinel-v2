"""
Micro-Scale Sentinel - Image Preprocessing Module
Handles image enhancement and physics-based feature extraction.
Optimized for holographic microscopy of microplastics (100-500μm).
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import json
import os

def load_config() -> Dict:
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default values if config missing
        return {
            "preprocessing": {
                "clahe_clip_limit": 3.0,
                "clahe_grid_size": 8,
                "bilateral_diameter": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75
            }
        }

def preprocess_image(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Preprocess holographic microscopy image for analysis.
    
    Steps:
    1. Convert to grayscale if needed
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) - enhances fringes
    3. Bilateral filtering - removes noise while preserving edges
    
    Args:
        image: Input image (RGB or grayscale numpy array)
        debug: If True, returns intermediate steps for visualization
        
    Returns:
        Processed image ready for feature extraction
    """
    config = load_config()
    pp_config = config.get('preprocessing', {})
    
    # Step 1: Ensure grayscale
    # Holographic analysis works on intensity, not color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Step 2: CLAHE (Contrast Enhancement)
    # Why: Holographic fringes often have low contrast
    # CLAHE enhances local contrast without amplifying noise uniformly
    clip_limit = pp_config.get('clahe_clip_limit', 3.0)
    grid_size = pp_config.get('clahe_grid_size', 8)
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    enhanced = clahe.apply(gray)
    
    # Step 3: Bilateral Filtering (Noise Reduction)
    # Why: Removes sensor noise but keeps particle edges sharp
    # Unlike Gaussian blur, bilateral doesn't blur across edges
    d = pp_config.get('bilateral_diameter', 9)
    sigma_color = pp_config.get('bilateral_sigma_color', 75)
    sigma_space = pp_config.get('bilateral_sigma_space', 75)
    
    denoised = cv2.bilateralFilter(
        enhanced, 
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    
    # Convert back to RGB for consistency with rest of pipeline
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    if debug:
        # Return side-by-side comparison for Streamlit display
        # Stack horizontally: Original Gray | Enhanced | Denoised
        comparison = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB),
            processed
        ])
        return comparison
    
    return processed

def detect_particles(image: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect particles in preprocessed image using contour detection.
    
    Physics: Microplastics appear as distinct boundaries with diffraction fringes.
    Contour detection finds closed curves where intensity changes rapidly.
    
    Args:
        image: Preprocessed RGB image
        
    Returns:
        Tuple of (contours, hierarchy)
    """
    # Convert to grayscale for contour detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold to binary
    # Use adaptive thresholding to handle uneven illumination
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    return contours, hierarchy

def analyze_shape(contour: np.ndarray) -> Dict[str, float]:
    """
    Calculate shape descriptors from contour.
    
    Key for microplastic detection:
    - Plastics: Irregular, jagged edges (low circularity, high aspect ratio)
    - Biological: Symmetric, smooth (high circularity, aspect ratio ~1)
    
    Args:
        contour: OpenCV contour points
        
    Returns:
        Dictionary with circularity, aspect_ratio, solidity
    """
    # Area and Perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return {"circularity": 0, "aspect_ratio": 0, "solidity": 0}
    
    # Circularity: 4π*Area/Perimeter² (1.0 = perfect circle)
    # Plastics typically <0.7, Biological >0.8
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    circularity = min(1.0, max(0.0, circularity))  # Clamp to 0-1
    
    # Aspect Ratio using minimum area rectangle
    # Tells us if particle is elongated (fiber) vs round (fragment)
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    
    if width == 0 or height == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = max(width, height) / min(width, height)
    
    # Solidity: Area / Convex Hull Area
    # Measures how "jagged" the edges are (concavities)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return {
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity)
    }

def estimate_refractive_index(image: np.ndarray, contour: np.ndarray) -> float:
    """
    Estimate refractive index from diffraction fringe analysis.
    
    Simplified Physics for Students:
    - Higher RI difference with water creates more visible fringes
    - Plastics (RI 1.5-1.6) create sharper, more numerous fringes than biology (RI 1.33-1.40)
    - We analyze intensity variance around the particle edge as a proxy
    
    Args:
        image: Grayscale image
        contour: Particle contour
        
    Returns:
        Estimated RI (1.3-1.7 range)
    """
    # Create mask for particle region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Create dilated mask for fringe region (area just outside particle)
    # Fringes appear in the immediate vicinity of the particle
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)
    fringe_region = cv2.subtract(mask_dilated, mask)
    
    # Analyze intensity variation in fringe region
    # High variation = visible fringes = likely plastic (higher RI)
    if np.sum(fringe_region) > 0:
        fringe_pixels = image[fringe_region > 0]
        fringe_contrast = np.std(fringe_pixels) / (np.mean(fringe_pixels) + 1e-5)
        
        # Map contrast to RI range
        # Low contrast (biological) -> 1.33-1.40
        # High contrast (plastic) -> 1.45-1.60
        if fringe_contrast < 0.1:
            ri = 1.35
        elif fringe_contrast > 0.3:
            ri = 1.55
        else:
            # Linear interpolation between ranges
            ri = 1.35 + (fringe_contrast - 0.1) * (0.2 / 0.2)
    else:
        ri = 1.33  # Default to water if no fringe detected
    
    return float(ri)

def extract_texture_features(image: np.ndarray, contour: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract texture features (entropy, variance) from particle region.
    
    Entropy measures randomness:
    - Plastics: Uniform texture, lower entropy
    - Biological: Complex internal structures, higher entropy
    
    Args:
        image: Grayscale image
        contour: Optional contour to mask specific region
        
    Returns:
        Dictionary with entropy, variance, mean_intensity
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # If contour provided, mask to particle region only
    if contour is not None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        # Avoid division by zero
        if cv2.countNonZero(mask) == 0:
            roi = gray
        else:
            roi = masked[mask > 0]
    else:
        roi = gray
    
    # Calculate features
    mean_intensity = float(np.mean(roi))
    variance = float(np.var(roi))
    
    # Entropy calculation (Shannon entropy of intensity histogram)
    hist = cv2.calcHist([roi.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    non_zero = hist[hist > 0]
    entropy = float(-np.sum(non_zero * np.log2(non_zero))) if len(non_zero) > 0 else 0.0
    
    return {
        "intensity_mean": mean_intensity,
        "intensity_variance": variance,
        "entropy": entropy,
        "contrast": float(np.std(roi))
    }

def extract_features(image: np.ndarray, scale_um_per_pixel: float = 0.5) -> Dict[str, Any]:
    """
    Complete feature extraction pipeline for particle classification.
    
    This combines all physics-based measurements into a feature dictionary
    that the AI classifier uses to make decisions.
    
    Args:
        image: Preprocessed RGB image
        scale_um_per_pixel: Calibration factor (micrometers per pixel)
        
    Returns:
        Dictionary with all extracted features
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Detect all particles
    contours, _ = detect_particles(image)
    
    if not contours:
        # No particles found - return defaults
        return {
            "particle_id": "P001",
            "size_um": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 1.0,
            "refractive_index_estimate": 1.33,
            "intensity_mean": 128.0,
            "intensity_variance": 0.0,
            "entropy": 0.0,
            "contrast": 0.0,
            "error": "No particles detected"
        }
    
    # Analyze the largest contour (assume main particle of interest)
    # In a real scenario, you might want to analyze all particles > threshold
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Size calculation
    area_pixels = cv2.contourArea(largest_contour)
    size_um = np.sqrt(area_pixels) * scale_um_per_pixel  # Approximate diameter
    
    # Shape analysis
    shape_features = analyze_shape(largest_contour)
    
    # Refractive index estimation from fringe analysis
    ri_estimate = estimate_refractive_index(gray, largest_contour)
    
    # Texture features
    texture_features = extract_texture_features(gray, largest_contour)
    
    # Combine all features
    features = {
        "particle_id": "P001",
        "size_um": float(size_um),
        "circularity": shape_features["circularity"],
        "aspect_ratio": shape_features["aspect_ratio"],
        "solidity": shape_features["solidity"],
        "refractive_index_estimate": ri_estimate,
        **texture_features  # Unpacks intensity_mean, variance, entropy, contrast
    }
    
    return features

# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================
if __name__ == "__main__":
    """
    Run this file directly to test preprocessing:
    python src/preprocessing.py
    
    This will create a test image and show the extracted features.
    """
    import matplotlib.pyplot as plt
    
    print("🔬 Micro-Scale Sentinel - Preprocessing Test")
    print("=" * 50)
    
    # Create a synthetic test image (simulating a plastic particle)
    print("Generating synthetic test image...")
    size = 512
    test_img = np.ones((size, size), dtype=np.uint8) * 240  # Light background
    
    # Draw particle with interference fringes
    center = (size // 2, size // 2)
    radius = 50
    
    # Main particle body (dark)
    cv2.circle(test_img, center, radius, 80, -1)
    
    # Add interference fringes (characteristic of holography)
    for r in range(radius + 5, radius + 40, 6):
        cv2.circle(test_img, center, r, 120, 1)
    
    # Add noise to simulate sensor
    noise = np.random.normal(0, 5, (size, size)).astype(np.int16)
    test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert to RGB (as if loaded from file)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
    
    print("Original image shape:", test_img_rgb.shape)
    
    # Test preprocessing
    print("\nApplying preprocessing...")
    processed = preprocess_image(test_img_rgb, debug=False)
    print("Processed image shape:", processed.shape)
    
    # Test feature extraction
    print("\nExtracting features...")
    features = extract_features(processed, scale_um_per_pixel=0.5)
    
    print("\n📊 Extracted Features:")
    print("-" * 30)
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.3f}")
        else:
            print(f"{key:25s}: {value}")
    
    print("\n✅ Preprocessing module test complete!")
    print("Run 'streamlit run app.py' to use the full application.")
