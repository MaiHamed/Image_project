import cv2
from matplotlib import pyplot as plt
import numpy as np
from descriptor import extract_rectangular_edges , describe_edge_color_pattern, rotate_image_90_times
from paper_algorithms import PaperPuzzleSolver

#==========PHASE 1: IMAGE ENHANCEMENT ==========#

#----------- FILTERING FUNCTIONS -------------
def selective_median_filter(img, threshold=50):
    # Use the optimized OpenCV version instead of manual loops for speed/consistency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    median_gray = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median_gray)
    
    mask = diff > threshold
    result = img.copy()
    
    if img.ndim == 3:
        median_color = cv2.medianBlur(img, 3)
        result[mask] = median_color[mask]
    else:
        result[mask] = median_gray[mask]
    
    return result

def gamma_correction(img, gamma=0.9):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def canny_edges(img, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray_blur, low_threshold, high_threshold)
    return edges

def clahe_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    # Return the enhanced Luminance channel and the original A/B channels
    return l_eq, a, b

def estimate_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def enhance_image(
    img, 
    apply_denoise=False, 
    denoise_threshold=60, 
    gamma=0.9,          
    sharpen_factor=0.08,   
    low_threshold=50, 
    high_threshold=150
):
    
    # 0. Initial Denoising (Optional)
    if apply_denoise:
        # Use the separate selective_median_filter function
        processing_img = selective_median_filter(img, threshold=denoise_threshold)
    else:
        processing_img = img.copy()
        
    # 1. Apply CLAHE (returns L_eq, A, B channels)
    l_eq, a, b = clahe_contrast(processing_img)
    
    # 2. Subtle Sharpening on L-channel (for detail preservation)
    
    # Blur L_eq before Laplacian to target broader features/edges
    blurred_l = cv2.GaussianBlur(l_eq, (3, 3), 0)
    lap = cv2.Laplacian(blurred_l, cv2.CV_32F)
    lap -= lap.mean()
    
    # Sharpen the equalized Luminance channel
    l_sharpened = l_eq.astype(np.float32) - sharpen_factor * lap
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)

    # 3. Merge L, A, B channels and convert back to BGR
    lab_merged = cv2.merge((l_sharpened, a, b))
    enhanced_bgr = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    
    # 4. Lightness Control (Gamma Correction)
    final_enhanced = gamma_correction(enhanced_bgr, gamma=gamma)
    
    # 5. Edge Detection
    edges_bw = canny_edges(final_enhanced, low_threshold, high_threshold)
    
    return final_enhanced, edges_bw

# ------------------ GRID CROPPING ------------------
def detect_grid_size(filename, dirname, default_n=2):
    s = (filename + dirname).lower()
    if "8x8" in s: return 8
    if "4x4" in s: return 4
    if "2x2" in s: return 2
    return default_n

def extract_generic_grid_pieces(img, N=2):
    if img is None:
        return []

    h, w = img.shape[:2]
    step_y, step_x = h // N, w // N
    pieces = []

    for r in range(N):
        for c in range(N):
            y1, y2 = r * step_y, (r + 1) * step_y
            x1, x2 = c * step_x, (c + 1) * step_x
            if r == N - 1: y2 = h
            if c == N - 1: x2 = w
            pieces.append(img[y1:y2, x1:x2].copy())

    return pieces

# ------------------ ASSEMBLY FUNCTIONS ------------------
def assemble_grid_from_pieces(pieces, grid, rotations=None, N=2):
    """
    Assemble pieces into a final image based on grid and rotations
    """
    if not pieces:
        return None
    
    # Get piece dimensions
    piece_h, piece_w = pieces[0].shape[:2]
    
    # Create empty canvas
    result_h = N * piece_h
    result_w = N * piece_w
    if len(pieces[0].shape) == 3:
        result = np.zeros((result_h, result_w, 3), dtype=np.uint8)
    else:
        result = np.zeros((result_h, result_w), dtype=np.uint8)
    
    # Place each piece
    for r in range(N):
        for c in range(N):
            piece_idx = grid[r][c]
            if piece_idx is None or piece_idx >= len(pieces):
                continue
                
            piece_img = pieces[piece_idx].copy()
            
            # Apply rotation if specified
            if rotations and piece_idx < len(rotations):
                rotation = rotations[piece_idx]
                # Handle both dict and int rotation formats
                if isinstance(rotation, dict):
                    rotation_angle = get_rotation_from_rotations_dict(rotation)
                else:
                    rotation_angle = rotation
                
                if rotation_angle != 0:
                    piece_img = rotate_image_90_times(piece_img, rotation_angle)
            
            # Calculate position
            y_start = r * piece_h
            y_end = y_start + piece_h
            x_start = c * piece_w
            x_end = x_start + piece_w
            
            result[y_start:y_end, x_start:x_end] = piece_img
    
    return result

def get_rotation_from_rotations_dict(rotations_dict):
    """
    Extract rotation angle from rotations dictionary structure
    """
    if isinstance(rotations_dict, dict):
        # Find which rotation is selected (usually 0 or the first key)
        if 0 in rotations_dict:
            return 0
        elif len(rotations_dict) > 0:
            first_key = list(rotations_dict.keys())[0]
            return first_key // 90  # Convert angle to number of 90° rotations
    elif isinstance(rotations_dict, (int, float)):
        # Already a rotation angle
        return int(rotations_dict) // 90 if rotations_dict >= 90 else int(rotations_dict)
    
    # Default: no rotation
    return 0
