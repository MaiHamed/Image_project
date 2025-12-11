import cv2
import numpy as np
from paper_algorithms import PaperPuzzleSolver

# ------------------ FILTERING FUNCTIONS ------------------
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

#---------------- DESCRIPTOR FUNCTIONS (KEPT FOR BACKWARD COMPATIBILITY) -----------------

def rotate_image_90_times(img, k):
    k = k % 4
    if k == 0:
        return img.copy()
    elif k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif k == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def extract_rectangular_edges(piece_img):
    if piece_img is None:
        return {}

    h, w = piece_img.shape[:2]
    return {
        'top': piece_img[0, :, :].copy(),
        'bottom': piece_img[-1, :, :].copy(),
        'left': piece_img[:, 0, :].copy(),
        'right': piece_img[:, -1, :].copy()
    }

def describe_edge_color_pattern(edge_pixels, target_length=100):
    """
    IMPROVED edge descriptor with multiple features
    """
    if edge_pixels is None or len(edge_pixels) == 0:
        return np.array([])
    
    # Handle both color and grayscale
    if len(edge_pixels.shape) > 1 and edge_pixels.shape[1] == 3:
        # Color image - extract multiple features
        # 1. Grayscale intensity
        gray = 0.299 * edge_pixels[:, 0] + 0.587 * edge_pixels[:, 1] + 0.114 * edge_pixels[:, 2]
        
        # 2. Color channels separately
        r_channel = edge_pixels[:, 0].astype(np.float32)
        g_channel = edge_pixels[:, 1].astype(np.float32)
        b_channel = edge_pixels[:, 2].astype(np.float32)
        
        # 3. Edge gradients (for texture)
        if len(gray) > 1:
            grad = np.gradient(gray)
        else:
            grad = np.zeros_like(gray)
        
        # Combine features (normalize each)
        features = []
        
        # Grayscale intensity
        if len(gray) > 0:
            gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
            features.append(gray_norm)
        
        # Color ratios (capture color patterns)
        color_sum = r_channel + g_channel + b_channel + 1e-10
        r_ratio = r_channel / color_sum
        g_ratio = g_channel / color_sum
        
        if len(r_ratio) > 0:
            features.append(r_ratio)
            features.append(g_ratio)
        
        # Gradient (edge information)
        if len(grad) > 0:
            grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-10)
            features.append(grad_norm)
        
        # Interpolate each feature to target length
        combined = []
        x_old = np.linspace(0, 1, len(gray))
        x_new = np.linspace(0, 1, target_length)
        
        for feat in features:
            if len(feat) > 1:
                feat_interp = np.interp(x_new, x_old, feat)
                combined.append(feat_interp)
        
        if combined:
            # Stack features and normalize
            combined_array = np.vstack(combined).mean(axis=0)  # Average features
            if combined_array.max() > combined_array.min():
                combined_array = (combined_array - combined_array.min()) / (combined_array.max() - combined_array.min())
            return combined_array
        else:
            return np.zeros(target_length)
    
    else:
        # Grayscale image
        intensities = edge_pixels.flatten().astype(np.float32)
        
        if len(intensities) < 2:
            return np.zeros(target_length)
        
        # Add gradient feature for grayscale too
        if len(intensities) > 1:
            grad = np.gradient(intensities)
        else:
            grad = np.zeros_like(intensities)
        
        # Normalize intensity
        if intensities.max() > intensities.min():
            intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        else:
            intensities_norm = np.zeros_like(intensities)
        
        # Normalize gradient
        if grad.max() > grad.min():
            grad_norm = (grad - grad.min()) / (grad.max() - grad.min())
        else:
            grad_norm = np.zeros_like(grad)
        
        # Combine intensity and gradient (weighted)
        combined = 0.7 * intensities_norm + 0.3 * grad_norm
        
        # Interpolate to target length
        x_old = np.linspace(0, 1, len(combined))
        x_new = np.linspace(0, 1, target_length)
        normalized = np.interp(x_new, x_old, combined)
        
        # Final normalization
        if normalized.max() > normalized.min():
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        
        return normalized

# ========== NEW PAPER-BASED ANALYSIS FUNCTIONS ==========

def analyze_all_possible_matches_paper_based(all_piece_images, piece_files, N):
    """
    NEW: Use simplified paper's algorithms for matching
    """
    print(f"\n📊 Running paper-based puzzle solver on {len(all_piece_images)} pieces")
    
    # Initialize solver
    solver = PaperPuzzleSolver(p=0.3, q=1/16, use_prediction=True, border_width=10)
    
    # Solve puzzle
    final_grid, compatibility_matrix, best_buddies = solver.solve(all_piece_images)
    
    # Convert to comparison format for visualization
    all_comparisons = []
    num_pieces = len(all_piece_images)
    
    # Create comparisons from compatibility matrix
    relations = ['right', 'left', 'top', 'bottom']
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j:
                continue
            for rel_idx, rel in enumerate(relations):
                opp_rel = {'right': 'left', 'left': 'right', 
                          'top': 'bottom', 'bottom': 'top'}[rel]
                
                all_comparisons.append({
                    'piece1': i,
                    'piece2': j,
                    'edge1': rel,
                    'edge2': opp_rel,
                    'rotation_of_piece2': 0,
                    'score': float(compatibility_matrix[i, j, rel_idx]),
                    'label': f"P{i+1} {rel} ↔ P{j+1} {opp_rel}",
                    'method': 'paper'
                })
    
    # Sort by score
    all_comparisons.sort(key=lambda x: x['score'], reverse=True)
    
    # Keep rotation structure for compatibility (optional)
    all_piece_rotations = []
    for p_img in all_piece_images:
        rotations = {}
        for k in range(4):
            angle = k * 90
            img_rot = rotate_image_90_times(p_img, k)
            raw_edges = extract_rectangular_edges(img_rot)
            descriptors = {
                e: describe_edge_color_pattern(raw_edges[e])
                for e in raw_edges
            }
            rotations[angle] = {
                'image': img_rot,
                'descriptors': descriptors
            }
        all_piece_rotations.append(rotations)
    
    print(f"✅ Paper solver completed. Found {len(best_buddies)} best-buddy pairs")
    
    return all_comparisons, all_piece_rotations, final_grid, best_buddies

def analyze_all_possible_matches_rotation_aware(all_piece_images, piece_files, N):
    """
    KEPT FOR BACKWARD COMPATIBILITY
    Calls the new paper-based function but returns same format
    """
    return analyze_all_possible_matches_paper_based(all_piece_images, piece_files, N)

def debug_edge_descriptors(piece_images):
    """
    Debug function to check if edge descriptors are working properly
    """
    print("\n🔍 DEBUG: Edge Descriptor Analysis")
    print("-" * 50)
    
    for i, img in enumerate(piece_images[:2]):  # Check first 2 pieces
        print(f"\nPiece {i+1} (shape: {img.shape})")
        
        # Get edges
        edges = extract_rectangular_edges(img)
        
        for edge_name, edge_pixels in edges.items():
            # Show edge statistics
            print(f"  {edge_name}: shape={edge_pixels.shape}")
            
            # Get descriptor
            desc = describe_edge_color_pattern(edge_pixels)
            print(f"    Descriptor length: {len(desc)}")
            print(f"    Descriptor range: [{desc.min():.3f}, {desc.max():.3f}]")
            print(f"    Descriptor mean: {desc.mean():.3f}")
            
            # Show first few values
            if len(desc) > 0:
                print(f"    First 5 values: {desc[:5]}")
                
def analyze_all_possible_matches_paper_based(all_piece_images, piece_files, N):
    """
    NEW: Use paper's algorithms for matching
    Replaces the old rotation-aware analysis
    """
    print(f"\n📊 Running paper-based puzzle solver on {len(all_piece_images)} pieces")
    
    # Initialize solver with paper's optimal parameters
    solver = PaperPuzzleSolver(p=0.3, q=1/16, use_prediction=True)
    
    # Solve puzzle using paper's complete pipeline
    final_grid, compatibility_matrix, best_buddies, assembled = solver.solve(all_piece_images)
    
    # DEBUG: Check what's returned
    print(f"DEBUG: compatibility_matrix shape: {compatibility_matrix.shape if hasattr(compatibility_matrix, 'shape') else 'No shape'}")
    print(f"DEBUG: final_grid type: {type(final_grid)}")
    
    # Convert grid to list of comparisons for compatibility with existing code
    all_comparisons = []
    num_pieces = len(all_piece_images)
    
    # Check if compatibility_matrix is valid
    if compatibility_matrix is None or (hasattr(compatibility_matrix, 'shape') and 0 in compatibility_matrix.shape):
        print("⚠️ WARNING: compatibility_matrix is empty or invalid")
        # Create a dummy comparison list
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                for rel_idx, rel in enumerate(['right', 'left', 'top', 'bottom']):
                    opp_rel = {'right': 'left', 'left': 'right', 
                              'top': 'bottom', 'bottom': 'top'}[rel]
                    
                    all_comparisons.append({
                        'piece1': i,
                        'piece2': j,
                        'edge1': rel,
                        'edge2': opp_rel,
                        'rotation_of_piece2': 0,
                        'score': 0.5,  # Default score
                        'label': f"P{i+1} {rel} ↔ P{j+1} {opp_rel}",
                        'method': 'paper'
                    })
    else:
        # Generate comparison list from compatibility matrix
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                for rel_idx, rel in enumerate(['right', 'left', 'top', 'bottom']):
                    opp_rel = {'right': 'left', 'left': 'right', 
                              'top': 'bottom', 'bottom': 'top'}[rel]
                    
                    score = compatibility_matrix[i, j, rel_idx]
                    
                    all_comparisons.append({
                        'piece1': i,
                        'piece2': j,
                        'edge1': rel,
                        'edge2': opp_rel,
                        'rotation_of_piece2': 0,  # Paper handles rotation in compatibility
                        'score': float(score),
                        'label': f"P{i+1} {rel} ↔ P{j+1} {opp_rel}",
                        'method': 'paper'
                    })
    
    # Sort by score (higher = better in paper's metric)
    all_comparisons.sort(key=lambda x: x['score'], reverse=True)
    
    # Create rotation data structure for compatibility (optional - can remove if not needed)
    all_piece_rotations = []
    for p_img in all_piece_images:
        rotations = {}
        for k in range(4):
            angle = k * 90
            img_rot = rotate_image_90_times(p_img, k)
            raw_edges = extract_rectangular_edges(img_rot)
            descriptors = {
                e: describe_edge_color_pattern(raw_edges[e])
                for e in raw_edges
            }
            rotations[angle] = {
                'image': img_rot,
                'descriptors': descriptors
            }
        all_piece_rotations.append(rotations)
    
    print(f"✅ Paper solver completed. Found {len(best_buddies)} best-buddy pairs")
    
    return all_comparisons, all_piece_rotations, final_grid, best_buddies            