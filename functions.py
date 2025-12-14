import cv2
from matplotlib import pyplot as plt
import numpy as np
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

#---------------- DESCRIPTOR FUNCTIONS -----------------

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

# ==========  PAPER-BASED ANALYSIS FUNCTIONS ==========

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

# ------------------ ASSEMBLY FUNCTIONS ------------------

def assemble_puzzle_from_comparisons(pieces, comparisons, N, top_n_matches=10):
    """
    Assemble puzzle using descriptor-based comparisons
    """
    print(f"\n🧩 Assembling {N}x{N} puzzle from {len(comparisons)} comparisons")
    
    # Create piece objects with rotations
    piece_rotations = []
    for piece_img in pieces:
        rotations = {}
        for k in range(4):
            angle = k * 90
            img_rot = rotate_image_90_times(piece_img, k)
            raw_edges = extract_rectangular_edges(img_rot)
            descriptors = {
                e: describe_edge_color_pattern(raw_edges[e])
                for e in raw_edges
            }
            rotations[angle] = {
                'image': img_rot,
                'descriptors': descriptors
            }
        piece_rotations.append(rotations)
    
    # Filter top matches
    top_comparisons = sorted(comparisons, key=lambda x: x['score'], reverse=True)[:top_n_matches]
    
    # Create compatibility matrix
    num_pieces = len(pieces)
    compatibility_matrix = np.zeros((num_pieces, num_pieces, 4))
    
    for comp in top_comparisons:
        i = comp['piece1']
        j = comp['piece2']
        edge_map = {'right': 0, 'left': 1, 'top': 2, 'bottom': 3}
        if comp['edge1'] in edge_map:
            rel_idx = edge_map[comp['edge1']]
            compatibility_matrix[i, j, rel_idx] = comp['score']
    
    # Simple assembly using best matches
    grid = [[None for _ in range(N)] for _ in range(N)]
    placed = [False] * num_pieces
    
    # Start with first piece at top-left
    grid[0][0] = 0
    placed[0] = True
    
    # Place pieces row by row
    for r in range(N):
        for c in range(N):
            if grid[r][c] is not None:
                continue
            
            # Find piece that matches neighbors
            best_piece = None
            best_score = -1
            best_rotation = 0
            
            for p in range(num_pieces):
                if placed[p]:
                    continue
                    
                # Try all rotations
                for rot in range(4):
                    score = 0
                    matches = 0
                    
                    # Check left neighbor
                    if c > 0 and grid[r][c-1] is not None:
                        left_piece = grid[r][c-1]
                        comp_score = compatibility_matrix[p, left_piece, 1]  # left relation
                        if comp_score > 0:
                            score += comp_score
                            matches += 1
                    
                    # Check top neighbor
                    if r > 0 and grid[r-1][c] is not None:
                        top_piece = grid[r-1][c]
                        comp_score = compatibility_matrix[p, top_piece, 3]  # bottom relation
                        if comp_score > 0:
                            score += comp_score
                            matches += 1
                    
                    if matches > 0:
                        avg_score = score / matches
                        if avg_score > best_score:
                            best_score = avg_score
                            best_piece = p
                            best_rotation = rot
            
            if best_piece is not None:
                grid[r][c] = best_piece
                placed[best_piece] = True
    
    # Fill any remaining positions
    for r in range(N):
        for c in range(N):
            if grid[r][c] is None:
                for p in range(num_pieces):
                    if not placed[p]:
                        grid[r][c] = p
                        placed[p] = True
                        break
    
    return grid, piece_rotations, compatibility_matrix

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

def reassemble_for_paper_algorithm(grid, N):
    """
    For paper algorithm: bottom part should be on top of upper part
    This swaps rows to match the paper's assembly
    """
    # Create a copy of the grid
    new_grid = [row.copy() for row in grid]
    
    # For a 2x2 puzzle, swap row 0 and row 1
    if N == 2:
        new_grid[0], new_grid[1] = new_grid[1], new_grid[0]
    elif N == 4:
        # For 4x4: swap rows 0↔3 and 1↔2
        new_grid[0], new_grid[3] = new_grid[3], new_grid[0]
        new_grid[1], new_grid[2] = new_grid[2], new_grid[1]
    elif N == 8:
        # For 8x8: reverse the rows
        new_grid = list(reversed(new_grid))
    
    return new_grid

# ------------------ EVALUATION FUNCTIONS ------------------
def evaluate_corner_compatibility_descriptor(pieces, grid, N, all_comparisons=None):
    """
    Evaluate how well corners match using descriptor-based edge compatibility
    Higher score means better corner matching
    """
    if not pieces:
        return 0
    
    score = 0
    total_corners = 0
    
    # If we have descriptor comparisons, use them for more accurate matching
    if all_comparisons and len(all_comparisons) > 0:
        # Create a dictionary for quick lookup of edge compatibility
        edge_compatibility = {}
        for comp in all_comparisons:
            key = (comp['piece1'], comp['piece2'], comp['edge1'], comp['edge2'])
            edge_compatibility[key] = comp['score']
    
    # For each corner piece, check if it has matching neighbors
    for r in [0, N-1]:
        for c in [0, N-1]:
            total_corners += 1
            piece_idx = grid[r][c]
            if piece_idx is None or piece_idx >= len(pieces):
                continue
            
            corner_score = 0
            neighbor_count = 0
            
            # Check right neighbor (if exists)
            if c < N-1:
                right_idx = grid[r][c+1]
                if right_idx is not None:
                    neighbor_count += 1
                    if all_comparisons:
                        # Look for descriptor compatibility score
                        for comp in all_comparisons:
                            if (comp['piece1'] == piece_idx and comp['piece2'] == right_idx and 
                                comp['edge1'] == 'right' and comp['edge2'] == 'left'):
                                corner_score += comp['score']
                                break
                        else:
                            # If no descriptor match found, use edge color similarity
                            piece = pieces[piece_idx]
                            right_piece = pieces[right_idx]
                            left_edge = piece[:, -1]  # Right edge of left piece
                            right_edge = right_piece[:, 0]  # Left edge of right piece
                            if left_edge.shape == right_edge.shape:
                                diff = np.mean(np.abs(left_edge.astype(float) - right_edge.astype(float)))
                                corner_score += 1.0 / (1.0 + diff/255.0)
                    else:
                        # Use edge color similarity
                        piece = pieces[piece_idx]
                        right_piece = pieces[right_idx]
                        left_edge = piece[:, -1]
                        right_edge = right_piece[:, 0]
                        if left_edge.shape == right_edge.shape:
                            diff = np.mean(np.abs(left_edge.astype(float) - right_edge.astype(float)))
                            corner_score += 1.0 / (1.0 + diff/255.0)
            
            # Check bottom neighbor (if exists)
            if r < N-1:
                bottom_idx = grid[r+1][c]
                if bottom_idx is not None:
                    neighbor_count += 1
                    if all_comparisons:
                        # Look for descriptor compatibility score
                        for comp in all_comparisons:
                            if (comp['piece1'] == piece_idx and comp['piece2'] == bottom_idx and 
                                comp['edge1'] == 'bottom' and comp['edge2'] == 'top'):
                                corner_score += comp['score']
                                break
                        else:
                            # If no descriptor match found, use edge color similarity
                            piece = pieces[piece_idx]
                            bottom_piece = pieces[bottom_idx]
                            top_edge = piece[-1, :]  # Bottom edge of top piece
                            bottom_edge = bottom_piece[0, :]  # Top edge of bottom piece
                            if top_edge.shape == bottom_edge.shape:
                                diff = np.mean(np.abs(top_edge.astype(float) - bottom_edge.astype(float)))
                                corner_score += 1.0 / (1.0 + diff/255.0)
                    else:
                        # Use edge color similarity
                        piece = pieces[piece_idx]
                        bottom_piece = pieces[bottom_idx]
                        top_edge = piece[-1, :]
                        bottom_edge = bottom_piece[0, :]
                        if top_edge.shape == bottom_edge.shape:
                            diff = np.mean(np.abs(top_edge.astype(float) - bottom_edge.astype(float)))
                            corner_score += 1.0 / (1.0 + diff/255.0)
            
            # Average the corner score
            if neighbor_count > 0:
                score += corner_score / neighbor_count
    
    return score / max(1, total_corners)

def evaluate_grid_compatibility(pieces, grid, N, all_comparisons=None):
    """
    Evaluate overall grid compatibility using descriptor-based matching
    Checks all adjacent pieces in the grid
    """
    if not pieces:
        return 0
    
    total_score = 0
    total_edges = 0
    
    # Check horizontal edges
    for r in range(N):
        for c in range(N-1):
            piece1_idx = grid[r][c]
            piece2_idx = grid[r][c+1]
            
            if piece1_idx is not None and piece2_idx is not None:
                total_edges += 1
                
                if all_comparisons:
                    # Look for descriptor compatibility score
                    for comp in all_comparisons:
                        if (comp['piece1'] == piece1_idx and comp['piece2'] == piece2_idx and 
                            comp['edge1'] == 'right' and comp['edge2'] == 'left'):
                            total_score += comp['score']
                            break
                    else:
                        # Fall back to edge similarity
                        piece1 = pieces[piece1_idx]
                        piece2 = pieces[piece2_idx]
                        left_edge = piece1[:, -1]
                        right_edge = piece2[:, 0]
                        if left_edge.shape == right_edge.shape:
                            diff = np.mean(np.abs(left_edge.astype(float) - right_edge.astype(float)))
                            total_score += 1.0 / (1.0 + diff/255.0)
                else:
                    # Use edge color similarity
                    piece1 = pieces[piece1_idx]
                    piece2 = pieces[piece2_idx]
                    left_edge = piece1[:, -1]
                    right_edge = piece2[:, 0]
                    if left_edge.shape == right_edge.shape:
                        diff = np.mean(np.abs(left_edge.astype(float) - right_edge.astype(float)))
                        total_score += 1.0 / (1.0 + diff/255.0)
    
    # Check vertical edges
    for r in range(N-1):
        for c in range(N):
            piece1_idx = grid[r][c]
            piece2_idx = grid[r+1][c]
            
            if piece1_idx is not None and piece2_idx is not None:
                total_edges += 1
                
                if all_comparisons:
                    # Look for descriptor compatibility score
                    for comp in all_comparisons:
                        if (comp['piece1'] == piece1_idx and comp['piece2'] == piece2_idx and 
                            comp['edge1'] == 'bottom' and comp['edge2'] == 'top'):
                            total_score += comp['score']
                            break
                    else:
                        # Fall back to edge similarity
                        piece1 = pieces[piece1_idx]
                        piece2 = pieces[piece2_idx]
                        top_edge = piece1[-1, :]
                        bottom_edge = piece2[0, :]
                        if top_edge.shape == bottom_edge.shape:
                            diff = np.mean(np.abs(top_edge.astype(float) - bottom_edge.astype(float)))
                            total_score += 1.0 / (1.0 + diff/255.0)
                else:
                    # Use edge color similarity
                    piece1 = pieces[piece1_idx]
                    piece2 = pieces[piece2_idx]
                    top_edge = piece1[-1, :]
                    bottom_edge = piece2[0, :]
                    if top_edge.shape == bottom_edge.shape:
                        diff = np.mean(np.abs(top_edge.astype(float) - bottom_edge.astype(float)))
                        total_score += 1.0 / (1.0 + diff/255.0)
    
    return total_score / max(1, total_edges)

def choose_best_orientation_descriptor(pieces, orientations, N, all_comparisons):
    """
    Choose the best orientation using descriptor-based compatibility
    """
    best_orientation = None
    best_combined_score = -1
    best_corner_score = -1
    best_grid_score = -1
    
    print(f"   🔍 Evaluating {len(orientations)} orientations using descriptors:")
    
    for i, orientation in enumerate(orientations):
        # Evaluate corner compatibility
        corner_score = evaluate_corner_compatibility_descriptor(pieces, orientation['grid'], N, all_comparisons)
        
        # Evaluate overall grid compatibility
        grid_score = evaluate_grid_compatibility(pieces, orientation['grid'], N, all_comparisons)
        
        # Combined score (weighted average)
        combined_score = 0.4 * corner_score + 0.6 * grid_score
        
        print(f"     {i+1}. {orientation['name']}:")
        print(f"        Corner score: {corner_score:.3f}")
        print(f"        Grid score: {grid_score:.3f}")
        print(f"        Combined: {combined_score:.3f}")
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_corner_score = corner_score
            best_grid_score = grid_score
            best_orientation = orientation
    
    if best_orientation:
        print(f"   ✅ Selected: {best_orientation['name']}")
        print(f"      Corner score: {best_corner_score:.3f}, Grid score: {best_grid_score:.3f}, Combined: {best_combined_score:.3f}")
    
    return best_orientation, best_corner_score, best_grid_score

def create_descriptor_compatibility_matrix(pieces, all_comparisons, N):
    """
    Create a compatibility matrix from descriptor comparisons
    """
    num_pieces = len(pieces)
    compat_matrix = np.zeros((num_pieces, num_pieces, 4))
    
    # Map edge names to indices
    edge_map = {'right': 0, 'left': 1, 'top': 2, 'bottom': 3}
    
    for comp in all_comparisons:
        i = comp['piece1']
        j = comp['piece2']
        if comp['edge1'] in edge_map:
            rel_idx = edge_map[comp['edge1']]
            compat_matrix[i, j, rel_idx] = comp['score']
    
    return compat_matrix

def evaluate_grid_with_matrix(grid, compat_matrix):
    """
    Evaluate grid using precomputed compatibility matrix
    """
    N = len(grid)
    total_score = 0
    total_edges = 0
    
    # Check horizontal edges
    for r in range(N):
        for c in range(N-1):
            i = grid[r][c]
            j = grid[r][c+1]
            if i is not None and j is not None:
                total_edges += 1
                total_score += compat_matrix[i, j, 0]  # right->left
    
    # Check vertical edges
    for r in range(N-1):
        for c in range(N):
            i = grid[r][c]
            j = grid[r+1][c]
            if i is not None and j is not None:
                total_edges += 1
                total_score += compat_matrix[i, j, 3]  # bottom->top
    
    return total_score / max(1, total_edges)

def optimize_grid_orientation(pieces, grid, N, all_comparisons):
    """
    Try all possible grid orientations and rotations to find the best one
    """
    if not all_comparisons:
        return grid, "Original", 0
    
    # Create compatibility matrix
    compat_matrix = create_descriptor_compatibility_matrix(pieces, all_comparisons, N)
    
    best_grid = grid
    best_score = -1
    best_orientation = "Original"
    
    # Generate all possible orientations
    orientations = reassemble_grid_all_orientations(grid, N)
    
    # Also try rotating the entire grid (if pieces are rotated differently)
    for orientation in orientations:
        current_grid = orientation['grid']
        
        # Evaluate this orientation
        score = evaluate_grid_with_matrix(current_grid, compat_matrix)
        
        print(f"   Testing {orientation['name']}: score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_grid = current_grid
            best_orientation = orientation['name']
    
    return best_grid, best_orientation, best_score

def reassemble_grid_all_orientations(grid, N):
    """
    Test all four orientations for the grid:
    1. Original (no change)
    2. Bottom → Top (swap rows)
    3. Left → Right (swap columns)
    4. Both swaps (rows and columns)
    """
    orientations = []
    
    # 1. Original orientation
    orientations.append({
        'name': 'Original',
        'grid': [row.copy() for row in grid],
        'description': 'No change'
    })
    
    # 2. Bottom → Top (swap rows)
    new_grid = [row.copy() for row in grid]
    if N == 2:
        new_grid[0], new_grid[1] = new_grid[1], new_grid[0]
    elif N == 4:
        new_grid[0], new_grid[3] = new_grid[3], new_grid[0]
        new_grid[1], new_grid[2] = new_grid[2], new_grid[1]
    elif N == 8:
        new_grid = list(reversed(new_grid))
    orientations.append({
        'name': 'Bottom→Top',
        'grid': new_grid,
        'description': 'Rows swapped (bottom moved to top)'
    })
    
    # 3. Left → Right (swap columns)
    new_grid = [row.copy() for row in grid]
    for r in range(N):
        if N == 2:
            new_grid[r][0], new_grid[r][1] = new_grid[r][1], new_grid[r][0]
        elif N == 4:
            new_grid[r][0], new_grid[r][3] = new_grid[r][3], new_grid[r][0]
            new_grid[r][1], new_grid[r][2] = new_grid[r][2], new_grid[r][1]
        elif N == 8:
            new_grid[r] = list(reversed(new_grid[r]))
    orientations.append({
        'name': 'Left→Right',
        'grid': new_grid,
        'description': 'Columns swapped (left moved to right)'
    })
    
    # 4. Both Bottom→Top and Left→Right
    new_grid = [row.copy() for row in grid]
    # First swap rows
    if N == 2:
        new_grid[0], new_grid[1] = new_grid[1], new_grid[0]
    elif N == 4:
        new_grid[0], new_grid[3] = new_grid[3], new_grid[0]
        new_grid[1], new_grid[2] = new_grid[2], new_grid[1]
    elif N == 8:
        new_grid = list(reversed(new_grid))
    # Then swap columns
    for r in range(N):
        if N == 2:
            new_grid[r][0], new_grid[r][1] = new_grid[r][1], new_grid[r][0]
        elif N == 4:
            new_grid[r][0], new_grid[r][3] = new_grid[r][3], new_grid[r][0]
            new_grid[r][1], new_grid[r][2] = new_grid[r][2], new_grid[r][1]
        elif N == 8:
            new_grid[r] = list(reversed(new_grid[r]))
    orientations.append({
        'name': 'Both',
        'grid': new_grid,
        'description': 'Both rows and columns swapped'
    })
    
    return orientations

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

def evaluate_corner_compatibility(pieces, grid, N):
    """
    Evaluate how well corners match in the assembled grid
    Higher score means better corner matching
    """
    if not pieces:
        return 0
    
    score = 0
    total_corners = 0
    
    # Get piece dimensions
    piece_h, piece_w = pieces[0].shape[:2]
    
    # For each corner piece, check if it has matching neighbors
    for r in [0, N-1]:
        for c in [0, N-1]:
            total_corners += 1
            piece_idx = grid[r][c]
            if piece_idx is None or piece_idx >= len(pieces):
                continue
                
            # Get the piece
            piece = pieces[piece_idx]
            
            # Check if this corner piece has smooth transitions to neighbors
            # Top-left corner (0,0): should have right and bottom neighbors
            if r == 0 and c == 0:
                if N > 1:
                    # Check right neighbor
                    right_idx = grid[0][1] if 1 < N else None
                    if right_idx is not None:
                        # Simple check: compare edge colors
                        left_edge = piece[:, -1]  # Right edge of left piece
                        right_piece = pieces[right_idx]
                        right_edge = right_piece[:, 0]  # Left edge of right piece
                        if left_edge.shape == right_edge.shape:
                            diff = np.mean(np.abs(left_edge.astype(float) - right_edge.astype(float)))
                            score += 1.0 / (1.0 + diff/255.0)  # Higher diff = lower score
            
            # Top-right corner (0,N-1): should have left and bottom neighbors
            elif r == 0 and c == N-1:
                if N > 1:
                    # Check left neighbor
                    left_idx = grid[0][N-2] if N-2 >= 0 else None
                    if left_idx is not None:
                        right_edge = piece[:, 0]  # Left edge of right piece
                        left_piece = pieces[left_idx]
                        left_edge = left_piece[:, -1]  # Right edge of left piece
                        if left_edge.shape == right_edge.shape:
                            diff = np.mean(np.abs(left_edge.astype(float) - right_edge.astype(float)))
                            score += 1.0 / (1.0 + diff/255.0)
    
    return score / max(1, total_corners)

def choose_best_orientation(pieces, orientations, N):
    """
    Choose the best orientation based on corner compatibility
    """
    best_orientation = None
    best_score = -1
    
    print(f"   🔍 Evaluating {len(orientations)} orientations:")
    
    for i, orientation in enumerate(orientations):
        score = evaluate_corner_compatibility(pieces, orientation['grid'], N)
        print(f"     {i+1}. {orientation['name']}: score = {score:.3f} ({orientation['description']})")
        
        if score > best_score:
            best_score = score
            best_orientation = orientation
    
    if best_orientation:
        print(f"   ✅ Selected: {best_orientation['name']} (score: {best_score:.3f})")
    
    return best_orientation, best_score

def evaluate_edge_similarity_direct(piece1, piece2, edge1_side, edge2_side):
    """
    Direct edge similarity evaluation without descriptors
    """
    # Extract edges
    edges1 = extract_rectangular_edges(piece1)
    edges2 = extract_rectangular_edges(piece2)
    
    if edge1_side not in edges1 or edge2_side not in edges2:
        return 0
    
    edge1 = edges1[edge1_side]
    edge2 = edges2[edge2_side]
    
    if edge1.shape != edge2.shape:
        # Resize if needed
        min_len = min(edge1.shape[0], edge2.shape[0])
        edge1 = edge1[:min_len]
        edge2 = edge2[:min_len]
    
    # Convert to grayscale for comparison
    if len(edge1.shape) == 3:
        gray1 = cv2.cvtColor(edge1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(edge2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = edge1
        gray2 = edge2
    
    # Calculate multiple similarity metrics
    # 1. Mean absolute difference (inverted)
    mae = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
    mae_score = 1.0 - (mae / 255.0)
    
    # 2. Structural similarity (if we have scikit-image)
    try:
        from skimage.metrics import structural_similarity as ssim
        # Ensure images are 2D
        if gray1.ndim == 1:
            gray1 = gray1.reshape(-1, 1)
        if gray2.ndim == 1:
            gray2 = gray2.reshape(-1, 1)
        
        # Resize to same dimensions
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        h = min(h1, h2)
        w = min(w1, w2)
        gray1_resized = cv2.resize(gray1, (w, h)) if h1 != h or w1 != w else gray1
        gray2_resized = cv2.resize(gray2, (w, h)) if h2 != h or w2 != w else gray2
        
        ssim_score = ssim(gray1_resized, gray2_resized, data_range=255)
    except:
        ssim_score = mae_score
    
    # 3. Edge gradient correlation
    grad1 = np.gradient(gray1.astype(float))
    grad2 = np.gradient(gray2.astype(float))
    
    if isinstance(grad1, tuple):  # For 2D gradients
        grad1_mag = np.sqrt(grad1[0]**2 + grad1[1]**2) if len(grad1) > 1 else np.abs(grad1[0])
        grad2_mag = np.sqrt(grad2[0]**2 + grad2[1]**2) if len(grad2) > 1 else np.abs(grad2[0])
    else:
        grad1_mag = np.abs(grad1)
        grad2_mag = np.abs(grad2)
    
    # Normalize gradients
    grad1_norm = (grad1_mag - grad1_mag.min()) / (grad1_mag.max() - grad1_mag.min() + 1e-10)
    grad2_norm = (grad2_mag - grad2_mag.min()) / (grad2_mag.max() - grad2_mag.min() + 1e-10)
    
    # Calculate correlation
    if len(grad1_norm) > 1 and len(grad2_norm) > 1:
        corr = np.corrcoef(grad1_norm.flatten(), grad2_norm.flatten())[0, 1]
        if np.isnan(corr):
            corr = 0
        corr_score = (corr + 1) / 2  # Convert from [-1, 1] to [0, 1]
    else:
        corr_score = 0
    
    # Combined score (weighted average)
    combined = 0.4 * mae_score + 0.4 * ssim_score + 0.2 * corr_score
    
    return max(0, min(1, combined))

def evaluate_grid_compatibility_direct(pieces, grid, N):
    """
    Evaluate grid compatibility using direct edge similarity
    """
    if not pieces:
        return 0
    
    total_score = 0
    total_edges = 0
    
    # Check horizontal edges
    for r in range(N):
        for c in range(N-1):
            piece1_idx = grid[r][c]
            piece2_idx = grid[r][c+1]
            
            if piece1_idx is not None and piece2_idx is not None:
                score = evaluate_edge_similarity_direct(
                    pieces[piece1_idx], 
                    pieces[piece2_idx], 
                    'right', 
                    'left'
                )
                total_score += score
                total_edges += 1
    
    # Check vertical edges
    for r in range(N-1):
        for c in range(N):
            piece1_idx = grid[r][c]
            piece2_idx = grid[r+1][c]
            
            if piece1_idx is not None and piece2_idx is not None:
                score = evaluate_edge_similarity_direct(
                    pieces[piece1_idx], 
                    pieces[piece2_idx], 
                    'bottom', 
                    'top'
                )
                total_score += score
                total_edges += 1
    
    return total_score / max(1, total_edges)

def evaluate_corner_compatibility_direct(pieces, grid, N):
    """
    Evaluate corner compatibility using direct edge similarity
    """
    if not pieces:
        return 0
    
    score = 0
    total_corners = 0
    
    # For each corner piece
    for r in [0, N-1]:
        for c in [0, N-1]:
            total_corners += 1
            piece_idx = grid[r][c]
            if piece_idx is None or piece_idx >= len(pieces):
                continue
            
            corner_score = 0
            neighbor_count = 0
            
            # Check right neighbor
            if c < N-1:
                right_idx = grid[r][c+1]
                if right_idx is not None:
                    neighbor_count += 1
                    corner_score += evaluate_edge_similarity_direct(
                        pieces[piece_idx], 
                        pieces[right_idx], 
                        'right', 
                        'left'
                    )
            
            # Check bottom neighbor
            if r < N-1:
                bottom_idx = grid[r+1][c]
                if bottom_idx is not None:
                    neighbor_count += 1
                    corner_score += evaluate_edge_similarity_direct(
                        pieces[piece_idx], 
                        pieces[bottom_idx], 
                        'bottom', 
                        'top'
                    )
            
            if neighbor_count > 0:
                score += corner_score / neighbor_count
    
    return score / max(1, total_corners)

def choose_best_orientation_hybrid(pieces, orientations, N, all_comparisons=None):
    """
    Hybrid approach: Use direct evaluation first, descriptor if available
    """
    best_orientation = None
    best_combined_score = -1
    best_corner_score = -1
    best_grid_score = -1
    
    print(f"   🔍 Evaluating {len(orientations)} orientations (hybrid approach):")
    
    for i, orientation in enumerate(orientations):
        # Always use direct evaluation (more reliable)
        corner_score = evaluate_corner_compatibility_direct(pieces, orientation['grid'], N)
        grid_score = evaluate_grid_compatibility_direct(pieces, orientation['grid'], N)
        
        # If we have descriptor comparisons, blend with direct evaluation
        if all_comparisons:
            desc_corner_score = evaluate_corner_compatibility_descriptor(pieces, orientation['grid'], N, all_comparisons)
            desc_grid_score = evaluate_grid_compatibility(pieces, orientation['grid'], N, all_comparisons)
            
            # Blend: 70% direct, 30% descriptor
            corner_score = 0.7 * corner_score + 0.3 * desc_corner_score
            grid_score = 0.7 * grid_score + 0.3 * desc_grid_score
        
        # Combined score with more weight on grid compatibility
        combined_score = 0.3 * corner_score + 0.7 * grid_score
        
        print(f"     {i+1}. {orientation['name']}:")
        print(f"        Corner: {corner_score:.3f}, Grid: {grid_score:.3f}, Combined: {combined_score:.3f}")
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_corner_score = corner_score
            best_grid_score = grid_score
            best_orientation = orientation
    
    if best_orientation:
        print(f"   ✅ Selected: {best_orientation['name']}")
        print(f"      Corner: {best_corner_score:.3f}, Grid: {best_grid_score:.3f}, Combined: {best_combined_score:.3f}")
    
    return best_orientation, best_corner_score, best_grid_score

def visualize_piece_relationships(pieces, grid, N, title="Piece Relationships"):
    """
    Visualize how pieces are related in the grid
    """
    fig, axes = plt.subplots(N, N, figsize=(10, 10))
    
    for r in range(N):
        for c in range(N):
            ax = axes[r, c] if N > 1 else axes
            piece_idx = grid[r][c]
            
            if piece_idx is not None and piece_idx < len(pieces):
                piece = pieces[piece_idx]
                ax.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Piece {piece_idx+1}", fontsize=8)
                
                # Draw edge indicators
                if c > 0:
                    ax.plot([0, 0], [0, piece.shape[0]], 'g-', linewidth=2)  # Left edge
                if c < N-1:
                    ax.plot([piece.shape[1]-1, piece.shape[1]-1], [0, piece.shape[0]], 'g-', linewidth=2)  # Right edge
                if r > 0:
                    ax.plot([0, piece.shape[1]], [0, 0], 'r-', linewidth=2)  # Top edge
                if r < N-1:
                    ax.plot([0, piece.shape[1]], [piece.shape[0]-1, piece.shape[0]-1], 'r-', linewidth=2)  # Bottom edge
            
            ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
