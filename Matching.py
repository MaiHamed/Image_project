"""
ENHANCED MATCHING MODULE
This module adds better matching capabilities WITHOUT changing existing code.
Just import and use alongside your current functions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_enhanced_edge_features(edge_pixels, target_length=100):
    """
    Enhanced feature extraction (adds to existing describe_edge_color_pattern)
    Returns BOTH the original intensity profile AND additional features
    """
    if edge_pixels is None or len(edge_pixels) == 0:
        return None
    
    features = {}
    
    # 1. Keep the original intensity profile (for compatibility)
    if len(edge_pixels.shape) > 1 and edge_pixels.shape[1] == 3:
        intensities = (
            0.299 * edge_pixels[:, 0] +
            0.587 * edge_pixels[:, 1] +
            0.114 * edge_pixels[:, 2]
        )
    else:
        intensities = edge_pixels.flatten()
    
    if len(intensities) < 2:
        intensities = np.zeros(target_length)
    
    # Normalize (same as original)
    x_old = np.linspace(0, 1, len(intensities))
    x_new = np.linspace(0, 1, target_length)
    intensity_profile = np.interp(x_new, x_old, intensities)
    
    if intensity_profile.max() > intensity_profile.min():
        intensity_profile = (intensity_profile - intensity_profile.min()) / (intensity_profile.max() - intensity_profile.min())
    else:
        intensity_profile = np.zeros(target_length)
    
    features['intensity'] = intensity_profile
    
    # 2. ADDITIONAL: Gradient features
    if len(edge_pixels.shape) > 2:
        # Convert to grayscale for gradient
        if edge_pixels.shape[2] == 3:
            gray = cv2.cvtColor(edge_pixels, cv2.COLOR_BGR2GRAY)
        else:
            gray = edge_pixels
    else:
        gray = edge_pixels
    
    # Calculate gradients
    if gray.ndim > 1 and gray.shape[0] > 1:
        grad = np.gradient(gray.mean(axis=1) if gray.ndim > 1 else gray)
        grad_normalized = np.interp(x_new, x_old, grad)
        if np.max(np.abs(grad_normalized)) > 0:
            grad_normalized = grad_normalized / np.max(np.abs(grad_normalized))
        features['gradient'] = grad_normalized
    else:
        features['gradient'] = np.zeros(target_length)
    
    # 3. ADDITIONAL: Color features (if color image)
    if len(edge_pixels.shape) > 2 and edge_pixels.shape[2] == 3:
        color_features = []
        for channel in range(3):
            hist = cv2.calcHist([edge_pixels], [channel], None, [8], [0, 256]).flatten()
            if hist.sum() > 0:
                hist = hist / hist.sum()
            color_features.extend(hist)
        features['color'] = np.array(color_features)
    else:
        features['color'] = np.array([])
    
    return features

def compute_enhanced_compatibility(desc1, desc2):
    """
    Enhanced compatibility calculation
    Works with BOTH original descriptors AND enhanced features
    """
    # If we have enhanced features
    if isinstance(desc1, dict) and isinstance(desc2, dict):
        scores = []
        
        # 1. Original intensity correlation (allowing reversal)
        if 'intensity' in desc1 and 'intensity' in desc2:
            corr_normal = np.correlate(desc1['intensity'], desc2['intensity'], mode='valid')
            corr_reverse = np.correlate(desc1['intensity'], desc2['intensity'][::-1], mode='valid')
            intensity_score = max(np.max(corr_normal), np.max(corr_reverse))
            scores.append(intensity_score * 0.5)
        
        # 2. Gradient similarity
        if 'gradient' in desc1 and 'gradient' in desc2:
            grad_diff = np.mean((desc1['gradient'] - desc2['gradient'])**2)
            grad_score = 1.0 / (1.0 + grad_diff)
            scores.append(grad_score * 0.3)
        
        # 3. Color similarity
        if 'color' in desc1 and 'color' in desc2 and len(desc1['color']) > 0:
            color_corr = np.corrcoef(desc1['color'], desc2['color'])[0, 1]
            color_score = (color_corr + 1) / 2  # Convert from [-1,1] to [0,1]
            scores.append(color_score * 0.2)
        
        return sum(scores) if scores else 0
    
    # Fallback to original comparison for regular descriptors
    else:
        # This is the ORIGINAL comparison logic
        if len(desc1) == 0 or len(desc2) == 0:
            return float('inf')
        
        min_len = min(len(desc1), len(desc2))
        mse1 = np.mean((desc1[:min_len] - desc2[:min_len]) ** 2)
        mse2 = np.mean((desc1[:min_len] - desc2[:min_len][::-1]) ** 2)
        
        # Convert MSE to similarity score (lower MSE = higher score)
        best_mse = min(mse1, mse2)
        similarity = 1.0 / (1.0 + best_mse)
        return similarity

def analyze_matches_with_enhanced_features(all_piece_images, piece_files, N):
    """
    Enhanced matching analysis that works alongside original analyze_all_possible_matches_rotation_aware
    """
    from functions import rotate_image_90_times, extract_rectangular_edges, describe_edge_color_pattern
    
    all_piece_rotations = []
    all_piece_features = []  # Store enhanced features
    
    # Precompute both original and enhanced features
    for p_img in all_piece_images:
        rotations = {}
        enhanced_features = {}
        
        for k in range(4):
            angle = k * 90
            img_rot = rotate_image_90_times(p_img, k)
            raw_edges = extract_rectangular_edges(img_rot)
            
            # Original descriptors (for compatibility)
            descriptors = {
                e: describe_edge_color_pattern(raw_edges[e])
                for e in raw_edges
            }
            
            # Enhanced features (for better matching)
            enhanced = {
                e: extract_enhanced_edge_features(raw_edges[e])
                for e in raw_edges
            }
            
            rotations[angle] = {
                'image': img_rot,
                'descriptors': descriptors,
                'enhanced_features': enhanced
            }
            
            if angle == 0:
                enhanced_features = enhanced
        
        all_piece_rotations.append(rotations)
        all_piece_features.append(enhanced_features)
    
    # Run comparisons with enhanced scoring
    all_comparisons = []
    num = len(all_piece_images)
    
    edge_pairs = [
        ('right', 'left'),
        ('bottom', 'top'),
        ('left', 'right'),
        ('top', 'bottom')
    ]
    
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            
            for angle, rot_data in all_piece_rotations[j].items():
                for e1, e2 in edge_pairs:
                    # Get both original and enhanced scores
                    desc1 = all_piece_rotations[i][0]['descriptors'][e1]
                    desc2 = rot_data['descriptors'][e2]
                    
                    # Original score (MSE-based, lower is better)
                    orig_score = min(
                        np.mean((desc1 - desc2) ** 2) if len(desc1) > 0 and len(desc2) > 0 else float('inf'),
                        np.mean((desc1 - desc2[::-1]) ** 2) if len(desc1) > 0 and len(desc2) > 0 else float('inf')
                    )
                    
                    # Enhanced score (similarity-based, higher is better)
                    enh1 = all_piece_rotations[i][0]['enhanced_features'][e1]
                    enh2 = rot_data['enhanced_features'][e2]
                    enhanced_score = compute_enhanced_compatibility(enh1, enh2)
                    
                    # Combine scores (you can adjust weights)
                    combined_score = enhanced_score  # Use enhanced by default
                    
                    all_comparisons.append({
                        'piece1': i,
                        'piece2': j,
                        'edge1': e1,
                        'edge2': e2,
                        'rotation_of_piece2': angle,
                        'original_score': orig_score,
                        'enhanced_score': enhanced_score,
                        'score': combined_score,  # Main score for sorting
                        'label': f"P{i+1} {e1} ↔ P{j+1} {e2} (rot {angle}°)"
                    })
    
    # Sort by combined score (higher is better for enhanced)
    all_comparisons.sort(key=lambda x: x['score'], reverse=True)
    
    return all_comparisons, all_piece_rotations, all_piece_features

def visualize_enhanced_match_analysis(piece_images, matches, top_n=10):
    """
    Enhanced visualization of matches
    """
    if not matches:
        print("No matches to visualize")
        return
    
    # Show top matches
    print(f"\n🎯 TOP {min(top_n, len(matches))} MATCHES:")
    print("-" * 60)
    for i, match in enumerate(matches[:top_n]):
        print(f"{i+1:2d}. {match['label']}")
        print(f"    Enhanced Score: {match['enhanced_score']:.4f} | Original MSE: {match['original_score']:.4f}")
        print(f"    Combined Score: {match['score']:.4f}")
        print("-" * 60)
    
    # Visualize matches
    fig, axes = plt.subplots(1, min(top_n, len(matches)), figsize=(15, 4))
    if top_n == 1:
        axes = [axes]
    
    for i, match in enumerate(matches[:top_n]):
        p1_idx = match['piece1']
        p2_idx = match['piece2']
        
        # Get images
        from functions import rotate_image_90_times
        p1_img = piece_images[p1_idx]
        p2_img = rotate_image_90_times(piece_images[p2_idx], match['rotation_of_piece2'] // 90)
        
        # Create side-by-side display
        combined = np.hstack([p1_img, p2_img])
        
        axes[i].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        
        # Add connecting line
        h, w = p1_img.shape[:2]
        if match['edge1'] == 'right' and match['edge2'] == 'left':
            axes[i].plot([w-1, w], [h//2, h//2], 'g-', linewidth=3)
        
        axes[i].set_title(f"Match {i+1}\nScore: {match['score']:.3f}")
        axes[i].axis('off')
    
    plt.suptitle(f"Top {min(top_n, len(matches))} Enhanced Matches", fontsize=14)
    plt.tight_layout()
    plt.show()
    # ========== PUZZLE ASSEMBLY FUNCTIONS ==========
def assemble_puzzle_from_matches(all_comparisons, piece_images, N):
    """
    Assemble puzzle pieces into a grid based on best matches.
    Returns assembled image grid.
    """
    if len(piece_images) != N * N:
        print(f"❌ Wrong number of pieces: got {len(piece_images)}, expected {N*N}")
        return None
    
    # Filter out self-matches and sort by score (best first)
    valid_matches = [m for m in all_comparisons if m['piece1'] != m['piece2']]
    valid_matches.sort(key=lambda x: x['score'])
    
    # Create adjacency matrix
    adjacency = {i: {'right': None, 'bottom': None, 'left': None, 'top': None} 
                 for i in range(len(piece_images))}
    
    # Fill adjacency based on best matches
    placed_edges = set()
    for match in valid_matches:
        p1, p2 = match['piece1'], match['piece2']
        edge1, edge2 = match['edge1'], match['edge2']
        
        # Only add if not already placed
        edge_key = (min(p1, p2), max(p1, p2), edge1, edge2)
        if edge_key not in placed_edges:
            # Map edges to directions
            if edge1 == 'right' and edge2 == 'left':
                if adjacency[p1]['right'] is None and adjacency[p2]['left'] is None:
                    adjacency[p1]['right'] = (p2, match['rotation_of_piece2'])
                    adjacency[p2]['left'] = (p1, (match['rotation_of_piece2'] + 180) % 360)
                    placed_edges.add(edge_key)
            elif edge1 == 'bottom' and edge2 == 'top':
                if adjacency[p1]['bottom'] is None and adjacency[p2]['top'] is None:
                    adjacency[p1]['bottom'] = (p2, match['rotation_of_piece2'])
                    adjacency[p2]['top'] = (p1, (match['rotation_of_piece2'] + 180) % 360)
                    placed_edges.add(edge_key)
    
    # DEBUG: Print adjacency
    print(f"\n🔗 Adjacency Matrix (N={N}):")
    for i in range(len(piece_images)):
        print(f"  P{i+1}: {adjacency[i]}")
    
    # Find top-left corner (piece with no left and no top connections)
    corners = []
    for i in range(len(piece_images)):
        if adjacency[i]['left'] is None and adjacency[i]['top'] is None:
            corners.append(i)
    
    if not corners:
        print("⚠️ No top-left corner found, using piece 0")
        top_left = 0
    else:
        top_left = corners[0]
    
    print(f"  Top-left corner: P{top_left+1}")
    
    # Build grid by traversing adjacency
    grid = [[None for _ in range(N)] for _ in range(N)]
    rotations = [[0 for _ in range(N)] for _ in range(N)]
    placed = set()
    
    # Place top-left piece
    grid[0][0] = top_left
    rotations[0][0] = 0
    placed.add(top_left)
    
    # Fill grid row by row
    from functions import rotate_image_90_times
    
    for row in range(N):
        for col in range(N):
            if row == 0 and col == 0:
                continue
            
            if col > 0:  # Get from left neighbor
                left_piece = grid[row][col-1]
                left_rot = rotations[row][col-1]
                
                # Find what's to the right of left_piece
                if adjacency[left_piece]['right'] is not None:
                    right_neighbor, rel_rotation = adjacency[left_piece]['right']
                    if right_neighbor not in placed:
                        grid[row][col] = right_neighbor
                        # Calculate absolute rotation
                        rotations[row][col] = (left_rot + rel_rotation) % 360
                        placed.add(right_neighbor)
                        continue
            
            if row > 0:  # Get from top neighbor
                top_piece = grid[row-1][col]
                top_rot = rotations[row-1][col]
                
                # Find what's below top_piece
                if adjacency[top_piece]['bottom'] is not None:
                    bottom_neighbor, rel_rotation = adjacency[top_piece]['bottom']
                    if bottom_neighbor not in placed:
                        grid[row][col] = bottom_neighbor
                        rotations[row][col] = (top_rot + rel_rotation) % 360
                        placed.add(bottom_neighbor)
                        continue
            
            # If no match found, use any unplaced piece
            for i in range(len(piece_images)):
                if i not in placed:
                    grid[row][col] = i
                    placed.add(i)
                    break
    
    # DEBUG: Print grid
    print(f"\n📊 Placement Grid:")
    for r in range(N):
        row_str = "  "
        for c in range(N):
            if grid[r][c] is not None:
                row_str += f"P{grid[r][c]+1}({rotations[r][c]}°) "
            else:
                row_str += "None "
        print(row_str)
    
    # Build assembled image
    piece_height, piece_width = piece_images[0].shape[:2]
    assembled_height = N * piece_height
    assembled_width = N * piece_width
    assembled = np.zeros((assembled_height, assembled_width, 3), dtype=np.uint8)
    
    for row in range(N):
        for col in range(N):
            piece_idx = grid[row][col]
            if piece_idx is not None:
                rotation = rotations[row][col]
                k = rotation // 90  # Convert degrees to 90° steps
                
                # Rotate piece
                piece_img = piece_images[piece_idx].copy()
                if k == 1:
                    piece_img = cv2.rotate(piece_img, cv2.ROTATE_90_CLOCKWISE)
                elif k == 2:
                    piece_img = cv2.rotate(piece_img, cv2.ROTATE_180)
                elif k == 3:
                    piece_img = cv2.rotate(piece_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                y_start = row * piece_height
                x_start = col * piece_width
                
                # Handle size mismatches
                h, w = piece_img.shape[:2]
                if h != piece_height or w != piece_width:
                    piece_img = cv2.resize(piece_img, (piece_width, piece_height))
                
                assembled[y_start:y_start+piece_height, x_start:x_start+piece_width] = piece_img
    
    print(f"\n✅ Assembly complete: {len(placed)}/{N*N} pieces placed")
    return assembled

def visualize_matches_with_lines(piece_images, all_comparisons, top_n=10):
    """
    Visualize best matches by drawing connecting lines between pieces.
    """
    if not all_comparisons:
        print("❌ No matches to visualize")
        return
    
    # Sort matches by score (lowest MSE first)
    sorted_matches = sorted(all_comparisons, key=lambda x: x['score'])[:top_n]
    
    # Create visualization
    fig, axes = plt.subplots(1, min(3, len(sorted_matches)), figsize=(15, 5))
    if len(sorted_matches) == 1:
        axes = [axes]
    
    for idx, match in enumerate(sorted_matches[:3]):  # Show first 3
        p1_idx = match['piece1']
        p2_idx = match['piece2']
        
        # Get images
        from functions import rotate_image_90_times
        p1_img = piece_images[p1_idx]
        p2_img = rotate_image_90_times(piece_images[p2_idx], match['rotation_of_piece2'] // 90)
        
        # Create side-by-side display
        combined = np.hstack([p1_img, p2_img])
        
        axes[idx].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        
        # Add connecting line between matching edges
        h, w = p1_img.shape[:2]
        line_color = 'lime'
        line_width = 3
        
        if match['edge1'] == 'right' and match['edge2'] == 'left':
            # Draw line from right edge of piece1 to left edge of piece2
            axes[idx].plot([w-5, w+5], [h//2, h//2], color=line_color, linewidth=line_width)
        elif match['edge1'] == 'left' and match['edge2'] == 'right':
            axes[idx].plot([5, w-5], [h//2, h//2], color=line_color, linewidth=line_width)
        elif match['edge1'] == 'bottom' and match['edge2'] == 'top':
            axes[idx].plot([w//2, w//2], [h-5, h+5], color=line_color, linewidth=line_width)
        elif match['edge1'] == 'top' and match['edge2'] == 'bottom':
            axes[idx].plot([w//2, w//2], [5, h-5], color=line_color, linewidth=line_width)
        
        axes[idx].set_title(f"Match {idx+1}: P{p1_idx+1}↔P{p2_idx+1}\nScore: {match['score']:.4f}")
        axes[idx].axis('off')
    
    plt.suptitle(f"Top {min(3, len(sorted_matches))} Best Matches", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print match details
    print(f"\n🔗 TOP {min(top_n, len(sorted_matches))} MATCHES:")
    print("-" * 60)
    for i, match in enumerate(sorted_matches[:top_n]):
        print(f"{i+1:2d}. P{match['piece1']+1} {match['edge1']} ↔ P{match['piece2']+1} {match['edge2']}")
        print(f"    Rotation: {match['rotation_of_piece2']}°, Score: {match['score']:.6f}")
        print("-" * 60)

def visualize_assembly(assembled_grid, piece_images, N, title="Assembled Puzzle"):
    """
    Display the assembled puzzle with grid lines.
    """
    if assembled_grid is None:
        print("❌ No assembled grid to display")
        return None
    
    plt.figure(figsize=(8, 8))
    
    # Draw grid lines
    h, w = assembled_grid.shape[:2]
    piece_h, piece_w = h // N, w // N
    
    # Make a copy to draw on
    display_img = assembled_grid.copy()
    
    # Draw vertical lines
    for i in range(1, N):
        x = i * piece_w
        cv2.line(display_img, (x, 0), (x, h), (0, 255, 0), 2)
    
    # Draw horizontal lines
    for i in range(1, N):
        y = i * piece_h
        cv2.line(display_img, (0, y), (w, y), (0, 255, 0), 2)
    
    # Display
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return display_img
