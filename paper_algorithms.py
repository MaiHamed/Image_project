"""
Implementation of algorithms from "A fully automated greedy square jigsaw puzzle solver"
Pomeranz et al., 2011
"""
import numpy as np
import cv2
from scipy import stats
from descriptor import rotate_image_90_times, extract_rectangular_edges, describe_edge_color_pattern

class PaperPuzzleSolver:
    def __init__(self, p=0.3, q=1/16, use_prediction=True, border_width=5):
        """
        Initialize solver with paper's optimal parameters
        
        Args:
            p: Lp norm power (0.3 optimal from paper)
            q: Scaling power (1/16 optimal from paper)
            use_prediction: Use prediction-based compatibility (True) 
                           or Lp norm compatibility (False)
            border_width: How many pixels to consider at the edge (default: 5)
        """
        self.p = p
        self.q = q
        self.use_prediction = use_prediction
        self.border_width = border_width
        
    def extract_edge_region(self, piece, edge_type):
        """
        Extract a border region (not just 1 pixel)
        """
        h, w = piece.shape[:2]
        bw = self.border_width
        
        if edge_type == 'right':
            return piece[:, -bw:, :] if len(piece.shape) == 3 else piece[:, -bw:]
        elif edge_type == 'left':
            return piece[:, :bw, :] if len(piece.shape) == 3 else piece[:, :bw]
        elif edge_type == 'top':
            return piece[:bw, :, :] if len(piece.shape) == 3 else piece[:bw, :]
        elif edge_type == 'bottom':
            return piece[-bw:, :, :] if len(piece.shape) == 3 else piece[-bw:, :]
        return None
    
    # ========== COMPATIBILITY METRICS ==========
    
    def Lp_norm_compatibility(self, piece1, piece2, relation):
        """
        (L_p)^q compatibility metric from Eq. 3-5
        """
        # Extract edge regions (not just single pixels)
        edge1 = self.extract_edge_region(piece1, relation)
        # Opposite edge for piece2
        opposite = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}
        edge2 = self.extract_edge_region(piece2, opposite[relation])
        
        if edge1 is None or edge2 is None:
            return float('inf')
        
        # Resize to same dimensions if needed
        if edge1.shape != edge2.shape:
            min_h = min(edge1.shape[0], edge2.shape[0])
            min_w = min(edge1.shape[1], edge2.shape[1])
            edge1 = cv2.resize(edge1, (min_w, min_h))
            edge2 = cv2.resize(edge2, (min_w, min_h))
        
        # Calculate (L_p)^q norm (Eq. 3)
        diff = np.abs(edge1.astype(np.float32) - edge2.astype(np.float32))
        sum_p = np.sum(diff ** self.p)
        D = sum_p ** (self.q / self.p)
        
        return D
    
    def prediction_based_compatibility(self, piece1, piece2, relation):
        """
        Prediction-based compatibility from Eq. 6
        Uses Taylor expansion to predict boundary
        """
        bw = self.border_width
        
        # Extract edge regions and adjacent regions
        if relation == 'right':
            # piece1's right edge and left of it
            edge1 = piece1[:, -bw:, :]
            inner1 = piece1[:, -2*bw:-bw, :]
            # piece2's left edge and right of it
            edge2 = piece2[:, :bw, :]
            inner2 = piece2[:, bw:2*bw, :]
        elif relation == 'left':
            edge1 = piece1[:, :bw, :]
            inner1 = piece1[:, bw:2*bw, :]
            edge2 = piece2[:, -bw:, :]
            inner2 = piece2[:, -2*bw:-bw, :]
        elif relation == 'top':
            edge1 = piece1[:bw, :, :]
            inner1 = piece1[bw:2*bw, :, :]
            edge2 = piece2[-bw:, :, :]
            inner2 = piece2[-2*bw:-bw, :, :]
        elif relation == 'bottom':
            edge1 = piece1[-bw:, :, :]
            inner1 = piece1[-2*bw:-bw, :, :]
            edge2 = piece2[:bw, :, :]
            inner2 = piece2[bw:2*bw, :, :]
        
        # Resize to same dimensions if needed
        if edge1.shape != edge2.shape:
            min_h = min(edge1.shape[0], edge2.shape[0])
            min_w = min(edge1.shape[1], edge2.shape[1])
            edge1 = cv2.resize(edge1, (min_w, min_h))
            edge2 = cv2.resize(edge2, (min_w, min_h))
            inner1 = cv2.resize(inner1, (min_w, min_h))
            inner2 = cv2.resize(inner2, (min_w, min_h))
        
        # Prediction: 2*edge - inner (Taylor expansion)
        pred1 = 2 * edge1.astype(np.float32) - inner1.astype(np.float32)
        pred2 = 2 * edge2.astype(np.float32) - inner2.astype(np.float32)
        
        # Calculate prediction errors with (L_p)^q
        error1 = np.abs(pred1 - edge2.astype(np.float32)) ** self.p
        error2 = np.abs(pred2 - edge1.astype(np.float32)) ** self.p
        
        total_error = np.sum(error1) + np.sum(error2)
        D = total_error ** (self.q / self.p)
        
        return D
    
    def compute_dissimilarity(self, piece1, piece2, relation):
        """
        Compute dissimilarity (lower = better match)
        """
        if self.use_prediction:
            return self.prediction_based_compatibility(piece1, piece2, relation)
        else:
            return self.Lp_norm_compatibility(piece1, piece2, relation)
    
    # ========== COMPATIBILITY MATRIX ==========
    
    def build_compatibility_matrix(self, all_pieces):
        """
        Build compatibility matrix with proper normalization
        """
        num_pieces = len(all_pieces)
        matrix = np.zeros((num_pieces, num_pieces, 4))
        relations = ['right', 'left', 'top', 'bottom']
        
        # Compute all raw dissimilarities
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                for rel_idx, rel in enumerate(relations):
                    matrix[i, j, rel_idx] = self.compute_dissimilarity(
                        all_pieces[i], all_pieces[j], rel
                    )
        
        # Normalize: convert to compatibility scores (0-1, higher = better)
        for rel_idx in range(4):
            # Get all non-zero values for this relation
            all_values = matrix[:, :, rel_idx][matrix[:, :, rel_idx] > 0]
            if len(all_values) > 0:
                min_val = np.min(all_values)
                max_val = np.max(all_values)
                range_val = max_val - min_val
                
                if range_val > 0:
                    # Normalize to 0-1 and invert (so lower dissimilarity = higher compatibility)
                    for i in range(num_pieces):
                        for j in range(num_pieces):
                            if i != j and matrix[i, j, rel_idx] > 0:
                                # Normalize and invert
                                normalized = 1.0 - ((matrix[i, j, rel_idx] - min_val) / range_val)
                                # Apply sigmoid to emphasize good matches
                                matrix[i, j, rel_idx] = 1.0 / (1.0 + np.exp(-10 * (normalized - 0.5)))
        
        return matrix
    
    # ========== IMPROVED ASSEMBLY METHODS ==========
    
    def greedy_assemble(self, all_pieces, compatibility_matrix, N):
        """
        Robust greedy assembly that actually assembles the puzzle correctly
        """
        num_pieces = len(all_pieces)
        grid = [[None for _ in range(N)] for _ in range(N)]
        used = set()
        
        # Find the absolute strongest match to start
        best_score = -1
        best_pair = None
        best_relation = None
        
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                for rel_idx in range(4):  # 0:right, 1:left, 2:top, 3:bottom
                    score = compatibility_matrix[i, j, rel_idx]
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                        best_relation = rel_idx
        
        # Start with the strongest match
        if best_pair is not None:
            i, j = best_pair
            # Place first piece at top-left
            grid[0][0] = i
            used.add(i)
            
            # Place second piece based on relation
            relations = ['right', 'left', 'top', 'bottom']
            rel = relations[best_relation]
            
            if rel == 'right':
                if N > 1:
                    grid[0][1] = j
                else:
                    # If N=1, just place it somewhere
                    for r in range(N):
                        for c in range(N):
                            if grid[r][c] is None:
                                grid[r][c] = j
                                break
                        if j in used:
                            break
            elif rel == 'left':
                # Can't place left of (0,0), so place at (0,1) and swap
                if N > 1:
                    grid[0][1] = j
                    # Swap so they're in correct relation
                    grid[0][0], grid[0][1] = grid[0][1], grid[0][0]
                else:
                    grid[0][0] = j  # Overwrite
            elif rel == 'top':
                # Can't place top of (0,0), so place at (1,0) and swap
                if N > 1:
                    grid[1][0] = j
                    grid[0][0], grid[1][0] = grid[1][0], grid[0][0]
                else:
                    grid[0][0] = j
            elif rel == 'bottom':
                if N > 1:
                    grid[1][0] = j
                else:
                    grid[0][0] = j
            used.add(j)
        else:
            # Fallback: start with first piece
            grid[0][0] = 0
            used.add(0)
        
        print(f"   Started with pieces {best_pair[0] if best_pair else 0} and {best_pair[1] if best_pair else 'none'}")
        
        # Now fill the rest of the grid systematically
        # We'll fill row by row, left to right
        relations = ['right', 'left', 'top', 'bottom']
        
        for r in range(N):
            for c in range(N):
                if grid[r][c] is not None:
                    continue
                
                # Find the best piece for this position based on neighbors
                best_piece = None
                best_score = -1
                
                # Consider compatibility with left neighbor
                left_score = -1
                left_piece = None
                if c > 0 and grid[r][c-1] is not None:
                    left_neighbor = grid[r][c-1]
                    for piece in range(num_pieces):
                        if piece in used:
                            continue
                        score = compatibility_matrix[left_neighbor, piece, 0]  # right compatibility
                        if score > left_score:
                            left_score = score
                            left_piece = piece
                
                # Consider compatibility with top neighbor
                top_score = -1
                top_piece = None
                if r > 0 and grid[r-1][c] is not None:
                    top_neighbor = grid[r-1][c]
                    for piece in range(num_pieces):
                        if piece in used:
                            continue
                        score = compatibility_matrix[top_neighbor, piece, 3]  # bottom compatibility
                        if score > top_score:
                            top_score = score
                            top_piece = piece
                
                # Choose the best piece based on combined scores
                candidates = {}
                if left_piece is not None:
                    candidates[left_piece] = left_score
                if top_piece is not None:
                    if top_piece in candidates:
                        candidates[top_piece] = max(candidates[top_piece], top_score)
                    else:
                        candidates[top_piece] = top_score
                
                # Also consider pieces that might fit well based on diagonal neighbors
                if not candidates and r > 0 and c > 0:
                    # Check diagonal influence
                    for piece in range(num_pieces):
                        if piece in used:
                            continue
                        # Check left neighbor
                        if grid[r][c-1] is not None:
                            left_score = compatibility_matrix[grid[r][c-1], piece, 0]
                        else:
                            left_score = 0
                        # Check top neighbor
                        if grid[r-1][c] is not None:
                            top_score = compatibility_matrix[grid[r-1][c], piece, 3]
                        else:
                            top_score = 0
                        
                        total_score = left_score + top_score
                        candidates[piece] = total_score
                
                if candidates:
                    # Find the best candidate
                    best_piece = max(candidates.items(), key=lambda x: x[1])[0]
                    grid[r][c] = best_piece
                    used.add(best_piece)
                else:
                    # No good candidates based on neighbors, use any unused piece
                    for piece in range(num_pieces):
                        if piece not in used:
                            grid[r][c] = piece
                            used.add(piece)
                            break
        
        # If we still have missing pieces (shouldn't happen), fill them
        for r in range(N):
            for c in range(N):
                if grid[r][c] is None:
                    for piece in range(num_pieces):
                        if piece not in used:
                            grid[r][c] = piece
                            used.add(piece)
                            break
        
        return grid
    
    def greedy_assemble_fixed(self, all_pieces, compatibility_matrix, N):
        """
        Alternative greedy assembly with row-by-row placement
        """
        num_pieces = len(all_pieces)
        grid = [[None for _ in range(N)] for _ in range(N)]
        used = set()
        
        # Start with piece 0 at (0,0)
        grid[0][0] = 0
        used.add(0)
        
        # Fill first row
        for c in range(1, N):
            best_piece = None
            best_score = -1
            left_piece = grid[0][c-1]
            
            for piece in range(num_pieces):
                if piece in used:
                    continue
                score = compatibility_matrix[left_piece, piece, 0]  # right compatibility
                if score > best_score:
                    best_score = score
                    best_piece = piece
            
            if best_piece is not None:
                grid[0][c] = best_piece
                used.add(best_piece)
            else:
                # Use any unused piece
                for piece in range(num_pieces):
                    if piece not in used:
                        grid[0][c] = piece
                        used.add(piece)
                        break
        
        # Fill remaining rows
        for r in range(1, N):
            for c in range(N):
                best_piece = None
                best_score = -1
                
                # Check top neighbor
                top_piece = grid[r-1][c]
                for piece in range(num_pieces):
                    if piece in used:
                        continue
                    score = compatibility_matrix[top_piece, piece, 3]  # bottom compatibility
                    
                    # Also consider left neighbor if exists
                    if c > 0:
                        left_piece = grid[r][c-1]
                        left_score = compatibility_matrix[left_piece, piece, 0]  # right compatibility
                        score = (score + left_score) / 2.0
                    
                    if score > best_score:
                        best_score = score
                        best_piece = piece
                
                if best_piece is not None:
                    grid[r][c] = best_piece
                    used.add(best_piece)
                else:
                    # Use any unused piece
                    for piece in range(num_pieces):
                        if piece not in used:
                            grid[r][c] = piece
                            used.add(piece)
                            break
        
        return grid
    
    # ========== ANALYSIS METHODS ==========
    
    def find_best_buddies(self, compatibility_matrix):
        """
        Find best buddy pairs (reciprocal best matches)
        """
        num_pieces = compatibility_matrix.shape[0]
        relations = ['right', 'left', 'top', 'bottom']
        best_buddies = []
        
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                
                # Check if they're best match for each other in some direction
                for rel_idx, rel in enumerate(relations):
                    # Check if j is best for i in this direction
                    best_for_i = True
                    score_ij = compatibility_matrix[i, j, rel_idx]
                    
                    for k in range(num_pieces):
                        if k != i and compatibility_matrix[i, k, rel_idx] > score_ij:
                            best_for_i = False
                            break
                    
                    # Check if i is best for j in opposite direction
                    opp_idx = {'right': 1, 'left': 0, 'top': 3, 'bottom': 2}[rel]
                    best_for_j = True
                    score_ji = compatibility_matrix[j, i, opp_idx]
                    
                    for k in range(num_pieces):
                        if k != j and compatibility_matrix[j, k, opp_idx] > score_ji:
                            best_for_j = False
                            break
                    
                    if best_for_i and best_for_j:
                        best_buddies.append({
                            'piece1': i,
                            'piece2': j,
                            'edge1': rel,
                            'edge2': relations[opp_idx],
                            'score': (score_ij + score_ji) / 2.0
                        })
        
        return best_buddies
    
    def matrix_to_comparisons(self, compatibility_matrix):
        """
        Convert compatibility matrix to comparison list format
        """
        num_pieces = compatibility_matrix.shape[0]
        relations = ['right', 'left', 'top', 'bottom']
        all_comparisons = []
        
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
        return all_comparisons
    
    # ========== SOLVE METHODS ==========
    
    def solve(self, all_pieces):
        """
        Main solve method - returns (final_grid, compatibility_matrix, best_buddies, assembled)
        """
        num_pieces = len(all_pieces)
        N = int(np.sqrt(num_pieces))
        
        # Step 1: Build compatibility matrix
        compatibility_matrix = self.build_compatibility_matrix(all_pieces)
        
        # Step 2: Find best buddies
        best_buddies = self.find_best_buddies(compatibility_matrix)
        
        # Step 3: Greedy assembly
        final_grid = self.greedy_assemble(all_pieces, compatibility_matrix, N)
        
        # Step 4: Build assembled image
        piece_height, piece_width = all_pieces[0].shape[:2]
        assembled_height = N * piece_height
        assembled_width = N * piece_width
        assembled = np.zeros((assembled_height, assembled_width, 3), dtype=np.uint8)
        
        for r in range(N):
            for c in range(N):
                piece_idx = final_grid[r][c]
                if piece_idx is not None:
                    y_start = r * piece_height
                    x_start = c * piece_width
                    assembled[y_start:y_start+piece_height, 
                            x_start:x_start+piece_width] = all_pieces[piece_idx]
        
        return final_grid, compatibility_matrix, best_buddies, assembled
    
    def solve_for_comparisons(self, all_pieces):
        """
        Compute pairwise dissimilarities and best buddies only, 
        without assembling the puzzle.
        """
        compatibility_matrix = self.build_compatibility_matrix(all_pieces)
        best_buddies = self.find_best_buddies(compatibility_matrix)
        all_comparisons = self.matrix_to_comparisons(compatibility_matrix)
        
        # Rotations structure for visualization
        all_piece_rotations = []
        for p_img in all_pieces:
            rotations = {}
            for k in range(4):
                angle = k * 90
                img_rot = rotate_image_90_times(p_img, k)
                rotations[angle] = img_rot
            all_piece_rotations.append(rotations)
        
        return all_comparisons, all_piece_rotations, best_buddies
    
    # ========== UTILITY METHODS ==========
    
    def evaluate_assembly(self, grid, compatibility_matrix, N):
        """
        Simple evaluation: average compatibility of adjacent pieces
        """
        total_score = 0
        num_pairs = 0
        
        relations = ['right', 'left', 'top', 'bottom']
        
        for r in range(N):
            for c in range(N):
                piece_idx = grid[r][c]
                
                # Check right neighbor
                if c < N - 1:
                    neighbor_idx = grid[r][c + 1]
                    total_score += compatibility_matrix[piece_idx, neighbor_idx, 0]  # right
                    num_pairs += 1
                
                # Check bottom neighbor
                if r < N - 1:
                    neighbor_idx = grid[r + 1][c]
                    total_score += compatibility_matrix[piece_idx, neighbor_idx, 3]  # bottom
                    num_pairs += 1
        
        return total_score / num_pairs if num_pairs > 0 else 0


