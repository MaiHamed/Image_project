"""
Implementation of algorithms from "A fully automated greedy square jigsaw puzzle solver"
Pomeranz et al., 2011
"""
import numpy as np
import cv2
from scipy import stats

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
    
    # ========== SIMPLIFIED SOLVER ==========
    
    def build_compatibility_matrix(self, all_pieces):
        """
        Build compatibility matrix with proper normalization
        """
        num_pieces = len(all_pieces)
        matrix = np.zeros((num_pieces, num_pieces, 4))
        relations = ['right', 'left', 'top', 'bottom']
        
        # Compute all raw dissimilarities
        print("   Computing raw dissimilarities...")
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                for rel_idx, rel in enumerate(relations):
                    matrix[i, j, rel_idx] = self.compute_dissimilarity(
                        all_pieces[i], all_pieces[j], rel
                    )
        
        # Normalize: convert to compatibility scores (0-1, higher = better)
        print("   Normalizing scores...")
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
                                # Apply exponential to emphasize good matches
                                matrix[i, j, rel_idx] = np.exp(2 * (normalized - 1))
        
        return matrix
    
    def greedy_assemble(self, all_pieces, compatibility_matrix, N):
        """
        Simple greedy assembly
        """
        num_pieces = len(all_pieces)
        grid = [[None for _ in range(N)] for _ in range(N)]
        used = set()
        
        # Start with piece 0 at center
        center_r, center_c = N // 2, N // 2
        grid[center_r][center_c] = 0
        used.add(0)
        
        # Map relation to direction
        rel_to_dir = {'right': (0, 1), 'left': (0, -1), 
                     'top': (-1, 0), 'bottom': (1, 0)}
        rel_to_idx = {'right': 0, 'left': 1, 'top': 2, 'bottom': 3}
        
        while len(used) < num_pieces:
            best_score = -1
            best_position = None
            best_piece = None
            
            # Find empty positions adjacent to placed pieces
            for r in range(N):
                for c in range(N):
                    if grid[r][c] is not None:
                        continue
                    
                    # Check all 4 directions for neighbors
                    neighbors = []
                    for rel, (dr, dc) in rel_to_dir.items():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] is not None:
                            neighbor_idx = grid[nr][nc]
                            rel_idx = rel_to_idx[rel]
                            neighbors.append((neighbor_idx, rel_idx))
                    
                    if neighbors:
                        # Find best piece for this position
                        for piece_idx in range(num_pieces):
                            if piece_idx in used:
                                continue
                            
                            # Average compatibility with all neighbors
                            total_score = 0
                            for neighbor_idx, rel_idx in neighbors:
                                total_score += compatibility_matrix[neighbor_idx, piece_idx, rel_idx]
                            avg_score = total_score / len(neighbors)
                            
                            if avg_score > best_score:
                                best_score = avg_score
                                best_position = (r, c)
                                best_piece = piece_idx
            
            # Place the best piece found
            if best_piece is not None:
                r, c = best_position
                grid[r][c] = best_piece
                used.add(best_piece)
            else:
                # Place any unused piece in first available spot
                for r in range(N):
                    for c in range(N):
                        if grid[r][c] is None:
                            for piece_idx in range(num_pieces):
                                if piece_idx not in used:
                                    grid[r][c] = piece_idx
                                    used.add(piece_idx)
                                    break
                            break
                    if len(used) == num_pieces:
                        break
        
        return grid
    
    def solve(self, all_pieces):
        """
        Simplified but effective solver
        """
        num_pieces = len(all_pieces)
        N = int(np.sqrt(num_pieces))
        
        print(f"🧩 Solving {N}x{N} puzzle with {num_pieces} pieces")
        print(f"📊 Using {'prediction-based' if self.use_prediction else 'Lp norm'} compatibility")
        
        # Step 1: Build compatibility matrix
        print("1️⃣ Building compatibility matrix...")
        compatibility_matrix = self.build_compatibility_matrix(all_pieces)
        
        # Step 2: Find best buddies (simplified)
        print("2️⃣ Finding best matches...")
        best_buddies = []
        
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                
                # Check if they're best match for each other in some direction
                for rel_idx, rel in enumerate(['right', 'left', 'top', 'bottom']):
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
                            'relation': rel,
                            'score': (score_ij + score_ji) / 2
                        })
        
        print(f"   Found {len(best_buddies)} reciprocal best matches")
        
        # Step 3: Greedy assembly
        print("3️⃣ Assembling puzzle...")
        final_grid = self.greedy_assemble(all_pieces, compatibility_matrix, N)  # FIXED: use all_pieces
        
        # Step 4: Build assembled image
        piece_height, piece_width = all_pieces[0].shape[:2]  # FIXED: use all_pieces
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
                            x_start:x_start+piece_width] = all_pieces[piece_idx]  # FIXED: use all_pieces
        
        # Apply orientation correction
        print("4️⃣ Checking orientation...")
        assembled = self.correct_orientation(assembled, all_pieces)  # FIXED: use all_pieces
        
        # Step 5: Evaluate
        print("5️⃣ Evaluating assembly...")
        score = self.evaluate_assembly(final_grid, compatibility_matrix, N)
        print(f"✅ Assembly score: {score:.3f}")
        
        return final_grid, compatibility_matrix, best_buddies, assembled      
    def evaluate_assembly(self, grid, compatibility_matrix, N):
        """
        Simple evaluation: average compatibility of adjacent pieces
        """
        total_score = 0
        num_pairs = 0
        
        rel_to_idx = {'right': 0, 'left': 1, 'top': 2, 'bottom': 3}
        
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
    
    def correct_orientation(self, assembled_img, all_piece_images):
        """
        Detect if assembled image needs rotation correction
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2GRAY)
        
        # Check for text-like features (text tends to be in upper half)
        h, w = gray.shape
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        # Text regions typically have higher variance
        top_variance = np.var(top_half)
        bottom_variance = np.var(bottom_half)
        
        # If bottom has higher variance, image might be upside down
        if bottom_variance > top_variance * 1.5:
            print("   ↻ Detected possible upside-down orientation, rotating 180°")
            assembled_img = cv2.rotate(assembled_img, cv2.ROTATE_180)
        
        # Check left/right orientation using edge detection
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_edges = cv2.Canny(left_half, 50, 150)
        right_edges = cv2.Canny(right_half, 50, 150)
        
        left_edge_density = np.sum(left_edges > 0) / left_edges.size
        right_edge_density = np.sum(right_edges > 0) / right_edges.size
        
        # If left side has significantly more edges, might need horizontal flip
        if left_edge_density > right_edge_density * 1.3:
            print("   ↔ Detected possible mirrored orientation, flipping horizontally")
            assembled_img = cv2.flip(assembled_img, 1)
        
        return assembled_img
    # Add this method to the PaperPuzzleSolver class:
def correct_orientation(self, assembled_img, all_piece_images):
    """
    Detect if assembled image needs rotation correction
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2GRAY)
    
    # Check for text-like features (text tends to be in upper half)
    h, w = gray.shape
    top_half = gray[:h//2, :]
    bottom_half = gray[h//2:, :]
    
    # Text regions typically have higher variance
    top_variance = np.var(top_half)
    bottom_variance = np.var(bottom_half)
    
    # If bottom has higher variance, image might be upside down
    if bottom_variance > top_variance * 1.5:
        print("   ↻ Detected possible upside-down orientation, rotating 180°")
        assembled_img = cv2.rotate(assembled_img, cv2.ROTATE_180)
    
    # Check left/right orientation using edge detection
    left_half = gray[:, :w//2]
    right_half = gray[:, w//2:]
    
    left_edges = cv2.Canny(left_half, 50, 150)
    right_edges = cv2.Canny(right_half, 50, 150)
    
    left_edge_density = np.sum(left_edges > 0) / left_edges.size
    right_edge_density = np.sum(right_edges > 0) / right_edges.size
    
    # If left side has significantly more edges, might need horizontal flip
    if left_edge_density > right_edge_density * 1.3:
        print("   ↔ Detected possible mirrored orientation, flipping horizontally")
        assembled_img = cv2.flip(assembled_img, 1)
    
    return assembled_img