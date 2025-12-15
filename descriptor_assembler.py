import cv2
import numpy as np
from descriptor import describe_edge_color_pattern, extract_rectangular_edges, compute_edge_compatibility


class DescriptorBasedAssembler:
    """
    Improved descriptor-based assembler using top matches for assembly
    """
    # you are not the issue
    def __init__(self, border_width=10, descriptor_length=100):
        self.border_width = border_width
        self.descriptor_length = descriptor_length
    
    def describe_edge_color_pattern(self, edge_pixels):
        """Effective edge descriptor using gradient information"""
        if edge_pixels is None or len(edge_pixels) == 0:
            return np.zeros(self.descriptor_length)
        
        # Convert to grayscale for simplicity
        if len(edge_pixels.shape) == 3:  # Color image
            gray = cv2.cvtColor(edge_pixels, cv2.COLOR_BGR2GRAY)
        else:
            gray = edge_pixels
        
        # Get edge profile
        if len(gray.shape) == 2:
            # Take middle row/column for edge profile
            if gray.shape[0] > gray.shape[1]:
                # Vertical edge (left/right)
                profile = gray[:, gray.shape[1]//2].astype(np.float32)
            else:
                # Horizontal edge (top/bottom)
                profile = gray[gray.shape[0]//2, :].astype(np.float32)
        else:
            profile = gray.flatten().astype(np.float32)
        
        if len(profile) < 5:
            return np.zeros(self.descriptor_length)
        
        # Apply smoothing
        profile_smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (3, 1), 1.0).flatten()
        
        # Calculate gradient (edge changes)
        if len(profile_smooth) > 1:
            gradient = np.gradient(profile_smooth)
        else:
            gradient = np.zeros_like(profile_smooth)
        
        # Combine profile and gradient
        profile_norm = (profile_smooth - profile_smooth.min()) / (profile_smooth.max() - profile_smooth.min() + 1e-10)
        gradient_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-10)
        combined = 0.6 * profile_norm + 0.4 * gradient_norm
        
        # Resize to target length
        if len(combined) > 1:
            x_old = np.linspace(0, 1, len(combined))
            x_new = np.linspace(0, 1, self.descriptor_length)
            descriptor = np.interp(x_new, x_old, combined)
        else:
            descriptor = np.zeros(self.descriptor_length)
        
        # Normalize to [0.1, 0.9] range
        descriptor_min, descriptor_max = descriptor.min(), descriptor.max()
        if descriptor_max > descriptor_min:
            descriptor = 0.1 + 0.8 * (descriptor - descriptor_min) / (descriptor_max - descriptor_min)
        else:
            descriptor = np.full_like(descriptor, 0.5)
        
        return descriptor


    def extract_rectangular_edges(self, piece_img):
        """Extract border regions with better edge focus"""
        if piece_img is None:
            return {}
        
        h, w = piece_img.shape[:2]
        bw = min(self.border_width, h // 4, w // 4)
        bw = max(bw, 3)
        
        if len(piece_img.shape) == 3:
            return {
                'top': piece_img[:bw, :, :].copy(),
                'bottom': piece_img[-bw:, :, :].copy(),
                'left': piece_img[:, :bw, :].copy(),
                'right': piece_img[:, -bw:, :].copy()
            }
        else:
            return {
                'top': piece_img[:bw, :].copy(),
                'bottom': piece_img[-bw:, :].copy(),
                'left': piece_img[:, :bw].copy(),
                'right': piece_img[:, -bw:].copy()
            }


    def compute_edge_similarity(self, desc1, desc2):
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0

        desc1_norm = desc1 - desc1.mean()
        desc2_norm = desc2 - desc2.mean()
        denom = np.sqrt(np.sum(desc1_norm**2) * np.sum(desc2_norm**2)) + 1e-10
        ncc = np.sum(desc1_norm * desc2_norm) / denom
        score = max(0.0, min(1.0, ncc))
        return score
    


    
    def compute_all_comparisons(self, all_pieces):
        """
        Compute all pairwise edge comparisons with proper edge pairing
        """
        num_pieces = len(all_pieces)
        
        # Extract descriptors for all pieces
        all_descriptors = []
        for piece_idx, piece in enumerate(all_pieces):
            edges = self.extract_rectangular_edges(piece)
            descriptors = {}
            for edge_name, edge_pixels in edges.items():
                descriptors[edge_name] = self.describe_edge_color_pattern(edge_pixels)
            all_descriptors.append(descriptors)
        
        # Compute all comparisons for COMPATIBLE edges only
        all_comparisons = []
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                
                # Only check compatible edge pairs
                compatible_pairs = [
                    ('right', 'left'),   # Piece i right edge to Piece j left edge
                    ('left', 'right'),   # Piece i left edge to Piece j right edge
                    ('bottom', 'top'),   # Piece i bottom edge to Piece j top edge
                    ('top', 'bottom')    # Piece i top edge to Piece j bottom edge
                ]
                
                for edge1, edge2 in compatible_pairs:
                    desc1 = all_descriptors[i][edge1]
                    desc2 = all_descriptors[j][edge2]
                    
                    score = self.compute_edge_similarity(desc1, desc2)
                    
                    all_comparisons.append({
                        'piece1': i,
                        'piece2': j,
                        'edge1': edge1,
                        'edge2': edge2,
                        'score': score
                    })
        
        return all_comparisons, all_descriptors
    
    def get_top_matches_by_type(self, all_comparisons, num_pieces):
        """
        Get top matches for each edge type
        """
        # Separate matches by type
        horizontal_matches = []  # right-left matches
        vertical_matches = []    # bottom-top matches
        
        for comp in all_comparisons:
            if comp['edge1'] == 'right' and comp['edge2'] == 'left':
                horizontal_matches.append(comp)
            elif comp['edge1'] == 'bottom' and comp['edge2'] == 'top':
                vertical_matches.append(comp)
        
        # Sort by score
        horizontal_matches.sort(key=lambda x: x['score'], reverse=True)
        vertical_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top matches
        top_horizontal = horizontal_matches[:min(10, len(horizontal_matches))]
        top_vertical = vertical_matches[:min(10, len(vertical_matches))]
        
        return top_horizontal, top_vertical

    def evaluate_grid_quality(self, grid, all_comparisons, N):
        """
        Evaluate how well the grid matches the top comparisons
        """
        total_score = 0
        match_count = 0
        
        for r in range(N):
            for c in range(N):
                piece = grid[r][c]
                
                # Check right neighbor
                if c < N - 1:
                    right_piece = grid[r][c+1]
                    # Look for right-left match
                    for comp in all_comparisons:
                        if (comp['piece1'] == piece and comp['piece2'] == right_piece and 
                            comp['edge1'] == 'right' and comp['edge2'] == 'left'):
                            total_score += comp['score']
                            match_count += 1
                            break
                
                # Check bottom neighbor
                if r < N - 1:
                    bottom_piece = grid[r+1][c]
                    # Look for bottom-top match
                    for comp in all_comparisons:
                        if (comp['piece1'] == piece and comp['piece2'] == bottom_piece and 
                            comp['edge1'] == 'bottom' and comp['edge2'] == 'top'):
                            total_score += comp['score']
                            match_count += 1
                            break
        
        if match_count > 0:
            return total_score / match_count
        return 0.0

 #. you are not the issue
    def build_grid_from_top_matches(self, top_horizontal, top_vertical, N):
        """
        Build grid using top matches with both neighbor checks
        """
        num_pieces = N * N
        grid = [[None for _ in range(N)] for _ in range(N)]
        used = set()

        print(f"\n   Using {len(top_horizontal)} top horizontal matches")
        print(f"   Using {len(top_vertical)} top vertical matches")

        # Start with piece 0 at position (0,0)
        start_piece = 0
        grid[0][0] = start_piece
        used.add(start_piece)

        for r in range(N):
            for c in range(N):
                if r == 0 and c == 0:
                    continue  # Already placed
                best_piece = None
                best_score = -1

                for piece in range(num_pieces):
                    if piece in used:
                        continue

                    score = 0
                    # Check left neighbor
                    if c > 0:
                        left_piece = grid[r][c-1]
                        for match in top_horizontal:
                            if match['piece1'] == left_piece and match['piece2'] == piece:
                                score += match['score']
                                break

                    # Check top neighbor
                    if r > 0:
                        top_piece = grid[r-1][c]
                        for match in top_vertical:
                            if match['piece1'] == top_piece and match['piece2'] == piece:
                                score += match['score']
                                break

                    if score > best_score:
                        best_score = score
                        best_piece = piece

                # Place the best scoring piece
                grid[r][c] = best_piece
                used.add(best_piece)
                print(f"   Placed P{best_piece+1} at ({r},{c}) with score {best_score:.3f}")

        return grid

 # you are not the issue
    def try_different_starting_pieces(self, top_horizontal, top_vertical, all_comparisons, N, num_pieces):
        """
        Try different starting pieces to find best grid with both neighbor checks
        """
        best_grid = None
        best_score = 0

        for start_piece in range(min(4, num_pieces)):  # Try first 4 pieces as starting point
            print(f"\n   Trying starting piece: P{start_piece+1}")

            grid = [[None for _ in range(N)] for _ in range(N)]
            used = set()
            grid[0][0] = start_piece
            used.add(start_piece)

            for r in range(N):
                for c in range(N):
                    if r == 0 and c == 0:
                        continue  # Already placed
                    best_piece = None
                    best_score_local = -1

                    for piece in range(num_pieces):
                        if piece in used:
                            continue

                        score = 0
                        # Check left neighbor
                        if c > 0:
                            left_piece = grid[r][c-1]
                            for match in top_horizontal:
                                if match['piece1'] == left_piece and match['piece2'] == piece:
                                    score += match['score']
                                    break

                        # Check top neighbor
                        if r > 0:
                            top_piece = grid[r-1][c]
                            for match in top_vertical:
                                if match['piece1'] == top_piece and match['piece2'] == piece:
                                    score += match['score']
                                    break

                        if score > best_score_local:
                            best_score_local = score
                            best_piece = piece

                    # Place the best scoring piece
                    grid[r][c] = best_piece
                    used.add(best_piece)

            # Evaluate this grid
            score = self.evaluate_grid_quality(grid, all_comparisons, N)
            print(f"   Starting with P{start_piece+1}: Score = {score:.3f}")

            if score > best_score:
                best_score = score
                best_grid = [row[:] for row in grid]  # Deep copy

        return best_grid, best_score

    # you are not the issue
    def solve(self, all_pieces):
        """
        Main solver using top matches for assembly
        """
        num_pieces = len(all_pieces)
        N = int(np.sqrt(num_pieces))
        
        print(f"\n🤖 IMPROVED DESCRIPTOR-BASED SOLVER")
        print(f"   Grid: {N}x{N}, Pieces: {num_pieces}")
        print(f"   Strategy: Using top matches for assembly")
        
        # Step 1: Compute all edge comparisons
        print("\n1️⃣ Computing edge descriptors and comparisons...")
        all_comparisons, all_descriptors = self.compute_all_comparisons(all_pieces)
        print(f"   Generated {len(all_comparisons)} edge comparisons")
        
        # Show top matches
        if all_comparisons:
            sorted_comparisons = sorted(all_comparisons, key=lambda x: x['score'], reverse=True)
            print(f"\n   🏆 TOP 10 OVERALL MATCHES:")
            for idx in range(min(10, len(sorted_comparisons))):
                match = sorted_comparisons[idx]
                print(f"   {idx+1:2d}. P{match['piece1']+1} {match['edge1']:6} ↔ "
                      f"P{match['piece2']+1} {match['edge2']:6} (score: {match['score']:.4f})")
        
        # Step 2: Get top matches by type
        print("\n2️⃣ Filtering mutual best-buddy matches...")
        mutual_matches = mutual_best_buddies(all_comparisons)
        filtered = mutual_matches + sorted(all_comparisons, key=lambda x: x['score'], reverse=True)[:50]


        print(f"   Mutual matches kept: {len(mutual_matches)} / {len(all_comparisons)}")

        # Use ONLY mutual matches from now on
        top_horizontal, top_vertical = self.get_top_matches_by_type(filtered, num_pieces)
        
        print(f"\n   🔗 TOP HORIZONTAL MATCHES (right-left):")
        for idx, match in enumerate(top_horizontal[:5]):
            print(f"   {idx+1:2d}. P{match['piece1']+1} right → P{match['piece2']+1} left (score: {match['score']:.4f})")
        
        print(f"\n   🔗 TOP VERTICAL MATCHES (bottom-top):")
        for idx, match in enumerate(top_vertical[:5]):
            print(f"   {idx+1:2d}. P{match['piece1']+1} bottom → P{match['piece2']+1} top (score: {match['score']:.4f})")
        
        # Step 3: Try different assembly strategies
        print("\n3️⃣ Assembling puzzle using top matches...")
        
        # Strategy 1: Try different starting pieces
        print("\n   🔍 Strategy 1: Trying different starting pieces...")
        best_grid, best_score = self.try_different_starting_pieces(top_horizontal, top_vertical, all_comparisons, N, num_pieces)
        
        # Strategy 2: If strategy 1 fails, use simple greedy
        if best_grid is None or best_score < 0.3:
            print("\n   🔍 Strategy 2: Using greedy assembly...")
            best_grid = self.build_grid_from_top_matches(top_horizontal, top_vertical, N)
            best_score = self.evaluate_grid_quality(best_grid, all_comparisons, N)
        
        # Step 4: Evaluate final assembly
        print("\n4️⃣ Evaluating final assembly...")
        print(f"   ✅ Final assembly score: {best_score:.3f}")
        
        if best_score > 0.3:
            print(f"   🎯 Good assembly (score > 0.3)")
        elif best_score > 0.1:
            print(f"   ⚠️  Fair assembly (score > 0.1)")
        else:
            print(f"   ❌ Poor assembly")
        
        # Create piece rotations list for compatibility
        all_piece_rotations = [{'0': all_pieces[i]} for i in range(num_pieces)]
        
        # Get best buddies (top matches)
        best_buddies = sorted(all_comparisons, key=lambda x: x['score'], reverse=True)[:20]
        
        return all_comparisons, all_piece_rotations, best_grid, best_buddies, best_score


# you are not the issue
def mutual_best_buddies(matches):
    best_for = {}

    # Best outgoing match per (piece, edge)
    for m in matches:
        key = (m['piece1'], m['edge1'])
        if key not in best_for or m['score'] > best_for[key]['score']:
            best_for[key] = m

    mutual = []
    seen = set()

    for m in best_for.values():
        rev_key = (m['piece2'], m['edge2'])
        if rev_key in best_for:
            rev = best_for[rev_key]
            if rev['piece2'] == m['piece1'] and rev['edge2'] == m['edge1']:
                pair_id = tuple(sorted([
                    (m['piece1'], m['edge1']),
                    (m['piece2'], m['edge2'])
                ]))
                if pair_id not in seen:
                    mutual.append(m)
                    seen.add(pair_id)

    return mutual

