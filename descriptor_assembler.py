import cv2
import numpy as np
import itertools

class DescriptorBasedAssembler:
    """
    Enhanced descriptor-based assembler using color + gradient profiles.
    Improved for larger puzzles while keeping original logic.
    """

    def __init__(self, border_width=10, descriptor_length=100, top_k_matches=20):
        self.border_width = border_width
        self.descriptor_length = descriptor_length
        self.top_k_matches = top_k_matches

    # --- Edge descriptor computation ---
    def describe_edge_color_pattern(self, edge_pixels):
        if edge_pixels is None or len(edge_pixels) == 0:
            return np.full(self.descriptor_length, 0.5, dtype=np.float32)

        # Convert to gray if needed
        gray = cv2.cvtColor(edge_pixels, cv2.COLOR_BGR2GRAY) if edge_pixels.ndim == 3 else edge_pixels
        h, w = gray.shape

        # Profiles
        profile1 = gray[:, w//2].astype(np.float32) if h > w else gray[h//2, :].astype(np.float32)
        row_profile = gray[h//2, :].astype(np.float32)
        col_profile = gray[:, w//2].astype(np.float32)
        profile2 = (row_profile + col_profile[:len(row_profile)]) / 2.0 if len(row_profile) == len(col_profile) else row_profile

        # Resample to uniform length
        target_len = max(len(profile1), len(profile2))
        profile1_resampled = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(profile1)), profile1)
        profile2_resampled = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(profile2)), profile2)

        # Adaptive weighting based on variance
        std1 = np.std(profile1_resampled)
        weight1 = np.clip(0.5 + 0.5*(std1/np.max([std1, 5])), 0.35, 0.75)
        profile = weight1*profile1_resampled + (1-weight1)*profile2_resampled

        # Smooth + gradient
        profile_smooth = cv2.GaussianBlur(profile.reshape(-1,1), (5,1), 1.0).flatten()
        gradient = np.gradient(profile_smooth)
        gradient += 1e-12 * (gradient == 0)

        # Normalize and combine
        profile_norm = (profile_smooth - profile_smooth.min()) / (np.ptp(profile_smooth) + 1e-10)
        gradient_norm = (gradient - gradient.min()) / (np.ptp(gradient) + 1e-10)
        combined = 0.6*profile_norm + 0.4*gradient_norm

        # Interpolate to descriptor length
        descriptor = np.interp(np.linspace(0,1,self.descriptor_length), np.linspace(0,1,len(combined)), combined)
        min_d, max_d = descriptor.min(), descriptor.max()
        descriptor = 0.1 + 0.8*(descriptor - min_d)/(max_d - min_d + 1e-10) if max_d-min_d>1e-10 else np.full(self.descriptor_length,0.5)

        return descriptor.astype(np.float32)

    # --- Extract edges ---
    def extract_rectangular_edges(self, piece_img):
        if piece_img is None:
            return {}
        h, w = piece_img.shape[:2]
        bw = max(min(self.border_width, h//2, w//2), 5)
        return {
            'top': piece_img[:bw].copy(),
            'bottom': piece_img[-bw:].copy(),
            'left': piece_img[:, :bw].copy(),
            'right': piece_img[:, -bw:].copy()
        }

    # --- Edge similarity ---
    def compute_edge_similarity(self, desc1, desc2):
        if len(desc1)==0 or len(desc2)==0:
            return 0.0
        desc1_norm = desc1 - desc1.mean()
        desc2_norm = desc2 - desc2.mean()
        denom = np.sqrt(np.sum(desc1_norm**2) * np.sum(desc2_norm**2))
        if denom < 1e-6:
            return 0.0
        ncc = np.sum(desc1_norm*desc2_norm)/denom
        return np.clip(ncc, 0.0, 1.0)

    # --- Compute all pairwise comparisons ---
    def compute_all_comparisons(self, all_pieces):
        num_pieces = len(all_pieces)
        all_descriptors = []
        for piece in all_pieces:
            edges = self.extract_rectangular_edges(piece)
            all_descriptors.append({k: self.describe_edge_color_pattern(v) for k,v in edges.items()})

        all_comparisons = []
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j: continue
                for edge1, edge2 in [('right','left'), ('left','right'), ('bottom','top'), ('top','bottom')]:
                    score = self.compute_edge_similarity(all_descriptors[i][edge1], all_descriptors[j][edge2])
                    all_comparisons.append({'piece1':i,'piece2':j,'edge1':edge1,'edge2':edge2,'score':score})
        return all_comparisons, all_descriptors

    # --- Get top matches ---
    def get_top_matches_by_type(self, all_comparisons):
        horizontal = [c for c in all_comparisons if c['edge1']=='right' and c['edge2']=='left']
        vertical = [c for c in all_comparisons if c['edge1']=='bottom' and c['edge2']=='top']
        horizontal.sort(key=lambda x:x['score'], reverse=True)
        vertical.sort(key=lambda x:x['score'], reverse=True)
        return horizontal[:self.top_k_matches], vertical[:self.top_k_matches]

    # --- Exact brute-force solver for 2x2 ---
    def solve_2x2_bruteforce(self, all_pieces):
        assert len(all_pieces) == 4, "2x2 brute force requires exactly 4 pieces"

        # Compute descriptors once
        all_comparisons, _ = self.compute_all_comparisons(all_pieces)

        # Build fast lookup: (p1, p2, edge) → score
        edge_score = {}
        for c in all_comparisons:
            edge_score[(c['piece1'], c['piece2'], c['edge1'])] = c['score']

        best_score = -1
        best_grid = None

        # Try all permutations of 4 pieces
        for perm in itertools.permutations(range(4)):
            A, B, C, D = perm

            score = 0.0
            score += edge_score.get((A, B, 'right'), 0)
            score += edge_score.get((A, C, 'bottom'), 0)
            score += edge_score.get((B, D, 'bottom'), 0)
            score += edge_score.get((C, D, 'right'), 0)

            if score > best_score:
                best_score = score
                best_grid = [[A, B],
                             [C, D]]

        return best_grid, best_score

    # --- Build grid from top matches (fallback for larger puzzles) ---
    def build_grid_from_top_matches(self, top_horizontal, top_vertical, N):
        num_pieces = N * N
        grid = [[None] * N for _ in range(N)]
        used = set()
        grid[0][0] = 0
        used.add(0)

        for r in range(N):
            for c in range(N):
                if r == 0 and c == 0:
                    continue
                best_piece, best_score = None, -1
                for piece in range(num_pieces):
                    if piece in used:
                        continue
                    score, matches_used = 0, 0
                    if c > 0:
                        left_piece = grid[r][c - 1]
                        for m in top_horizontal:
                            if m['piece1'] == left_piece and m['piece2'] == piece:
                                score += m['score']
                                matches_used += 1
                                break
                    if r > 0:
                        top_piece = grid[r - 1][c]
                        for m in top_vertical:
                            if m['piece1'] == top_piece and m['piece2'] == piece:
                                score += m['score']
                                matches_used += 1
                                break
                    required = (1 if c > 0 else 0) + (1 if r > 0 else 0)
                    if matches_used < required:
                        continue
                    if matches_used == 2:
                        score *= 1.2
                    if score > best_score:
                        best_score = score
                        best_piece = piece
                # fallback: pick any unused piece
                if best_piece is None:
                    best_piece = [p for p in range(num_pieces) if p not in used][0]
                grid[r][c] = best_piece
                used.add(best_piece)

        return grid

    # --- Evaluate grid ---
    def evaluate_grid_quality(self, grid, all_comparisons, N):
        total_score, match_count = 0,0
        edge_lookup = {(c['piece1'], c['piece2'], c['edge1']): c['score'] for c in all_comparisons}
        for r in range(N):
            for c in range(N):
                piece = grid[r][c]
                if c < N-1:
                    right_piece = grid[r][c+1]
                    total_score += edge_lookup.get((piece, right_piece, 'right'), 0)
                    match_count +=1
                if r < N-1:
                    bottom_piece = grid[r+1][c]
                    total_score += edge_lookup.get((piece, bottom_piece, 'bottom'), 0)
                    match_count +=1
        return total_score/match_count if match_count>0 else 0.0

    # --- Main solver ---
    def solve(self, all_pieces):
        num_pieces = len(all_pieces)
        N = int(np.sqrt(num_pieces))
        print(f"🤖 Solver: Grid {N}x{N}, Pieces {num_pieces}")

        # --- Use exact brute force for 2x2 ---
        if N == 2:
            best_grid, best_score = self.solve_2x2_bruteforce(all_pieces)
            all_comparisons, _ = self.compute_all_comparisons(all_pieces)
            all_piece_rotations = [{'0': all_pieces[i]} for i in range(num_pieces)]
            best_buddies = sorted(all_comparisons, key=lambda x:x['score'], reverse=True)[:20]
            print(f"✅ Exact 2x2 brute-force score: {best_score:.3f}")
            return all_comparisons, all_piece_rotations, best_grid, best_buddies, best_score

        # --- Standard solver for larger puzzles ---
        all_comparisons, all_descriptors = self.compute_all_comparisons(all_pieces)
        print(f"   {len(all_comparisons)} edge comparisons generated")

        top_horizontal, top_vertical = self.get_top_matches_by_type(all_comparisons)

        best_grid = self.build_grid_from_top_matches(top_horizontal, top_vertical, N)
        best_score = self.evaluate_grid_quality(best_grid, all_comparisons, N)

        all_piece_rotations = [{'0': all_pieces[i]} for i in range(num_pieces)]
        best_buddies = sorted(all_comparisons, key=lambda x:x['score'], reverse=True)[:20]

        print(f"✅ Final assembly score: {best_score:.3f}")

        return all_comparisons, all_piece_rotations, best_grid, best_buddies, best_score
