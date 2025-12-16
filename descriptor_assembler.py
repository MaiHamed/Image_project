import cv2
import numpy as np
import itertools
from descriptor import extract_rectangular_edges, edge_features, normalize_features, compute_edge_distance

class DescriptorBasedAssembler:
    """
    Updated assembler using the working code's approach:
    - MSE-based edge comparison
    - Brute-force search for 2x2
    - LAB + gradient + laplacian features
    """

    def __init__(self, border_width=3, descriptor_length=100, top_k_matches=20):
        self.border_width = border_width
        self.descriptor_length = descriptor_length
        self.top_k_matches = top_k_matches

    def build_costs(self, pieces):
        """
        Build cost matrix exactly like the working code
        """
        sides = ['top', 'bottom', 'left', 'right']
        opposite = {'top': 'bottom', 'bottom': 'top', 
                   'left': 'right', 'right': 'left'}
        
        n = len(pieces)
        
        # Pre-compute features for all pieces and sides
        features = {}
        for i, piece in enumerate(pieces):
            edges = extract_rectangular_edges(piece, self.border_width)
            for side in sides:
                features[(i, side)] = edge_features(edges[side])
        
        # Initialize cost matrices
        cost = {s: np.full((n, n), np.inf) for s in sides}
        
        # Fill cost matrices
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for side in sides:
                    cost[side][i, j] = compute_edge_distance(
                        features[(i, side)],
                        features[(j, opposite[side])],
                        side, opposite[side]
                    )
        
        return cost

    def solve_2x2_bruteforce(self, pieces):
        """
        Brute-force solver for 2x2 puzzles
        """
        assert len(pieces) == 4, "2x2 solver requires exactly 4 pieces"
        
        cost = self.build_costs(pieces)
        
        best_order = None
        best_score = float('inf')
        second_best = float('inf')
        
        # Try all permutations
        for perm in itertools.permutations(range(4)):
            # Calculate total cost for this arrangement
            score = (
                cost['right'][perm[0], perm[1]] +
                cost['bottom'][perm[0], perm[2]] +
                cost['right'][perm[2], perm[3]] +
                cost['bottom'][perm[1], perm[3]]
            )
            
            if score < best_score:
                second_best = best_score
                best_score = score
                best_order = perm
            elif score < second_best:
                second_best = score
        
        # Calculate confidence margin
        margin = second_best - best_score if second_best != float('inf') else 0
        
        return list(best_order), margin

    def compute_all_comparisons(self, pieces):
        """
        Generate all pairwise comparisons for visualization
        """
        cost = self.build_costs(pieces)
        n = len(pieces)
        all_comparisons = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Convert cost to compatibility score
                compatibility = lambda c: np.exp(-c * 2) if c != np.inf else 0.0
                
                # Horizontal comparisons
                all_comparisons.append({
                    'piece1': i, 'piece2': j,
                    'edge1': 'right', 'edge2': 'left',
                    'score': compatibility(cost['right'][i, j])
                })
                
                # Vertical comparisons
                all_comparisons.append({
                    'piece1': i, 'piece2': j,
                    'edge1': 'bottom', 'edge2': 'top',
                    'score': compatibility(cost['bottom'][i, j])
                })
        
        return all_comparisons, None  # Return None for rotations to match interface

    def solve(self, all_pieces):
        """
        Main solver interface
        """
        num_pieces = len(all_pieces)
        N = int(np.sqrt(num_pieces))
        
        print(f"🤖 Solver: Grid {N}x{N}, Pieces {num_pieces}")
        
        if N == 2:
            # Use brute-force for 2x2
            best_order, margin = self.solve_2x2_bruteforce(all_pieces)
            
            # Convert to grid format
            best_grid = [
                [best_order[0], best_order[1]],
                [best_order[2], best_order[3]]
            ]
            
            # Get all comparisons for visualization
            all_comparisons, _ = self.compute_all_comparisons(all_pieces)
            all_piece_rotations = [{'0': all_pieces[i]} for i in range(num_pieces)]
            best_buddies = sorted(all_comparisons, key=lambda x: x['score'], reverse=True)[:20]
            
            # Convert margin to assembly score (higher margin = higher confidence)
            assembly_score = min(1.0, margin * 10)  # Scale margin to 0-1 range
            
            print(f"✅ 2x2 brute-force solution found")
            print(f"   Order: {best_order}")
            print(f"   Margin: {margin:.3f}")
            print(f"   Assembly score: {assembly_score:.3f}")
            
            return all_comparisons, all_piece_rotations, best_grid, best_buddies, assembly_score
        
        else:
            # For larger puzzles, fall back to original approach
            print("⚠️  Larger puzzles use heuristic approach")
            return self._solve_larger_puzzle(all_pieces, N)
    
    def _solve_larger_puzzle(self, all_pieces, N):
        """
        Fallback for puzzles larger than 2x2
        """
        # This is your original logic for larger puzzles
        all_comparisons = []
        num_pieces = len(all_pieces)
        
        # Build cost matrix
        cost = self.build_costs(all_pieces)
        
        # Generate comparisons from cost matrix
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                
                # Convert cost to compatibility
                compatibility = lambda c: np.exp(-c * 2) if c != np.inf else 0.0
                
                all_comparisons.append({
                    'piece1': i, 'piece2': j,
                    'edge1': 'right', 'edge2': 'left',
                    'score': compatibility(cost['right'][i, j])
                })
                
                all_comparisons.append({
                    'piece1': i, 'piece2': j,
                    'edge1': 'bottom', 'edge2': 'top',
                    'score': compatibility(cost['bottom'][i, j])
                })
        
        # Simple heuristic for larger puzzles (placeholder)
        all_piece_rotations = [{'0': all_pieces[i]} for i in range(num_pieces)]
        
        # Create a simple grid (just in order)
        best_grid = [[i * N + j for j in range(N)] for i in range(N)]
        
        best_buddies = sorted(all_comparisons, key=lambda x: x['score'], reverse=True)[:20]
        
        # Estimate assembly score
        assembly_score = 0.5  # Placeholder
        
        return all_comparisons, all_piece_rotations, best_grid, best_buddies, assembly_score