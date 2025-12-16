import cv2
import numpy as np
from typing import List, Tuple, Dict
import random

class AdvancedPuzzleSolver:
    """
    Advanced jigsaw puzzle solver using multi-feature compatibility analysis
    and best-buddy relationship matching for optimal piece placement.
    """
    
    def __init__(
        self,
        strip_width: int = 3,
        color_weight: float = 0.4,
        gradient_magnitude_weight: float = 0.2,
        gradient_direction_weight: float = 0.36,
        laplacian_weight: float = 0.4,
        distance_p: float = 0.3,
        distance_q: float = 1/16,
        gaussian_kernel_size: int = 3,
        gaussian_sigma: float = 0.0,
        sobel_kernel_size: int = 3,
        laplacian_kernel_size: int = 1
    ):
        """
        Initialize the puzzle solver with customizable parameters.
        
        Args:
            strip_width: Width of border strip to analyze
            color_weight: Weight for color difference in compatibility score
            gradient_magnitude_weight: Weight for gradient magnitude difference
            gradient_direction_weight: Weight for gradient direction difference
            laplacian_weight: Weight for Laplacian difference
            distance_p: Power for Minkowski distance
            distance_q: Final power for distance normalization
            gaussian_kernel_size: Kernel size for Gaussian blur
            gaussian_sigma: Sigma for Gaussian blur
            sobel_kernel_size: Kernel size for Sobel operator
            laplacian_kernel_size: Kernel size for Laplacian operator
        """
        self.strip_width = strip_width
        self.color_weight = color_weight
        self.gradient_magnitude_weight = gradient_magnitude_weight
        self.gradient_direction_weight = gradient_direction_weight
        self.laplacian_weight = laplacian_weight
        self.distance_p = distance_p
        self.distance_q = distance_q
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.sobel_kernel_size = sobel_kernel_size
        self.laplacian_kernel_size = laplacian_kernel_size
        
        # Internal state
        self._pieces = None
        self._compatibility_matrices = None
        self._piece_features = None
        
    class BorderExtractor:
        """Helper class for extracting and processing border features."""
        
        def __init__(self, parent_solver):
            self.solver = parent_solver
            
        def extract_border_features(self, image_piece: np.ndarray) -> Dict[int, np.ndarray]:
            """
            Extract multi-modal features from all four borders of a puzzle piece.
            
            Args:
                image_piece: Input puzzle piece image
                
            Returns:
                Dictionary mapping border direction to feature tensor
            """
            # Convert to LAB color space for better color perception
            lab_image = cv2.cvtColor(image_piece, cv2.COLOR_BGR2LAB).astype(np.float32)
            height, width = lab_image.shape[:2]
            
            # Ensure strip width is valid
            strip_size = min(self.solver.strip_width, height // 2, width // 2)
            
            def process_border_region(border_region: np.ndarray) -> np.ndarray:
                """Process a single border region to extract all features."""
                # Convert back to BGR for grayscale conversion
                border_bgr = cv2.cvtColor(border_region.astype(np.uint8), 
                                         cv2.COLOR_LAB2BGR)
                
                # Compute grayscale and apply smoothing
                grayscale = cv2.cvtColor(border_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                smoothed = cv2.GaussianBlur(
                    grayscale,
                    (self.solver.gaussian_kernel_size, self.solver.gaussian_kernel_size),
                    self.solver.gaussian_sigma
                )
                
                # Compute gradient components
                gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, 
                                      ksize=self.solver.sobel_kernel_size)
                gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, 
                                      ksize=self.solver.sobel_kernel_size)
                
                # Compute gradient magnitude and direction
                gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)[..., None]
                gradient_direction = cv2.phase(gradient_x, gradient_y, 
                                              angleInDegrees=True)[..., None]
                
                # Compute Laplacian for edge detection
                laplacian = cv2.Laplacian(smoothed, cv2.CV_32F, 
                                         ksize=self.solver.laplacian_kernel_size)[..., None]
                
                # Concatenate all features: LAB color + gradient + Laplacian
                feature_tensor = np.concatenate([
                    border_region,  # LAB channels (3)
                    gradient_magnitude,  # (1)
                    gradient_direction,  # (1)
                    laplacian  # (1)
                ], axis=2)
                
                return feature_tensor
            
            # Extract features from all four borders
            border_features = {
                0: process_border_region(lab_image[0:strip_size, :, :]),  # Top
                1: process_border_region(lab_image[:, width - strip_size:width, :]),  # Right
                2: process_border_region(lab_image[height - strip_size:height, :, :]),  # Bottom
                3: process_border_region(lab_image[:, 0:strip_size, :])  # Left
            }
            
            return border_features
        
    def _normalize_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """
        Normalize feature tensor using z-score normalization per channel.
        
        Args:
            feature_tensor: Input feature tensor
            
        Returns:
            Normalized feature tensor
        """
        normalized = feature_tensor.astype(np.float32)
        
        for channel_idx in range(normalized.shape[2]):
            channel_data = normalized[..., channel_idx]
            channel_mean = channel_data.mean()
            channel_std = channel_data.std()
            
            # Avoid division by zero
            if channel_std > 1e-6:
                normalized[..., channel_idx] = (channel_data - channel_mean) / channel_std
            else:
                normalized[..., channel_idx] = channel_data - channel_mean
                
        return normalized
    
    def _compute_border_compatibility(
        self, 
        border_a: np.ndarray, 
        border_b: np.ndarray,
        side_a: int, 
        side_b: int
    ) -> float:
        """
        Compute compatibility score between two borders.
        
        Args:
            border_a: First border features
            border_b: Second border features
            side_a: Side index of first border (0:top, 1:right, 2:bottom, 3:left)
            side_b: Side index of second border
            
        Returns:
            Compatibility score (lower is better)
        """
        # Transpose if borders are vertical for consistent comparison
        if side_a in (1, 3):  # Right or left side
            border_a = np.transpose(border_a, (1, 0, 2))
        if side_b in (1, 3):  # Right or left side
            border_b = np.transpose(border_b, (1, 0, 2))
        
        # Normalize both feature tensors
        border_a_norm = self._normalize_features(border_a)
        border_b_norm = self._normalize_features(border_b)
        
        # Resize if dimensions don't match
        if border_a_norm.shape[:2] != border_b_norm.shape[:2]:
            border_b_norm = cv2.resize(border_b_norm, 
                                      (border_a_norm.shape[1], border_a_norm.shape[0]))
        
        # Compute feature distances with individual weights
        color_distance = np.sum(np.abs(border_a_norm[..., :3] - border_b_norm[..., :3]) ** self.distance_p)
        gradient_mag_distance = np.sum(np.abs(border_a_norm[..., 3:4] - border_b_norm[..., 3:4]) ** self.distance_p)
        gradient_dir_distance = np.sum(np.abs(border_a_norm[..., 4:5] - border_b_norm[..., 4:5]) ** self.distance_p)
        laplacian_distance = np.sum(np.abs(border_a_norm[..., 5:6] - border_b_norm[..., 5:6]) ** self.distance_p)
        
        # Weighted combination of all distances
        weighted_distance = (
            self.color_weight * color_distance +
            self.gradient_magnitude_weight * gradient_mag_distance +
            self.gradient_direction_weight * gradient_dir_distance +
            self.laplacian_weight * laplacian_distance
        )
        
        # Final normalization
        return weighted_distance ** (self.distance_q / self.distance_p)
    
    def _build_compatibility_matrices(self, pieces: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Build compatibility matrices for all piece pairs and all sides.
        
        Args:
            pieces: List of puzzle piece images
            
        Returns:
            Dictionary of compatibility matrices for each side
        """
        num_pieces = len(pieces)
        extractor = self.BorderExtractor(self)
        
        # Extract features for all pieces
        piece_border_features = [extractor.extract_border_features(piece) for piece in pieces]
        
        # Initialize compatibility matrices
        compatibility_dict = {
            side: np.full((num_pieces, num_pieces), 1e9, dtype=np.float32)
            for side in range(4)
        }
        
        # Fill compatibility matrices
        for i in range(num_pieces):
            for j in range(num_pieces):
                if i == j:
                    continue
                
                # Compute compatibility for all four matching directions
                compatibility_dict[0][i, j] = self._compute_border_compatibility(
                    piece_border_features[i][0], piece_border_features[j][2], 0, 2
                )  # Top of i matches bottom of j
                
                compatibility_dict[1][i, j] = self._compute_border_compatibility(
                    piece_border_features[i][1], piece_border_features[j][3], 1, 3
                )  # Right of i matches left of j
                
                compatibility_dict[2][i, j] = self._compute_border_compatibility(
                    piece_border_features[i][2], piece_border_features[j][0], 2, 0
                )  # Bottom of i matches top of j
                
                compatibility_dict[3][i, j] = self._compute_border_compatibility(
                    piece_border_features[i][3], piece_border_features[j][1], 3, 1
                )  # Left of i matches right of j
        
        return compatibility_dict
    
    @staticmethod
    def _get_opposite_side(side: int) -> int:
        """Get opposite side index."""
        return (side + 2) % 4
    
    def _is_best_buddy_pair(
        self, 
        piece_a_idx: int, 
        side_a: int, 
        piece_b_idx: int,
        compatibility_matrices: Dict[int, np.ndarray]
    ) -> bool:
        """
        Check if two pieces are best buddies on the specified sides.
        
        Args:
            piece_a_idx: Index of first piece
            side_a: Side of first piece
            piece_b_idx: Index of second piece
            compatibility_matrices: Compatibility matrices
            
        Returns:
            True if pieces are best buddies
        """
        if piece_a_idx == piece_b_idx:
            return False
        
        side_b = self._get_opposite_side(side_a)
        
        # Check mutual best compatibility
        is_a_best_for_b = np.argmin(compatibility_matrices[side_a][piece_a_idx]) == piece_b_idx
        is_b_best_for_a = np.argmin(compatibility_matrices[side_b][piece_b_idx]) == piece_a_idx
        
        return is_a_best_for_b and is_b_best_for_a
    
    def _greedy_placement(
        self, 
        grid_size: int, 
        compatibility_matrices: Dict[int, np.ndarray]
    ) -> List[int]:
        """
        Perform greedy placement of pieces using best-buddy relationships.
        
        Args:
            grid_size: Size of the puzzle grid (grid_size x grid_size)
            compatibility_matrices: Compatibility matrices
            
        Returns:
            List of piece indices in their placed positions
        """
        total_pieces = grid_size * grid_size
        position_to_piece = [-1] * total_pieces
        used_pieces = [False] * total_pieces
        
        # Random initialization
        random_seed_piece = random.randint(0, total_pieces - 1)
        random_start_position = random.randint(0, total_pieces - 1)
        
        position_to_piece[random_start_position] = random_seed_piece
        used_pieces[random_seed_piece] = True
        
        def get_neighbor_info(position: int) -> List[Tuple[int, int]]:
            """Get all filled neighbors of a position with their connecting sides."""
            row, col = divmod(position, grid_size)
            neighbors = []
            
            # Check all four directions
            if row > 0 and position_to_piece[position - grid_size] != -1:
                neighbors.append((position - grid_size, 2))  # Neighbor below connects to top
            if row < grid_size - 1 and position_to_piece[position + grid_size] != -1:
                neighbors.append((position + grid_size, 0))  # Neighbor above connects to bottom
            if col > 0 and position_to_piece[position - 1] != -1:
                neighbors.append((position - 1, 1))  # Neighbor right connects to left
            if col < grid_size - 1 and position_to_piece[position + 1] != -1:
                neighbors.append((position + 1, 3))  # Neighbor left connects to right
                
            return neighbors
        
        # Greedy placement loop
        while -1 in position_to_piece:
            best_candidate = None
            
            # Evaluate all empty positions
            for position in range(total_pieces):
                if position_to_piece[position] != -1:
                    continue
                    
                neighbor_info = get_neighbor_info(position)
                if not neighbor_info:
                    continue
                
                # Evaluate all unused pieces for this position
                for candidate_piece in range(total_pieces):
                    if used_pieces[candidate_piece]:
                        continue
                    
                    # Count best-buddy relationships and compute total compatibility
                    best_buddy_count = 0
                    total_compatibility = 0.0
                    
                    for neighbor_position, connection_side in neighbor_info:
                        neighbor_piece = position_to_piece[neighbor_position]
                        
                        if self._is_best_buddy_pair(
                            neighbor_piece, connection_side, 
                            candidate_piece, compatibility_matrices
                        ):
                            best_buddy_count += 1
                            
                        total_compatibility += compatibility_matrices[connection_side][
                            neighbor_piece, candidate_piece
                        ]
                    
                    # Scoring: prioritize best buddies, then compatibility
                    candidate_score = (
                        best_buddy_count,
                        -total_compatibility,  # Negative because lower compatibility is better
                        position,
                        candidate_piece
                    )
                    
                    if best_candidate is None or candidate_score > best_candidate:
                        best_candidate = candidate_score
            
            if best_candidate is None:
                # No valid placement found, fill remaining randomly
                for position in range(total_pieces):
                    if position_to_piece[position] == -1:
                        for piece in range(total_pieces):
                            if not used_pieces[piece]:
                                position_to_piece[position] = piece
                                used_pieces[piece] = True
                                break
                break
            
            _, _, best_position, best_piece = best_candidate
            position_to_piece[best_position] = best_piece
            used_pieces[best_piece] = True
        
        return position_to_piece
    
    def solve(self, puzzle_pieces: List[np.ndarray], grid_size: int) -> List[int]:
        """
        Solve the jigsaw puzzle by finding optimal piece arrangement.
        
        Args:
            puzzle_pieces: List of puzzle piece images as numpy arrays
            grid_size: Size of the square puzzle grid
            
        Returns:
            List of piece indices representing the solved arrangement
            
        Example:
            >>> solver = AdvancedPuzzleSolver()
            >>> pieces = [piece1, piece2, piece3, ...]  # List of 9 images for 3x3 puzzle
            >>> arrangement = solver.solve(pieces, 3)
            >>> print(arrangement)  # [5, 2, 7, 0, 1, 6, 3, 8, 4]
        """
        # Validate input
        if not puzzle_pieces:
            raise ValueError("Puzzle pieces list cannot be empty")
        
        expected_pieces = grid_size * grid_size
        if len(puzzle_pieces) != expected_pieces:
            raise ValueError(
                f"Expected {expected_pieces} pieces for {grid_size}x{grid_size} grid, "
                f"but got {len(puzzle_pieces)}"
            )
        
        # Compute compatibility matrices
        self._compatibility_matrices = self._build_compatibility_matrices(puzzle_pieces)
        
        # Perform greedy placement
        solution = self._greedy_placement(grid_size, self._compatibility_matrices)
        
        return solution
