import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import random


class AdvancedPuzzleSolver:

    def __init__(
        self,
        border_analysis_width: int = 3,
        color_importance: float = 0.4,
        gradient_strength_importance: float = 0.2,
        gradient_orientation_importance: float = 0.36,
        edge_detection_importance: float = 0.4,
        distance_exponent: float = 0.3,
        normalization_exponent: float = 1/16,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.0,
        edge_kernel_size: int = 3,
        laplacian_kernel_size: int = 1
    ):

        self.border_width = border_analysis_width
        self.color_weight = color_importance
        self.gradient_magnitude_weight = gradient_strength_importance
        self.gradient_direction_weight = gradient_orientation_importance
        self.laplacian_weight = edge_detection_importance
        self.distance_power = distance_exponent
        self.normalization_power = normalization_exponent
        self.gaussian_size = smoothing_kernel_size
        self.gaussian_sigma = smoothing_sigma
        self.sobel_size = edge_kernel_size
        self.laplacian_size = laplacian_kernel_size
        
        # Internal state management
        self._puzzle_pieces = None
        self._compatibility_data = None
        self._feature_cache = None
    
    class BorderAnalyzer:
        
        def __init__(self, solver_instance):
            self.solver = solver_instance
        
        def extract_piece_features(self, piece_image: np.ndarray) -> Dict[int, np.ndarray]:
            # Convert to perceptually uniform LAB color space
            lab_converted = cv2.cvtColor(piece_image, cv2.COLOR_BGR2LAB).astype(np.float32)
            height, width = lab_converted.shape[:2]
            
            # Adjust strip width to fit piece dimensions
            strip_dimension = min(
                self.solver.border_width,
                height // 2,
                width // 2
            )
            
            def analyze_border_region(region: np.ndarray) -> np.ndarray:
                """Process a border region to extract all relevant features."""
                # Convert back to BGR for grayscale processing
                region_bgr = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_LAB2BGR)
                
                # Convert to grayscale and apply smoothing
                gray_scale = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                smoothed = cv2.GaussianBlur(
                    gray_scale,
                    (self.solver.gaussian_size, self.solver.gaussian_size),
                    self.solver.gaussian_sigma
                )
                
                # Calculate gradient information
                sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=self.solver.sobel_size)
                sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=self.solver.sobel_size)
                
                # Compute gradient magnitude and orientation
                gradient_strength = cv2.magnitude(sobel_x, sobel_y)[..., None]
                gradient_angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)[..., None]
                
                # Compute edge detection response
                edge_response = cv2.Laplacian(
                    smoothed,
                    cv2.CV_32F,
                    ksize=self.solver.laplacian_size
                )[..., None]
                
                # Combine all features into single tensor
                combined_features = np.concatenate([
                    region,              # LAB color channels
                    gradient_strength,   # Gradient magnitude
                    gradient_angle,      # Gradient direction
                    edge_response        # Laplacian edge response
                ], axis=2)
                
                return combined_features
            
            # Define border regions for feature extraction
            border_regions = {
                0: lab_converted[:strip_dimension, :, :],           # Top border
                1: lab_converted[:, -strip_dimension:, :],         # Right border
                2: lab_converted[-strip_dimension:, :, :],         # Bottom border
                3: lab_converted[:, :strip_dimension, :]           # Left border
            }
            
            # Process each border region
            return {
                side: analyze_border_region(region)
                for side, region in border_regions.items()
            }
    
    def _standardize_features(self, feature_data: np.ndarray) -> np.ndarray:

        standardized = feature_data.astype(np.float32)
        
        for channel in range(standardized.shape[2]):
            channel_values = standardized[..., channel]
            channel_mean = np.mean(channel_values)
            channel_std = np.std(channel_values)
            
            if channel_std > 1e-8:
                standardized[..., channel] = (channel_values - channel_mean) / channel_std
            else:
                standardized[..., channel] = channel_values - channel_mean
        
        return standardized
    
    def _calculate_border_similarity(
        self,
        border_features_a: np.ndarray,
        border_features_b: np.ndarray,
        side_index_a: int,
        side_index_b: int
    ) -> float:
        """
        Compute similarity score between two border feature sets.
        
        Parameters:
            border_features_a: Features from first border
            border_features_b: Features from second border
            side_index_a: Orientation index of first border
            side_index_b: Orientation index of second border
            
        Returns:
            Compatibility score (lower values indicate better match)
        """
        # Reorient vertical borders for consistent comparison
        if side_index_a in (1, 3):  # Vertical borders
            border_features_a = np.transpose(border_features_a, (1, 0, 2))
        if side_index_b in (1, 3):  # Vertical borders
            border_features_b = np.transpose(border_features_b, (1, 0, 2))
        
        # Standardize feature representations
        standardized_a = self._standardize_features(border_features_a)
        standardized_b = self._standardize_features(border_features_b)
        
        # Ensure consistent dimensions
        if standardized_a.shape[:2] != standardized_b.shape[:2]:
            standardized_b = cv2.resize(
                standardized_b,
                (standardized_a.shape[1], standardized_a.shape[0])
            )
        
        # Compute weighted distance components
        color_difference = np.sum(
            np.abs(standardized_a[..., :3] - standardized_b[..., :3]) ** self.distance_power
        )
        
        gradient_strength_difference = np.sum(
            np.abs(standardized_a[..., 3:4] - standardized_b[..., 3:4]) ** self.distance_power
        )
        
        gradient_direction_difference = np.sum(
            np.abs(standardized_a[..., 4:5] - standardized_b[..., 4:5]) ** self.distance_power
        )
        
        edge_response_difference = np.sum(
            np.abs(standardized_a[..., 5:6] - standardized_b[..., 5:6]) ** self.distance_power
        )
        
        # Combine weighted distances
        combined_distance = (
            self.color_weight * color_difference +
            self.gradient_magnitude_weight * gradient_strength_difference +
            self.gradient_direction_weight * gradient_direction_difference +
            self.laplacian_weight * edge_response_difference
        )
        
        # Apply final normalization
        return combined_distance ** (self.normalization_power / self.distance_power)
    
    def _generate_compatibility_data(self, pieces: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Generate comprehensive compatibility assessments for all piece pairs.
        
        Parameters:
            pieces: Collection of puzzle piece images
            
        Returns:
            Dictionary containing compatibility matrices for each orientation
        """
        piece_count = len(pieces)
        feature_extractor = self.BorderAnalyzer(self)
        
        # Extract border features for all pieces
        piece_features = [
            feature_extractor.extract_piece_features(piece)
            for piece in pieces
        ]
        
        # Initialize compatibility storage
        compatibility_storage = {
            orientation: np.full((piece_count, piece_count), float('inf'), dtype=np.float32)
            for orientation in range(4)
        }
        
        # Calculate compatibility for all piece combinations
        for idx_a in range(piece_count):
            for idx_b in range(piece_count):
                if idx_a == idx_b:
                    continue
                
                # Top of A matches bottom of B
                compatibility_storage[0][idx_a, idx_b] = self._calculate_border_similarity(
                    piece_features[idx_a][0],
                    piece_features[idx_b][2],
                    0, 2
                )
                
                # Right of A matches left of B
                compatibility_storage[1][idx_a, idx_b] = self._calculate_border_similarity(
                    piece_features[idx_a][1],
                    piece_features[idx_b][3],
                    1, 3
                )
                
                # Bottom of A matches top of B
                compatibility_storage[2][idx_a, idx_b] = self._calculate_border_similarity(
                    piece_features[idx_a][2],
                    piece_features[idx_b][0],
                    2, 0
                )
                
                # Left of A matches right of B
                compatibility_storage[3][idx_a, idx_b] = self._calculate_border_similarity(
                    piece_features[idx_a][3],
                    piece_features[idx_b][1],
                    3, 1
                )
        
        return compatibility_storage
    
    @staticmethod
    def _determine_opposite_side(side_index: int) -> int:
        """Calculate opposite side index."""
        return (side_index + 2) % 4
    
    def _check_reciprocal_best_match(
        self,
        piece_a_index: int,
        side_a: int,
        piece_b_index: int,
        compatibility_data: Dict[int, np.ndarray]
    ) -> bool:
        """
        Verify if two pieces are mutually optimal matches on specified sides.
        
        Parameters:
            piece_a_index: Index of first piece
            side_a: Side index on first piece
            piece_b_index: Index of second piece
            compatibility_data: Compatibility assessment data
            
        Returns:
            Boolean indicating reciprocal best match status
        """
        if piece_a_index == piece_b_index:
            return False
        
        opposite_side = self._determine_opposite_side(side_a)
        
        # Check mutual optimality
        a_optimal_for_b = np.argmin(compatibility_data[side_a][piece_a_index]) == piece_b_index
        b_optimal_for_a = np.argmin(compatibility_data[opposite_side][piece_b_index]) == piece_a_index
        
        return a_optimal_for_b and b_optimal_for_a
    
    def _assemble_pieces_greedily(
        self,
        puzzle_dimension: int,
        compatibility_data: Dict[int, np.ndarray]
    ) -> List[int]:
        """
        Construct puzzle arrangement using greedy optimization with
        reciprocal match prioritization.
        
        Parameters:
            puzzle_dimension: Grid dimension (N x N)
            compatibility_data: Compatibility assessment data
            
        Returns:
            Ordered list of piece indices in final arrangement
        """
        total_pieces = puzzle_dimension * puzzle_dimension
        position_assignment = [-1] * total_pieces
        piece_usage = [False] * total_pieces
        
        # Initialize with random placement
        initial_piece = random.randint(0, total_pieces - 1)
        initial_position = random.randint(0, total_pieces - 1)
        
        position_assignment[initial_position] = initial_piece
        piece_usage[initial_piece] = True
        
        def identify_adjacent_placements(position: int) -> List[Tuple[int, int]]:
            """Identify occupied neighboring positions and their connection sides."""
            row, column = divmod(position, puzzle_dimension)
            adjacent_placements = []
            
            # Check all four potential neighbors
            if row > 0 and position_assignment[position - puzzle_dimension] != -1:
                adjacent_placements.append((position - puzzle_dimension, 2))
            if row < puzzle_dimension - 1 and position_assignment[position + puzzle_dimension] != -1:
                adjacent_placements.append((position + puzzle_dimension, 0))
            if column > 0 and position_assignment[position - 1] != -1:
                adjacent_placements.append((position - 1, 1))
            if column < puzzle_dimension - 1 and position_assignment[position + 1] != -1:
                adjacent_placements.append((position + 1, 3))
            
            return adjacent_placements
        
        # Iterative greedy placement
        while -1 in position_assignment:
            optimal_placement = None
            
            # Evaluate all empty positions
            for current_position in range(total_pieces):
                if position_assignment[current_position] != -1:
                    continue
                
                adjacent_info = identify_adjacent_placements(current_position)
                if not adjacent_info:
                    continue
                
                # Assess all available pieces for current position
                for candidate_index in range(total_pieces):
                    if piece_usage[candidate_index]:
                        continue
                    
                    # Calculate reciprocal matches and total compatibility
                    reciprocal_match_count = 0
                    cumulative_compatibility = 0.0
                    
                    for neighbor_position, connection_side in adjacent_info:
                        neighbor_index = position_assignment[neighbor_position]
                        
                        if self._check_reciprocal_best_match(
                            neighbor_index,
                            connection_side,
                            candidate_index,
                            compatibility_data
                        ):
                            reciprocal_match_count += 1
                        
                        cumulative_compatibility += compatibility_data[connection_side][
                            neighbor_index, candidate_index
                        ]
                    
                    # Scoring: prioritize reciprocal matches, then compatibility
                    placement_score = (
                        reciprocal_match_count,
                        -cumulative_compatibility,  # Negate for proper ordering
                        current_position,
                        candidate_index
                    )
                    
                    if optimal_placement is None or placement_score > optimal_placement:
                        optimal_placement = placement_score
            
            if optimal_placement is None:
                # Fill remaining positions randomly
                for position in range(total_pieces):
                    if position_assignment[position] == -1:
                        for piece_index in range(total_pieces):
                            if not piece_usage[piece_index]:
                                position_assignment[position] = piece_index
                                piece_usage[piece_index] = True
                                break
                break
            
            _, _, selected_position, selected_piece = optimal_placement
            position_assignment[selected_position] = selected_piece
            piece_usage[selected_piece] = True
        
        return position_assignment
    
    def solve_puzzle(
        self,
        puzzle_pieces: List[np.ndarray],
        grid_dimension: int
    ) -> List[int]:
        """
        Determine optimal arrangement for jigsaw puzzle pieces.
        
        Parameters:
            puzzle_pieces: List of puzzle piece images (numpy arrays)
            grid_dimension: Dimension of square puzzle grid
            
        Returns:
            List of piece indices representing solved arrangement
            
        Example Usage:
            >>> solver = AdvancedPuzzleSolver()
            >>> pieces = [piece1, piece2, ..., piece9]  # 9 pieces for 3x3 puzzle
            >>> arrangement = solver.solve_puzzle(pieces, 3)
            >>> print(arrangement)  # Example: [2, 5, 0, 7, 1, 3, 6, 8, 4]
        """
        # Validate input parameters
        if not puzzle_pieces:
            raise ValueError("Puzzle pieces collection must not be empty")
        
        expected_count = grid_dimension * grid_dimension
        actual_count = len(puzzle_pieces)
        
        if actual_count != expected_count:
            raise ValueError(
                f"Grid dimension {grid_dimension}x{grid_dimension} requires "
                f"{expected_count} pieces, but received {actual_count}"
            )
        
        # Generate compatibility assessments
        self._compatibility_data = self._generate_compatibility_data(puzzle_pieces)
        
        # Determine optimal arrangement
        solution = self._assemble_pieces_greedily(grid_dimension, self._compatibility_data)
        
        return solution