import cv2
import numpy as np
from typing import List, Dict, Tuple
import random

class AdvancedPuzzleSolver:
    """
    Optimized jigsaw puzzle solver using border-based features and best-buddy placement.
    """

    def __init__(
        self,
        strip_width: int = 3,
        color_weight: float = 0.4,
        gradient_mag_weight: float = 0.2,
        gradient_dir_weight: float = 0.36,
        laplacian_weight: float = 0.4,
        distance_p: float = 0.3,
        distance_q: float = 1/16,
        gaussian_ksize: int = 3,
        gaussian_sigma: float = 0.0,
        sobel_ksize: int = 3,
        laplacian_ksize: int = 1
    ):
        self.strip_width = strip_width
        self.w_color = color_weight
        self.w_grad_mag = gradient_mag_weight
        self.w_grad_dir = gradient_dir_weight
        self.w_lap = laplacian_weight
        self.p = distance_p
        self.q = distance_q
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma
        self.sobel_ksize = sobel_ksize
        self.laplacian_ksize = laplacian_ksize

    # -----------------------------
    # Feature Extraction
    # -----------------------------
    def _extract_borders(self, piece: np.ndarray) -> Dict[int, np.ndarray]:
        lab_img = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab_img.shape[:2]
        sw = min(self.strip_width, h // 2, w // 2)

        def _process(patch: np.ndarray) -> np.ndarray:
            patch_bgr = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.GaussianBlur(gray, (self.gaussian_ksize, self.gaussian_ksize), self.gaussian_sigma)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)
            grad_mag = cv2.magnitude(gx, gy)[..., None]
            grad_dir = cv2.phase(gx, gy, angleInDegrees=True)[..., None]
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=self.laplacian_ksize)[..., None]

            return np.concatenate([patch, grad_mag, grad_dir, lap], axis=2)

        return {
            0: _process(lab_img[0:sw, :, :]),
            1: _process(lab_img[:, w-sw:w, :]),
            2: _process(lab_img[h-sw:h, :, :]),
            3: _process(lab_img[:, 0:sw, :])
        }

    def _normalize(self, strip: np.ndarray) -> np.ndarray:
        arr = strip.astype(np.float32)
        for ch in range(arr.shape[2]):
            mean, std = arr[..., ch].mean(), arr[..., ch].std()
            arr[..., ch] = (arr[..., ch] - mean) / (std if std > 1e-6 else 1.0)
        return arr

    def _border_distance(self, a: np.ndarray, b: np.ndarray, side_a: int, side_b: int) -> float:
        if side_a in (1, 3):
            a = np.transpose(a, (1, 0, 2))
        if side_b in (1, 3):
            b = np.transpose(b, (1, 0, 2))

        a, b = self._normalize(a), self._normalize(b)
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))

        dc = np.sum(np.abs(a[..., :3] - b[..., :3]) ** self.p)
        dm = np.sum(np.abs(a[..., 3:4] - b[..., 3:4]) ** self.p)
        dd = np.sum(np.abs(a[..., 4:5] - b[..., 4:5]) ** self.p)
        dl = np.sum(np.abs(a[..., 5:6] - b[..., 5:6]) ** self.p)

        return (self.w_color * dc + self.w_grad_mag * dm + self.w_grad_dir * dd + self.w_lap * dl) ** (self.q / self.p)

    # -----------------------------
    # Compatibility Matrix
    # -----------------------------
    def _compute_compatibility_matrix(self, pieces: List[np.ndarray]) -> Dict[int, np.ndarray]:
        n = len(pieces)
        borders = [self._extract_borders(p) for p in pieces]
        compat = {s: np.full((n, n), 1e9, dtype=np.float32) for s in range(4)}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                compat[0][i, j] = self._border_distance(borders[i][0], borders[j][2], 0, 2)
                compat[1][i, j] = self._border_distance(borders[i][1], borders[j][3], 1, 3)
                compat[2][i, j] = self._border_distance(borders[i][2], borders[j][0], 2, 0)
                compat[3][i, j] = self._border_distance(borders[i][3], borders[j][1], 3, 1)
        return compat

    # -----------------------------
    # Best-Buddy Logic
    # -----------------------------
    @staticmethod
    def _opposite_side(side: int) -> int:
        return (side + 2) % 4

    def _is_best_buddy(self, i: int, side: int, j: int, compat: Dict[int, np.ndarray]) -> bool:
        if i == j:
            return False
        return (
            np.argmin(compat[side][i]) == j and
            np.argmin(compat[self._opposite_side(side)][j]) == i
        )

    # -----------------------------
    # Greedy Placement
    # -----------------------------
    def _greedy_place(self, grid_size: int, compat: Dict[int, np.ndarray]) -> List[int]:
        total = grid_size * grid_size
        placement = [-1] * total
        used = [False] * total

        seed_piece = random.randint(0, total - 1)
        start_pos = random.randint(0, total - 1)
        placement[start_pos] = seed_piece
        used[seed_piece] = True

        def neighbors(pos: int) -> List[Tuple[int, int]]:
            r, c = divmod(pos, grid_size)
            result = []
            if r > 0 and placement[pos-grid_size] != -1: result.append((pos-grid_size, 2))
            if r < grid_size-1 and placement[pos+grid_size] != -1: result.append((pos+grid_size, 0))
            if c > 0 and placement[pos-1] != -1: result.append((pos-1, 1))
            if c < grid_size-1 and placement[pos+1] != -1: result.append((pos+1, 3))
            return result

        while -1 in placement:
            best_candidate = None
            for pos in range(total):
                if placement[pos] != -1: continue
                neighs = neighbors(pos)
                if not neighs: continue
                for pid in range(total):
                    if used[pid]: continue
                    bb = 0
                    score = 0
                    for n_pos, side in neighs:
                        if self._is_best_buddy(placement[n_pos], side, pid, compat):
                            bb += 1
                        score += compat[side][placement[n_pos], pid]
                    cand = (bb, -score, pos, pid)
                    if best_candidate is None or cand > best_candidate:
                        best_candidate = cand
            if best_candidate is None:
                # Fill remaining randomly
                for pos in range(total):
                    if placement[pos] == -1:
                        for pid in range(total):
                            if not used[pid]:
                                placement[pos] = pid
                                used[pid] = True
                                break
                break
            _, _, pos, pid = best_candidate
            placement[pos] = pid
            used[pid] = True

        return placement

    # -----------------------------
    # Public Solve API
    # -----------------------------
    def solve(self, pieces: List[np.ndarray], grid_size: int) -> List[int]:
        if len(pieces) != grid_size ** 2:
            raise ValueError(f"Expected {grid_size**2} pieces, got {len(pieces)}")
        compat = self._compute_compatibility_matrix(pieces)
        return self._greedy_place(grid_size, compat)
