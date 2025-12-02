import cv2
import numpy as np

# ------------------ FILTERING FUNCTIONS ------------------
def selective_median_filter(img, threshold=50):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    denoised = img.copy()

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            neighborhood = gray[i-1:i+2, j-1:j+2]
            median_val = np.median(neighborhood)
            current_pixel = gray[i, j]

            if abs(current_pixel - median_val) > threshold:
                if len(img.shape) == 3:
                    color_neighborhood = img[i-1:i+2, j-1:j+2]
                    denoised[i, j] = np.median(color_neighborhood, axis=(0,1))
                else:
                    denoised[i, j] = median_val

    return denoised

def canny_edges(img, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray_blur, low_threshold, high_threshold)
    return edges

def enhance_image(img, low_threshold=50, high_threshold=150):
    enhanced = img.copy()

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    lap = cv2.Laplacian(enhanced, cv2.CV_32F)
    lap -= lap.mean()
    sharpened = enhanced - 0.1 * lap
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    edges_bw = canny_edges(sharpened, low_threshold, high_threshold)
    return sharpened, edges_bw


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

#---------------- Descriptor--------------------

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
    if edge_pixels is None or len(edge_pixels) == 0:
        return np.array([])

    # grayscale
    if len(edge_pixels.shape) > 1 and edge_pixels.shape[1] == 3:
        intensities = (
            0.299 * edge_pixels[:, 0] +
            0.587 * edge_pixels[:, 1] +
            0.114 * edge_pixels[:, 2]
        )
    else:
        intensities = edge_pixels.flatten()

    if len(intensities) < 2:
        return np.array([])

    x_old = np.linspace(0, 1, len(intensities))
    x_new = np.linspace(0, 1, target_length)
    normalized = np.interp(x_new, x_old, intensities)

    mn, mx = normalized.min(), normalized.max()
    if mx > mn:
        normalized = (normalized - mn) / (mx - mn)
    else:
        normalized = np.zeros(target_length)

    return normalized

def compare_edges(desc1, desc2):
    if len(desc1) == 0 or len(desc2) == 0:
        return float('inf')

    min_len = min(len(desc1), len(desc2))
    return float(np.mean((desc1[:min_len] - desc2[:min_len]) ** 2))

def best_score_between_descriptors(desc_a, desc_b):
    if len(desc_a) == 0 or len(desc_b) == 0:
        return float('inf')
    return min(compare_edges(desc_a, desc_b),
               compare_edges(desc_a, desc_b[::-1]))

def analyze_all_possible_matches_rotation_aware(all_piece_images, piece_files, N):
    all_piece_rotations = []

    # Precompute:
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

    all_comparisons = []
    num = len(all_piece_images)

    edge_pairs = [
        ('right', 'left'),
        ('bottom', 'top'),
        ('left', 'right'),
        ('top', 'bottom')
    ]

    # Compare:
    for i in range(num):
        for j in range(num):
            if i == j:  
                continue

            piece1_desc = all_piece_rotations[i][0]['descriptors']

            for angle, rot_data in all_piece_rotations[j].items():
                piece2_desc = rot_data['descriptors']

                for e1, e2 in edge_pairs:
                    s = best_score_between_descriptors(
                        piece1_desc[e1], piece2_desc[e2]
                    )
                    all_comparisons.append({
                        'piece1': i,
                        'piece2': j,
                        'edge1': e1,
                        'edge2': e2,
                        'rotation_of_piece2': angle,
                        'score': s,
                        'label': f"P{i+1} {e1} ↔ P{j+1} {e2} (rot {angle}°)"
                    })

    all_comparisons.sort(key=lambda x: x['score'])
    return all_comparisons, all_piece_rotations
