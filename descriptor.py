import cv2
import numpy as np

BORDER_WIDTH = 3  # Match the working code

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

def extract_rectangular_edges(piece_img, border_width=BORDER_WIDTH):
    """
    Extract border region matching the working code
    """
    if piece_img is None:
        return {}

    h, w = piece_img.shape[:2]
    bw = min(border_width, h//3, w//3)
    
    if len(piece_img.shape) == 3:  # Color image
        return {
            'top': piece_img[:bw, :, :].copy(),
            'bottom': piece_img[-bw:, :, :].copy(),
            'left': piece_img[:, :bw, :].copy(),
            'right': piece_img[:, -bw:, :].copy()
        }
    else:  # Grayscale image
        return {
            'top': piece_img[:bw, :].copy(),
            'bottom': piece_img[-bw:, :].copy(),
            'left': piece_img[:, :bw].copy(),
            'right': piece_img[:, -bw:].copy()
        }

def edge_features(edge):
    """
    Extract features exactly like the working code:
    - Gaussian blur
    - LAB color space
    - Gradient magnitude
    - Laplacian
    """
    # Apply Gaussian blur
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    
    # Convert to LAB color space
    if edge.ndim == 3 and edge.shape[2] == 3:  # Color image
        lab = cv2.cvtColor(edge, cv2.COLOR_BGR2LAB).astype(np.float32)
    else:  # Grayscale
        lab = np.stack([edge.astype(np.float32)] * 3, axis=2)
    
    # Compute gradients
    gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)[..., None]  # Gradient magnitude
    lap = cv2.Laplacian(gray, cv2.CV_32F)[..., None]  # Laplacian
    
    # Combine features: LAB + gradient magnitude + laplacian
    features = np.concatenate([lab, mag, lap], axis=2)
    return features

def normalize_features(f):
    """
    Normalize features channel-wise (mean=0, std=1)
    """
    f = f.copy()
    for c in range(f.shape[2]):
        m = f[..., c].mean()
        s = f[..., c].std()
        f[..., c] = (f[..., c] - m) / (s if s > 1e-6 else 1.0)
    return f

def compute_edge_distance(a, b, side_a, side_b):
    """
    Compute edge compatibility using MSE approach
    Returns lower score = better match (like the working code)
    """
    # Normalize features
    a_norm = normalize_features(a)
    b_norm = normalize_features(b)
    
    # Transpose vertical edges for proper comparison
    if side_a in ('left', 'right'):
        a_norm = np.transpose(a_norm, (1, 0, 2))
    if side_b in ('left', 'right'):
        b_norm = np.transpose(b_norm, (1, 0, 2))
    
    # Resize if necessary
    if a_norm.shape != b_norm.shape:
        b_norm = cv2.resize(b_norm, (a_norm.shape[1], a_norm.shape[0]))
    
    # Compute absolute differences with weights
    d = np.abs(a_norm - b_norm)
    
    # Weighted average: LAB color (0.5) + gradient (0.3) + laplacian (0.2)
    distance = (
        0.5 * d[..., 0:3].mean() +    # LAB channels
        0.3 * d[..., 3].mean() +      # Gradient magnitude
        0.2 * d[..., 4].mean()        # Laplacian
    )
    
    return distance

def describe_edge_color_pattern(edge_pixels, target_length=100, border_width=BORDER_WIDTH):
    """
    Keep interface but use new approach internally
    For compatibility with existing code
    """
    features = edge_features(edge_pixels)
    
    # Convert to 1D descriptor by averaging along the edge
    if features.ndim == 3:
        # Average across the border dimension
        if features.shape[0] == border_width:  # Top/bottom
            desc = features.mean(axis=0).flatten()
        else:  # Left/right
            desc = features.mean(axis=1).flatten()
    else:
        desc = features.flatten()
    
    # Ensure descriptor length
    if len(desc) > target_length:
        # Downsample
        desc = np.interp(
            np.linspace(0, 1, target_length),
            np.linspace(0, 1, len(desc)),
            desc
        )
    elif len(desc) < target_length:
        # Pad
        desc = np.pad(desc, (0, target_length - len(desc)), mode='edge')
    
    return desc

def compute_edge_compatibility(desc1, desc2):
    """
    Convert distance to compatibility score (0-1)
    Lower distance = higher compatibility
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return 0.0
    
    # Convert from MSE distance to compatibility score
    # distance of 0 = perfect match (score 1.0)
    # distance > 2 = poor match (score ~0.0)
    distance = np.mean(np.abs(desc1 - desc2))
    compatibility = np.exp(-distance * 2)
    return np.clip(compatibility, 0.0, 0.99)