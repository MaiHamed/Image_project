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

def extract_rectangular_edges(piece_img):
    #Extract all 4 edges from a rectangular puzzle piece
    if piece_img is None:
        return {}
    height, width = piece_img.shape[:2]
    return {
        'top': piece_img[0, :, :],
        'bottom': piece_img[-1, :, :],  
        'left': piece_img[:, 0, :],
        'right': piece_img[:, -1, :]
    }

def describe_edge_color_pattern(edge_pixels, target_length=100):
    #Convert edge pixels to normalized intensity pattern
    if len(edge_pixels) == 0:
        return np.array([])
    
    if len(edge_pixels.shape) > 1 and edge_pixels.shape[1] == 3:
        intensities = 0.299 * edge_pixels[:, 0] + 0.587 * edge_pixels[:, 1] + 0.114 * edge_pixels[:, 2]
    else:
        intensities = edge_pixels.flatten()
    
    if len(intensities) < 2:
        return np.array([])
        
    x_old = np.linspace(0, 1, len(intensities))
    x_new = np.linspace(0, 1, target_length)
    normalized = np.interp(x_new, x_old, intensities)
    
    if normalized.max() > normalized.min():
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    else:
        normalized = np.zeros(target_length)
    
    return normalized

def compare_edges(desc1, desc2):
    #Compare two edge descriptors (lower = better match)
    if len(desc1) == 0 or len(desc2) == 0:
        return float('inf')
    return np.mean((desc1 - desc2) ** 2)

def analyze_all_possible_matches(all_pieces_data, piece_files, N):
    #compare all pieces against all other pieces
    print(f"   🔍 COMPARISON ANALYSIS for {N}x{N} puzzle:")
    print(f"   Testing {len(all_pieces_data)} pieces against each other...")
    
    all_comparisons = []
    
    # Compare every piece with every other piece
    for i in range(len(all_pieces_data)):
        for j in range(len(all_pieces_data)):
            if i == j:  # Skip comparing piece with itself
                continue
                
            piece1_data = all_pieces_data[i]
            piece2_data = all_pieces_data[j]
            
            # Compare all edge combinations
            edge_pairs = [
                ('right', 'left', 'P{i}→P{j}'),    # Horizontal neighbors
                ('bottom', 'top', 'P{i}↓P{j}'),    # Vertical neighbors
                ('left', 'right', 'P{i}←P{j}'),    # Reverse horizontal
                ('top', 'bottom', 'P{i}↑P{j}')     # Reverse vertical
            ]
            
            for edge1, edge2, label in edge_pairs:
                if edge1 in piece1_data and edge2 in piece2_data:
                    desc1 = piece1_data[edge1]
                    desc2 = piece2_data[edge2]
                    
                    score = compare_edges(desc1, desc2)
                    
                    all_comparisons.append({
                        'piece1': i, 'piece2': j,
                        'edge1': edge1, 'edge2': edge2, 
                        'score': score,
                        'label': f"P{i+1} {edge1} ↔ P{j+1} {edge2}"
                    })
    
    # Sort by best matches
    all_comparisons.sort(key=lambda x: x['score'])
    
    # Show analysis results
    print(f"\n   📊 MATCH ANALYSIS RESULTS:")
    print(f"   Found {len(all_comparisons)} possible edge matches")
    
    # Show best matches
    print(f"\n   🏆 TOP 15 BEST MATCHES:")
    for idx, match in enumerate(all_comparisons[:15]):
        quality = "🌟" if match['score'] < 0.01 else "✅" if match['score'] < 0.05 else "⚠️"
        print(f"      {idx+1:2d}. {quality} {match['label']}: {match['score']:.4f}")
    
    # Show worst matches
    print(f"\n   🔻 TOP 15 WORST MATCHES:")
    for idx, match in enumerate(all_comparisons[-15:]):
        print(f"      {idx+1:2d}. ❌ {match['label']}: {match['score']:.4f}")
    
    return all_comparisons

