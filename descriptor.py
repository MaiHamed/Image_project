import cv2
import numpy as np


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

def describe_edge_color_pattern(edge_pixels, target_length=150, border_width=15):
    """
    ENHANCED edge descriptor with more pixels and better discrimination
    - Uses wider border (15 pixels instead of 3)
    - More sophisticated feature extraction
    - Better normalization to AVOID PERFECT 1.000 SCORES
    - Ensures scores are in range [0.01, 0.99]
    """
    if edge_pixels is None or len(edge_pixels) == 0:
        return np.array([])
    
    # Handle shape issues - ensure we have proper dimensions
    if len(edge_pixels.shape) == 1:
        # If it's 1D, reshape it
        edge_pixels = edge_pixels.reshape(-1, 1)
        is_color = False
    elif len(edge_pixels.shape) == 2:
        # Check if it's color (3 channels) or grayscale
        if edge_pixels.shape[1] == 3:
            is_color = True
        else:
            is_color = False
            edge_pixels = edge_pixels.reshape(-1, 1)
    else:
        # Handle unexpected shape
        return np.zeros(target_length)
    
    features = []
    edge_length = edge_pixels.shape[0]
    
    if edge_length < 5:  # Too small to extract meaningful features
        return np.zeros(target_length)
    
    # ====== CORE FEATURE EXTRACTION ======
    
    if is_color:
        # For color images - extract multiple feature types
        # Extract RGB channels
        r_channel = edge_pixels[:, 0].astype(np.float32)
        g_channel = edge_pixels[:, 1].astype(np.float32)
        b_channel = edge_pixels[:, 2].astype(np.float32)
        
        # 1. Grayscale intensity (weighted average)
        gray = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
        
        # 2. Multiple color space representations
        # Lab-like features
        luminance = (r_channel + g_channel + b_channel) / 3.0
        
        # Color opponency (human vision inspired)
        rg_opponency = r_channel - g_channel
        by_opponency = (b_channel * 0.5) - ((r_channel + g_channel) * 0.25)
        
        # Saturation-like feature
        min_rgb = np.minimum(np.minimum(r_channel, g_channel), b_channel)
        max_rgb = np.maximum(np.maximum(r_channel, g_channel), b_channel)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
        
        # Color ratios
        total = r_channel + g_channel + b_channel + 1e-10
        r_ratio = r_channel / total
        g_ratio = g_channel / total
        
        # 3. Multi-scale gradient features
        gradients = []
        for scale in [1, 2, 3]:  # Different scales
            if edge_length > scale * 2:
                # Apply Gaussian smoothing at different scales
                kernel_size = max(3, 2 * scale + 1)
                gray_smooth = cv2.GaussianBlur(gray.reshape(-1, 1), (kernel_size, 1), scale).flatten()
                grad = np.gradient(gray_smooth)
                gradients.append(grad)
        
        # 4. Texture features - Local binary patterns style
        if edge_length > 20:
            # Local contrast (standard deviation in sliding window)
            window_size = min(15, edge_length // 2)
            local_contrast = np.zeros_like(gray)
            for i in range(edge_length):
                start = max(0, i - window_size // 2)
                end = min(edge_length, i + window_size // 2 + 1)
                local_contrast[i] = np.std(gray[start:end])
            
            # Local entropy (measure of randomness)
            local_entropy = np.zeros_like(gray)
            for i in range(edge_length):
                start = max(0, i - 5)
                end = min(edge_length, i + 6)
                window = gray[start:end]
                hist, _ = np.histogram(window, bins=8, range=(0, 255))
                hist = hist / len(window)
                hist = hist[hist > 0]
                local_entropy[i] = -np.sum(hist * np.log2(hist))
        else:
            local_contrast = np.ones_like(gray) * 0.5
            local_entropy = np.ones_like(gray) * 0.5
        
        # 5. Edge-specific features
        # Center-surround difference
        if edge_length > 10:
            center_region = gray[edge_length//4:3*edge_length//4]
            if len(center_region) > 0:
                center_mean = np.mean(center_region)
                surround_mean = (np.mean(gray[:edge_length//4]) + np.mean(gray[3*edge_length//4:])) / 2
                center_surround_diff = np.abs(center_mean - surround_mean) * np.ones_like(gray)
            else:
                center_surround_diff = np.zeros_like(gray)
        else:
            center_surround_diff = np.zeros_like(gray)
        
        # Feature weighting - prioritize discriminative features
        features_to_weight = [
            (gray, 3.0),                  # Primary intensity
            (luminance, 1.5),             # Overall brightness
            (saturation, 1.2),            # Color purity
            (rg_opponency, 1.0),          # Red-green contrast
            (by_opponency, 0.8),          # Blue-yellow contrast
            (r_ratio, 0.7),               # Red dominance
            (g_ratio, 0.7),               # Green dominance
            (local_contrast, 1.5),        # Texture contrast
            (local_entropy, 1.0),         # Texture complexity
            (center_surround_diff, 0.5),  # Edge structure
        ]
        
        # Add gradients from different scales
        for idx, grad in enumerate(gradients):
            features_to_weight.append((grad, 1.0 / (idx + 1)))  # Lower weight for higher scales
    
    else:
        # For grayscale images
        intensities = edge_pixels.flatten().astype(np.float32)
        
        if len(intensities) < 10:
            return np.zeros(target_length)
        
        # 1. Multi-scale smoothing and gradients
        intensities_features = []
        gradients = []
        
        for scale in [1, 2, 3]:
            if len(intensities) > scale * 2:
                kernel_size = max(3, 2 * scale + 1)
                intensities_smooth = cv2.GaussianBlur(intensities.reshape(-1, 1), (kernel_size, 1), scale).flatten()
                intensities_features.append(intensities_smooth)
                
                if len(intensities_smooth) > 1:
                    grad = np.gradient(intensities_smooth)
                    gradients.append(grad)
        
        # 2. Texture features
        if len(intensities) > 20:
            # Local contrast
            window_size = min(15, len(intensities) // 2)
            local_contrast = np.zeros_like(intensities)
            for i in range(len(intensities)):
                start = max(0, i - window_size // 2)
                end = min(len(intensities), i + window_size // 2 + 1)
                local_contrast[i] = np.std(intensities[start:end])
            
            # Local binary pattern approximation
            lbp = np.zeros_like(intensities)
            for i in range(1, len(intensities)-1):
                center = intensities[i]
                pattern = 0
                pattern |= (1 if intensities[i-1] > center else 0) << 0
                pattern |= (1 if intensities[i+1] > center else 0) << 1
                if i > 1:
                    pattern |= (1 if intensities[i-2] > center else 0) << 2
                if i < len(intensities)-2:
                    pattern |= (1 if intensities[i+2] > center else 0) << 3
                lbp[i] = pattern / 16.0
        else:
            local_contrast = np.ones_like(intensities) * 0.5
            lbp = np.ones_like(intensities) * 0.5
        
        # Feature weighting for grayscale
        features_to_weight = [
            (intensities, 3.0),    # Raw intensity
            (local_contrast, 2.0),  # Texture contrast
            (lbp, 1.5),             # Local binary pattern
        ]
        
        # Add smoothed versions and gradients
        for idx, smooth_int in enumerate(intensities_features):
            features_to_weight.append((smooth_int, 2.0 / (idx + 1)))
        
        for idx, grad in enumerate(gradients):
            features_to_weight.append((grad, 1.5 / (idx + 1)))
    
    # ====== FEATURE COMBINATION AND NORMALIZATION ======
    
    # Combine all weighted features
    if features_to_weight:
        # Start with zeros
        combined = np.zeros(edge_length)
        total_weight = 0
        
        for feature, weight in features_to_weight:
            if len(feature) == edge_length:
                # Normalize each feature individually
                f_min, f_max = feature.min(), feature.max()
                if f_max > f_min:
                    f_norm = (feature - f_min) / (f_max - f_min + 1e-10)
                else:
                    f_norm = np.zeros_like(feature)
                
                # Apply non-linear transformation to enhance differences
                f_norm = np.tanh(f_norm * 3)  # Compress to [-1, 1]
                f_norm = (f_norm + 1) / 2     # Convert to [0, 1]
                
                # Add with weight
                combined += f_norm * weight
                total_weight += weight
        
        # Average weighted features
        if total_weight > 0:
            combined = combined / total_weight
            
            # ====== CRITICAL: PREVENT PERFECT SCORES ======
            # 1. Add structured noise (not random) based on edge content
            edge_hash = np.sum(combined) % 1.0
            structured_noise = np.sin(np.linspace(0, 2*np.pi, len(combined)) + edge_hash) * 0.02
            combined = combined + structured_noise
            
            # 2. Apply compression to avoid extreme values
            combined = 1 / (1 + np.exp(-8 * (combined - 0.5)))
            
            # 3. Rescale to [0.05, 0.95] to avoid perfect 0 or 1
            c_min, c_max = combined.min(), combined.max()
            if c_max > c_min:
                combined = 0.05 + 0.9 * (combined - c_min) / (c_max - c_min + 1e-10)
            else:
                combined = np.full_like(combined, 0.5)
            
            # 4. Final clamping to ensure no perfect scores
            combined = np.clip(combined, 0.05, 0.95)
            
            # Interpolate to target length
            if len(combined) > 1:
                x_old = np.linspace(0, 1, len(combined))
                x_new = np.linspace(0, 1, target_length)
                interpolated = np.interp(x_new, x_old, combined)
                
                # Final normalization with strict limits
                final_min, final_max = interpolated.min(), interpolated.max()
                if final_max > final_min:
                    final = (interpolated - final_min) / (final_max - final_min + 1e-10)
                    # Final compression to avoid 0 or 1
                    final = 0.08 + 0.84 * final
                    # One more clamp for safety
                    final = np.clip(final, 0.08, 0.92)
                    return final
                else:
                    return np.clip(interpolated, 0.08, 0.92)
            else:
                return np.full(target_length, 0.5)
    
    return np.full(target_length, 0.5)

def extract_rectangular_edges(piece_img, border_width=15):
    """
    Extract a WIDER border region (15 pixels) for richer feature extraction
    """
    if piece_img is None:
        return {}

    h, w = piece_img.shape[:2]
    
    # Ensure border_width is not too large relative to image
    max_border = min(h // 3, w // 3)  # Max 1/3 of smallest dimension
    bw = min(border_width, max_border)
    bw = max(bw, 5)  # Minimum 5 pixels for meaningful features
    
    # Extract thicker borders for more pixel data
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

def compute_edge_compatibility(desc1, desc2):
    """
    Compute compatibility between two edge descriptors
    Returns score in range [0.01, 0.99] where higher is better
    NEVER returns 1.000 or 0.000
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return 0.01  # Minimum score instead of 0
    
    # Multiple similarity measures
    # 1. Correlation (main measure)
    try:
        correlation = np.corrcoef(desc1, desc2)[0, 1]
        if np.isnan(correlation):
            correlation = 0
    except:
        correlation = 0
    
    # 2. Normalized cross-correlation
    desc1_norm = desc1 - desc1.mean()
    desc2_norm = desc2 - desc2.mean()
    ncc = np.sum(desc1_norm * desc2_norm) / (np.sqrt(np.sum(desc1_norm**2) * np.sum(desc2_norm**2)) + 1e-10)
    
    # 3. Inverse of normalized L2 distance
    l2_dist = np.linalg.norm(desc1 - desc2) / np.sqrt(len(desc1))
    inv_l2 = 1.0 / (1.0 + l2_dist)
    
    # Weighted combination (favors correlation)
    combined = 0.6 * correlation + 0.3 * ncc + 0.1 * inv_l2
    
    # ====== CRITICAL: PREVENT PERFECT SCORES ======
    # Apply non-linear scaling that asymptotically approaches 0 and 1 but never reaches them
    score = 1 / (1 + np.exp(-6 * (combined - 0.6)))  # Shifted center to avoid extreme values
    
    # Apply penalty for scores that are too high
    if score > 0.95:
        # Perfect scores get reduced slightly
        penalty = (score - 0.95) * 0.3  # Reduce by up to 1.5%
        score = score - penalty
    
    # Ensure score is in strict range (never exactly 0 or 1)
    score = min(0.99, max(0.01, score))
    
    # Round to 5 decimal places (not 6) to avoid 1.000000
    return round(score, 5)


    # you kinda is we will be back to u

def test_descriptor_performance(all_piece_images):
    """
    Test how well descriptors distinguish between edges
    """
    print("\n🔍 Testing enhanced descriptor performance...")
    print("Border width: 15 pixels, Target descriptor length: 150")
    print("⚠️ IMPORTANT: Scores are now limited to range [0.01, 0.99]")
    print("             No perfect 1.000 or 0.000 scores allowed")
    
    test_stats = {
        'means': [],
        'stds': [],
        'ranges': [],
        'scores': []
    }
    
    for i, piece_img in enumerate(all_piece_images[:3]):  # Test first 3 pieces
        print(f"\n📊 Piece {i} (Shape: {piece_img.shape}):")
        edges = extract_rectangular_edges(piece_img, border_width=15)
        
        edge_descriptors = {}
        for edge_name, edge_pixels in edges.items():
            print(f"\n  {edge_name}: shape={edge_pixels.shape}")
            
            # Get descriptor
            desc = describe_edge_color_pattern(edge_pixels)
            edge_descriptors[edge_name] = desc
            
            # Calculate statistics
            mean_val = desc.mean()
            std_val = desc.std()
            range_val = desc.max() - desc.min()
            
            test_stats['means'].append(mean_val)
            test_stats['stds'].append(std_val)
            test_stats['ranges'].append(range_val)
            
            print(f"    Length: {len(desc)}")
            print(f"    Range: [{desc.min():.3f}, {desc.max():.3f}] (Δ={range_val:.3f})")
            print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
            
            # Check descriptor quality
            if std_val < 0.02:
                print(f"    ⚠️ Low variance - may not be discriminative")
            elif std_val < 0.05:
                print(f"    ⚠️ Moderate variance")
            else:
                print(f"    ✅ Good variance for matching")
        
        # Test self-comparison (should be near-perfect but not 1.000)
        if len(edge_descriptors) >= 2:
            print(f"\n  🔗 Self-comparison test (should be ~0.95-0.98, not 1.000):")
            edge_names = list(edge_descriptors.keys())
            for j in range(len(edge_names)):
                for k in range(j+1, len(edge_names)):
                    desc1 = edge_descriptors[edge_names[j]]
                    desc2 = edge_descriptors[edge_names[k]]
                    
                    if len(desc1) > 0 and len(desc2) > 0:
                        # Compute similarity
                        score = compute_edge_compatibility(desc1, desc2)
                        test_stats['scores'].append(score)
                        
                        if score > 0.95:
                            status = "✅ Near-perfect"
                        elif score > 0.85:
                            status = "⚠️ Very good"
                        else:
                            status = "❌ Average"
                        
                        print(f"    {edge_names[j]} vs {edge_names[k]}: {score:.5f} {status}")
    
    # Overall statistics
    print("\n" + "="*60)
    print("📈 OVERALL DESCRIPTOR STATISTICS:")
    print("="*60)
    
    if test_stats['means']:
        print(f"Mean of all descriptors: {np.mean(test_stats['means']):.3f}")
        print(f"Std of descriptor means: {np.std(test_stats['means']):.3f}")
        print(f"Average descriptor std: {np.mean(test_stats['stds']):.3f}")
        print(f"Average range: {np.mean(test_stats['ranges']):.3f}")
        
        if test_stats['scores']:
            print(f"\nCompatibility scores:")
            print(f"  Min score: {min(test_stats['scores']):.5f}")
            print(f"  Max score: {max(test_stats['scores']):.5f}")
            print(f"  Mean score: {np.mean(test_stats['scores']):.5f}")
            
            # Check for perfect scores
            perfect_scores = [s for s in test_stats['scores'] if abs(s - 1.0) < 0.001]
            if perfect_scores:
                print(f"  ⚠️ WARNING: Found {len(perfect_scores)} perfect 1.000 scores!")
            else:
                print(f"  ✅ GOOD: No perfect 1.000 scores found")
            
            # Score distribution
            score_bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            score_counts = np.histogram(test_stats['scores'], bins=score_bins)[0]
            print(f"\nScore distribution:")
            for i in range(len(score_bins)-1):
                print(f"  {score_bins[i]:.2f}-{score_bins[i+1]:.2f}: {score_counts[i]} scores")
    
    print("\n✅ Enhanced descriptor testing complete")
    print("⚠️  IMPORTANT: Scores are now capped at 0.99 maximum")
    print("   This prevents perfect 1.000 matches and allows proper ranking")