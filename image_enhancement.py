import cv2
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename

print("🖼️ Puzzle Image Processor")
print("=" * 50)

def selective_median_filter(img, threshold=50):
    # Convert to grayscale for noise detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    denoised = img.copy()

    # Process each pixel
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
    
    kernel = np.ones((1,1), np.uint8) 
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    return edges

def enhance_image(img, low_threshold=50, high_threshold=150):
    enhanced = img.copy()

    # CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    #sharpening
    lap = cv2.Laplacian(enhanced, cv2.CV_32F)
    lap -= lap.mean()
    sharpened = enhanced - 0.1 * lap
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # clean edges using Canny
    edges_bw = canny_edges(sharpened, low_threshold, high_threshold)

    return sharpened, edges_bw


# -------------------- UPLOAD ZIP --------------------
print("\n📦 SELECT PUZZLE ZIP FOLDER")

Tk().withdraw() 
zip_file = askopenfilename(title="Select ZIP file", filetypes=[("ZIP files", "*.zip")])

if not zip_file:
    print("❌ No file selected!")
    exit()

print(f"✅ Selected: {zip_file}")

# Extract ZIP
extract_dir = os.path.join(os.path.dirname(zip_file), "puzzle_images")

if os.path.exists(extract_dir):
    print("🧹 Cleaning previous extraction...")
    shutil.rmtree(extract_dir)

os.makedirs(extract_dir, exist_ok=True)

print("📂 Extracting zip file...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Extraction complete!")

# -------------------- FIND IMAGES --------------------
print("\n🔍 Finding all images...")
image_files = []

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.startswith("._"):
            continue
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            full_path = os.path.join(root, file)
            image_files.append({
                'full_path': full_path,
                'filename': file,
                'directory': os.path.relpath(root, extract_dir)
            })

if not image_files:
    print("❌ No images found in the extracted folder!")
    for item in os.listdir(extract_dir):
        item_path = os.path.join(extract_dir, item)
        print(f"  {'📁' if os.path.isdir(item_path) else '📄'} {item}")
else:
    images_by_dir = {}
    for img in image_files:
        dir_name = img['directory'] if img['directory'] != '.' else 'root'
        images_by_dir.setdefault(dir_name, []).append(img)

    print(f"\n📊 Found {len(image_files)} images in {len(images_by_dir)} locations:")
    for dir_name, images in images_by_dir.items():
        print(f"   📁 {dir_name}: {len(images)} images")

    print(f"\n🖼️ First 10 images:")
    for i, img in enumerate(image_files[:10]):
        print(f"   {i+1}. {img['filename']} ({img['directory']})")

    if len(image_files) > 10:
        print(f"   ... and {len(image_files) - 10} more images")

    print(f"\n🎯 Ready to process {len(image_files)} images!")

print("\n" + "=" * 50)
print("🎉 Puzzle Image Processing Complete!")



# PART 2: APPLY FILTERS TO ALL EXTRACTED IMAGES
print("\n" + "🔧" * 10 + " PART 2: APPLY FILTER TO ALL IMAGES " + "🔧" * 10)

# Create main output directory relative to zip location
output_dir = os.path.join(os.path.dirname(zip_file), "processed_puzzles")
os.makedirs(output_dir, exist_ok=True)

# Process ALL images found during extraction
if 'image_files' not in locals() or not image_files:
    print("❌ No images found! Please run Part 1 first.")
else:
    print(f"🎯 Processing ALL extracted images")
    print(f"📊 Total images found: {len(image_files)}")

    # Group by directory for organized output
    images_by_dir = {}
    for img in image_files:
        dir_name = img['directory'] if img['directory'] != '.' else 'root'
        images_by_dir.setdefault(dir_name, []).append(img)

    # Process images from each directory
    examples = []
    total_successful = 0

    for dir_name, dir_images in images_by_dir.items():
        print(f"\n📁 Processing directory: {dir_name} ({len(dir_images)} images)")

        # Create output subfolder for this directory
        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        os.makedirs(puzzle_output_dir, exist_ok=True)

        orig_dir = os.path.join(puzzle_output_dir, "original")
        denoise_dir = os.path.join(puzzle_output_dir, "denoised")
        enhance_dir = os.path.join(puzzle_output_dir, "enhanced")
        edges_dir = os.path.join(puzzle_output_dir, "edges")

        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(denoise_dir, exist_ok=True)
        os.makedirs(enhance_dir, exist_ok=True)
        os.makedirs(edges_dir, exist_ok=True)

        dir_successful = 0

        for i, img_info in enumerate(dir_images, 1):
            img_path = img_info['full_path']
            filename = img_info['filename']

            img = cv2.imread(img_path)

            if img is not None:
                # Apply selective median filter
                denoised = selective_median_filter(img, threshold=50)

                # Apply enhancement
                enhanced, edges_bw = enhance_image(denoised, low_threshold=50, high_threshold=150)


                cv2.imwrite(os.path.join(orig_dir, filename), img)
                cv2.imwrite(os.path.join(denoise_dir, filename), denoised)
                cv2.imwrite(os.path.join(enhance_dir, filename), enhanced)
                cv2.imwrite(os.path.join(edges_dir, f"edges_{filename}"), edges_bw)


                # Store examples for preview
                
                examples.append({
                    'original': img,
                    'denoised': denoised,
                    'enhanced': enhanced,
                    'edges_bw': edges_bw,
                    'filename': filename,
                    'directory': dir_name
                })


                dir_successful += 1
                total_successful += 1

                if i <= 5 or i == len(dir_images):  # Show first 5 and last
                    print(f"   ✅ [{i}/{len(dir_images)}] {filename}")
                elif i == 6:
                    print(f"   ... processing {len(dir_images) - 5} more images ...")
            else:
                print(f"   ❌ [{i}/{len(dir_images)}] Failed: {filename}")

        print(f"    {dir_name}: {dir_successful}/{len(dir_images)} successful")

    # Final summary
    print(f"\n PROCESSING COMPLETE:")
    print(f"    Total processed: {total_successful}/{len(image_files)} images")
    print(f"    Output directory: {output_dir}")

print("\n" + "=" * 60)
print("PART 2 COMPLETE - ALL IMAGES PROCESSED!")
print("=" * 60)

print(f"\nCheck the output folder here: {output_dir}")
print("Folders inside:")
print(os.listdir(output_dir))

# PART 3: SHOW EXAMPLES
print("\n" + "🔍" * 10 + " PART 3: SHOW EXAMPLES & METRICS " + "🔍" * 10)

if 'examples' not in locals() or not examples:
    print("❌ No images were processed! Please run Part 2 first.")
else:
    # Show processing summary
    print(f"🎯 You processed {len(image_files)} images from {len(images_by_dir)} locations")

    # Show locations summary
    print(f"\n📂 Processed locations:")
    for dir_name, dir_images in images_by_dir.items():
        print(f"   {dir_name}: {len(dir_images)} images")

    # Ask user how many examples to show
    try:
        max_examples = len(examples)
        num_examples = int(input(f"\nHow many before/after examples do you want to see? (1-{max_examples}): "))
        num_examples = max(1, min(num_examples, max_examples))
        print(f"✅ Will show {num_examples} examples")
    except ValueError:
        num_examples = min(3, len(examples)) 
        print(f"⚠️ Invalid input. Showing all {num_examples} examples by default")



    # Show examples
    print(f"\n🖼️ Before/After Examples:")
    print("-" * 60)

    for i, example in enumerate(examples[:num_examples], 1):
        print(f"\n🖼️ Example {i}/{num_examples}: {example['filename']}")
        print(f"📁 Location: {example['directory']}")

        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(example['original'], cv2.COLOR_BGR2RGB)
        denoised_rgb = cv2.cvtColor(example['denoised'], cv2.COLOR_BGR2RGB)

        # Display side by side
        plt.figure(figsize=(18, 6))

        plt.subplot(1,4,1)
        plt.imshow(cv2.cvtColor(example['original'], cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(cv2.cvtColor(example['denoised'], cv2.COLOR_BGR2RGB))
        plt.title("Denoised")
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(cv2.cvtColor(example['enhanced'], cv2.COLOR_BGR2RGB))
        plt.title("Enhanced")
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(example['edges_bw'], cmap='gray')
        plt.title("Edges")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Print image info
        print(f"📏 Image size: {example['original'].shape[1]} x {example['original'].shape[0]}")
        print(f"🎨 Channels: {example['original'].shape[2] if len(example['original'].shape) == 3 else 1}")

        # Noise reduction metrics
        orig_gray = cv2.cvtColor(example['original'], cv2.COLOR_BGR2GRAY)
        denoised_gray = cv2.cvtColor(example['denoised'], cv2.COLOR_BGR2GRAY)

        orig_variance = np.var(orig_gray)
        denoised_variance = np.var(denoised_gray)
        noise_reduction = ((orig_variance - denoised_variance) / orig_variance) * 100

        print(f"📊 Noise reduction: {noise_reduction:.1f}%")
        print(f"   Original variance: {orig_variance:.1f}")
        print(f"   Denoised variance: {denoised_variance:.1f}")

    # Show available output directories
    output_subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if output_subdirs:
        print(f"\n📁 Available processed folders:")
        for i, subdir in enumerate(output_subdirs, 1):
            subdir_path = os.path.join(output_dir, subdir)
            processed_count = len([f for f in os.listdir(subdir_path) if f.startswith('filtered_')])
            print(f"   {i}. {subdir} ({processed_count} images)")

print("\n" + "=" * 60)
print("PART 3 COMPLETE - EXAMPLES SHOWN ")
print("=" * 60)

# ==============================================================
# PART 4 VISUALIZATION: SLIDING TILE PUZZLE SEGMENTATION
# ==============================================================
print("\n" + "📊" * 10 + " SLIDING TILE PUZZLE SEGMENTATION RESULTS " + "📊" * 10)

# Define segmentation_output path
segmentation_output = os.path.join(output_dir, "segmented_pieces")
print(f"📁 Segmentation output directory: {segmentation_output}")

def improved_cleaning(thresh):
    """
    Improved cleaning with multiple strategies and automatic selection
    """
    height, width = thresh.shape
    
    # Strategy 1: Very gentle cleaning (small kernel, 1 iteration)
    kernel_small = np.ones((2, 2), np.uint8)
    clean1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Strategy 2: No cleaning (if threshold is already good)
    clean2 = thresh.copy()
    
    # Strategy 3: Try opening instead of closing (removes small noise)
    kernel_tiny = np.ones((1, 1), np.uint8)
    clean3 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
    
    # Strategy 4: Adaptive cleaning based on image characteristics
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    white_ratio = white_pixels / total_pixels
    
    # Auto-select the best strategy
    if white_ratio < 0.1:  # Very few white pixels - might need gentle closing
        print(f"   🧹 Using gentle closing (low white ratio: {white_ratio:.3f})")
        return clean1
    elif white_ratio > 0.8:  # Too many white pixels - might need opening
        print(f"   🧹 Using opening (high white ratio: {white_ratio:.3f})")
        return clean3
    else:  # Moderate white ratio - try no cleaning first
        print(f"   🧹 Using no cleaning (moderate white ratio: {white_ratio:.3f})")
        return clean2

def visualize_sliding_tile_thresholding(original_img, processed_images, filename):
    """
    Visualize sliding tile puzzle thresholding process - stops at cleaned mask
    """
    # Display side by side
    plt.figure(figsize=(18, 6))

    plt.subplot(1,4,1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("1. Original Puzzle", fontweight='bold')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(processed_images['gray'], cmap='gray')
    plt.title("2. Grayscale", fontweight='bold')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(processed_images['thresh'], cmap='gray')
    plt.title("3. Adaptive Threshold", fontweight='bold')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(processed_images['clean'], cmap='gray')
    plt.title("4. Cleaned Mask", fontweight='bold')
    plt.axis('off')

    plt.suptitle(f'Sliding Tile Puzzle Thresholding: {filename}', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()

def analyze_threshold_quality(thresh_img, clean_img):
    """
    Analyze the quality of thresholding results
    """
    # Count white pixels (potential puzzle pieces)
    white_pixels_thresh = np.sum(thresh_img == 255)
    white_pixels_clean = np.sum(clean_img == 255)
    
    # Calculate noise reduction
    noise_reduction = ((white_pixels_thresh - white_pixels_clean) / white_pixels_thresh * 100) if white_pixels_thresh > 0 else 0
    
    return {
        'white_pixels_thresh': white_pixels_thresh,
        'white_pixels_clean': white_pixels_clean,
        'noise_reduction': noise_reduction,
        'total_pixels': thresh_img.size
    }

# Run the thresholding visualization
if 'image_files' in locals() and image_files:
    print("🔍 Processing ALL images for thresholding pipeline...")
    print("   Applying grayscale, threshold, and cleaned mask to ALL images")
    
    examples_to_show = min(2, len(image_files))
    total_processed = 0
    
    for dir_name, dir_images in images_by_dir.items():
        print(f"\n📁 Processing directory: {dir_name} ({len(dir_images)} images)")
        
        # Create threshold subfolders in the same directory structure
        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        
        grayscale_dir = os.path.join(puzzle_output_dir, "grayscale")
        threshold_dir = os.path.join(puzzle_output_dir, "threshold")
        cleaned_dir = os.path.join(puzzle_output_dir, "cleaned")
        
        os.makedirs(grayscale_dir, exist_ok=True)
        os.makedirs(threshold_dir, exist_ok=True)
        os.makedirs(cleaned_dir, exist_ok=True)
        
        dir_processed = 0
        
        for i, img_info in enumerate(dir_images, 1):
            img_path = img_info['full_path']
            filename = img_info['filename']
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"   ❌ [{i}/{len(dir_images)}] Failed: {filename}")
                continue
            
            # Thresholding pipeline for SLIDING TILES
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 51, 10)
            
            # Use IMPROVED cleaning instead of aggressive cleaning
            clean = improved_cleaning(thresh)
            
            # Save the thresholding results for ALL images
            cv2.imwrite(os.path.join(grayscale_dir, filename), gray)
            cv2.imwrite(os.path.join(threshold_dir, filename), thresh)
            cv2.imwrite(os.path.join(cleaned_dir, filename), clean)
            
            dir_processed += 1
            total_processed += 1
            
            # Only show visualization for first few examples
            if i <= examples_to_show:
                print(f"\n🧩 Visualizing Sliding Tile Puzzle {i}/{examples_to_show}: {filename}")
                
                # Store processed images for visualization
                processed_imgs = {
                    'gray': gray,
                    'thresh': thresh,
                    'clean': clean
                }
                
                # Show the thresholding visualization
                visualize_sliding_tile_thresholding(img, processed_imgs, filename)
                
                # Analyze threshold quality
                analysis = analyze_threshold_quality(thresh, clean)
                
                # Print analysis results
                print(f"   📊 Threshold Analysis:")
                print(f"      • White pixels in threshold: {analysis['white_pixels_thresh']:,}")
                print(f"      • White pixels after cleaning: {analysis['white_pixels_clean']:,}")
                print(f"      • Noise reduction: {analysis['noise_reduction']:.1f}%")
                print(f"      • Coverage: {(analysis['white_pixels_clean']/analysis['total_pixels']*100):.1f}% of image")
                
                # Quality assessment
                coverage = analysis['white_pixels_clean'] / analysis['total_pixels']
                if coverage < 0.1:
                    print(f"   ⚠️  LOW COVERAGE: May be missing pieces")
                elif coverage > 0.8:
                    print(f"   ⚠️  HIGH COVERAGE: May have too much noise")
                else:
                    print(f"   ✅ GOOD COVERAGE: Appropriate for puzzle pieces")
            
            # Show progress for all images
            if i <= 5 or i == len(dir_images):
                print(f"   ✅ [{i}/{len(dir_images)}] Processed: {filename}")
            elif i == 6:
                print(f"   ... processing {len(dir_images) - 5} more images ...")
        
        print(f"    {dir_name}: {dir_processed}/{len(dir_images)} successful")

    # Final summary
    print(f"\n📊 THRESHOLDING PROCESSING COMPLETE:")
    print(f"    Total images processed: {total_processed}/{len(image_files)}")
    print(f"    Visualization shown for: {examples_to_show} examples")
    
    # Show updated folder structure
    print(f"\n📁 Updated folder structure in each directory:")
    for dir_name in images_by_dir.keys():
        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        if os.path.exists(puzzle_output_dir):
            subdirs = [d for d in os.listdir(puzzle_output_dir) if os.path.isdir(os.path.join(puzzle_output_dir, d))]
            print(f"   {puzzle_output_dir}/")
            for subdir in sorted(subdirs):
                file_count = len([f for f in os.listdir(os.path.join(puzzle_output_dir, subdir)) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"   ├── {subdir}/ ({file_count} images)")

else:
    print("❌ No images available for thresholding visualization!")

print("\n" + "=" * 70)
print("SLIDING TILE PUZZLE THRESHOLDING COMPLETE!")
print("=" * 70)

# ==============================================================
# PART 5: GENERIC FIXED GRID CROPPING (2x2, 4x4, 8x8)
# ==============================================================
print("\n" + "🧩" * 10 + " PART 5: GENERIC GRID CROPPING " + "🧩" * 10)

def detect_grid_size(filename, dirname, default_n=2):
    """
    Determines grid size (N x N) by looking for patterns like '4x4', '8x8' 
    in the filename or directory name. Defaults to 2x2.
    """
    search_str = (filename + "_" + dirname).lower()
    
    if "8x8" in search_str:
        return 8
    elif "4x4" in search_str:
        return 4
    elif "2x2" in search_str:
        return 2
    
    # If no label found, assume standard 2x2
    return default_n

def extract_generic_grid_pieces(img, N=2):
    """
    Slices an image into an N x N grid.
    Returns: List of (N*N) images ordered row by row.
    """
    if img is None:
        return []
        
    height, width = img.shape[:2]
    
    # Calculate step sizes
    step_y = height // N
    step_x = width // N
    
    pieces = []
    
    # Loop through rows and columns
    for row in range(N):
        for col in range(N):
            # Calculate coordinates
            y1 = row * step_y
            y2 = (row + 1) * step_y
            x1 = col * step_x
            x2 = (col + 1) * step_x
            
            # Handle edge cases (last row/col takes remaining pixels)
            if row == N - 1: y2 = height
            if col == N - 1: x2 = width
            
            # Crop
            piece = img[y1:y2, x1:x2].copy()
            pieces.append(piece)
            
    return pieces

def visualize_generic_grid(original_img, pieces, N, filename):
    """
    Visualizes the N x N cuts and the resulting pieces
    """
    height, width = original_img.shape[:2]
    
    # 1. Create visualization of the cuts
    cut_viz = original_img.copy()
    step_y = height // N
    step_x = width // N
    
    # Draw vertical lines
    for col in range(1, N):
        x = col * step_x
        cv2.line(cut_viz, (x, 0), (x, height), (0, 255, 0), 2)
        
    # Draw horizontal lines
    for row in range(1, N):
        y = row * step_y
        cv2.line(cut_viz, (0, y), (width, y), (0, 255, 0), 2)

    # Setup Plot
    plt.figure(figsize=(15, 6))
    
    # Left: The cut lines
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cut_viz, cv2.COLOR_BGR2RGB))
    plt.title(f"Grid Slicing ({N}x{N}): {filename}", fontweight='bold')
    plt.axis('off')
    
    # Right: The extracted pieces in a grid
    plt.subplot(1, 2, 2)
    
    # Create a display grid
    # Add gaps between pieces
    gap = 5
    piece_h, piece_w = pieces[0].shape[:2]
    
    grid_h = N * piece_h + (N-1) * gap
    grid_w = N * piece_w + (N-1) * gap
    
    display_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8) + 255 # White bg
    
    idx = 0
    for row in range(N):
        for col in range(N):
            if idx < len(pieces):
                y = row * (piece_h + gap)
                x = col * (piece_w + gap)
                
                # Resize strictly to match expected slot if slight variation
                p = cv2.resize(pieces[idx], (piece_w, piece_h))
                
                display_grid[y:y+piece_h, x:x+piece_w] = p
                idx += 1
    
    plt.imshow(cv2.cvtColor(display_grid, cv2.COLOR_BGR2RGB))
    plt.title(f"Extracted {N*N} Pieces", fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return cut_viz

# Run Generic Grid Cropping
if 'image_files' in locals() and image_files:
    print("🔍 Executing Generic Grid Cropping...")
    print("   Method: Auto-detecting 2x2, 4x4, or 8x8 based on folder/filenames")
    
    total_pieces_extracted = 0
    processed_count = 0
    examples_to_show = 2
    
    for dir_name, dir_images in images_by_dir.items():
        print(f"\n📁 Processing directory: {dir_name} ({len(dir_images)} images)")
        
        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")
        os.makedirs(rectangular_dir, exist_ok=True)
        
        examples_shown = 0
        
        for i, img_info in enumerate(dir_images, 1):
            img_path = img_info['full_path']
            filename = img_info['filename']
            
            img = cv2.imread(img_path)
            if img is None: continue
                
            # --- 1. DETECT GRID SIZE ---
            # Looks at filename or foldername for "4x4", "8x8", etc.
            # Defaults to 2 if not found.
            N = detect_grid_size(filename, dir_name, default_n=2)
            
            # --- 2. EXTRACT PIECES ---
            pieces = extract_generic_grid_pieces(img, N=N)
            
            # Save the pieces
            for p_idx, piece in enumerate(pieces):
                # naming: piece_1_image.jpg ... piece_16_image.jpg
                piece_filename = f"piece_{p_idx+1}_{filename}"
                cv2.imwrite(os.path.join(rectangular_dir, piece_filename), piece)
            
            total_pieces_extracted += len(pieces)
            processed_count += 1
            
            # Visualization
            if examples_shown < examples_to_show:
                print(f"\n🧩 Visualizing {N}x{N} Crop {examples_shown + 1}/{examples_to_show}: {filename}")
                viz_img = visualize_generic_grid(img, pieces, N, filename)
                cv2.imwrite(os.path.join(rectangular_dir, f"grid_cut_{filename}"), viz_img)
                examples_shown += 1
                
            if i <= 5:
                print(f"   ✅ [{i}/{len(dir_images)}] {filename} -> {N}x{N} ({len(pieces)} pieces)")

    print(f"\n" + "=" * 70)
    print(f"CROPPING COMPLETE: {total_pieces_extracted} pieces extracted.")
    print("=" * 70)
else:
    print("❌ No images found.")

# ==============================================================
# PART 6: EDGE DESCRIPTOR ANALYSIS & COMPARISON
# ==============================================================
print("\n" + "🧬" * 10 + " PART 6: EDGE DESCRIPTOR ANALYSIS " + "🧬" * 10)

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

def visualize_comparison_heatmap(all_comparisons, piece_files, N, puzzle_name):
    #Create a heatmap showing how well pieces match each other
    num_pieces = len(piece_files)
    
    # Create score matrices for different edge types
    horizontal_scores = np.full((num_pieces, num_pieces), np.nan)
    vertical_scores = np.full((num_pieces, num_pieces), np.nan)
    
    for match in all_comparisons:
        i, j = match['piece1'], match['piece2']
        
        if match['edge1'] == 'right' and match['edge2'] == 'left':
            horizontal_scores[i, j] = match['score']
        elif match['edge1'] == 'bottom' and match['edge2'] == 'top':
            vertical_scores[i, j] = match['score']
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Horizontal matches heatmap
    im1 = ax1.imshow(horizontal_scores, cmap='RdYlGn_r', aspect='auto')
    ax1.set_title(f'Horizontal Match Scores\n(Piece Right ↔ Other Piece Left)', fontweight='bold')
    ax1.set_xlabel('Other Piece Number')
    ax1.set_ylabel('Piece Number')
    ax1.set_xticks(range(num_pieces))
    ax1.set_yticks(range(num_pieces))
    ax1.set_xticklabels([f'P{i+1}' for i in range(num_pieces)])
    ax1.set_yticklabels([f'P{i+1}' for i in range(num_pieces)])
    plt.colorbar(im1, ax=ax1, label='Match Score (lower = better)')
    
    # Add score values to heatmap
    for i in range(num_pieces):
        for j in range(num_pieces):
            if not np.isnan(horizontal_scores[i, j]):
                ax1.text(j, i, f'{horizontal_scores[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8, 
                        color='white' if horizontal_scores[i, j] > 0.5 else 'black')
    
    # Vertical matches heatmap
    im2 = ax2.imshow(vertical_scores, cmap='RdYlGn_r', aspect='auto')
    ax2.set_title(f'Vertical Match Scores\n(Piece Bottom ↔ Other Piece Top)', fontweight='bold')
    ax2.set_xlabel('Other Piece Number')
    ax2.set_ylabel('Piece Number')
    ax2.set_xticks(range(num_pieces))
    ax2.set_yticks(range(num_pieces))
    ax2.set_xticklabels([f'P{i+1}' for i in range(num_pieces)])
    ax2.set_yticklabels([f'P{i+1}' for i in range(num_pieces)])
    plt.colorbar(im2, ax=ax2, label='Match Score (lower = better)')
    
    # Add score values to heatmap
    for i in range(num_pieces):
        for j in range(num_pieces):
            if not np.isnan(vertical_scores[i, j]):
                ax2.text(j, i, f'{vertical_scores[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8,
                        color='white' if vertical_scores[i, j] > 0.5 else 'black')
    
    plt.suptitle(f'Puzzle Match Analysis: {puzzle_name} ({N}x{N})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return horizontal_scores, vertical_scores

def visualize_best_match_pair(piece1_img, piece2_img, desc1, desc2, score, match_info):
    #Visualize one good match pair
    piece1_num = match_info['piece1'] + 1
    piece2_num = match_info['piece2'] + 1
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Show pieces separately
    ax1.imshow(cv2.cvtColor(piece1_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Piece {piece1_num}\n{match_info["edge1"]} edge', fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(piece2_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Piece {piece2_num}\n{match_info["edge2"]} edge', fontweight='bold')
    ax2.axis('off')
    
    # Show descriptor comparison
    if match_info['edge1'] == 'right' and match_info['edge2'] == 'left':
        # For horizontal match, reverse the second descriptor
        desc2_plot = desc2[::-1]
        match_type = "Horizontal"
    else:
        desc2_plot = desc2
        match_type = "Vertical"
    
    ax3.plot(desc1, 'b-', label=f'P{piece1_num} {match_info["edge1"]}', linewidth=2)
    ax3.plot(desc2_plot, 'r--', label=f'P{piece2_num} {match_info["edge2"]}', linewidth=2, alpha=0.7)
    ax3.set_title(f'Descriptor Comparison\n{match_type} Match\nScore: {score:.4f}')
    ax3.set_xlabel('Position along edge')
    ax3.set_ylabel('Normalized Intensity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Run the comparison analysis
if 'images_by_dir' in locals():
    print("🔍 ANALYZING PIECE COMPATIBILITY - Comparing all edges...")
    
    for dir_name, dir_images in images_by_dir.items():
        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")
        
        if not os.path.exists(rectangular_dir):
            continue
            
        print(f"\n📁 Analyzing puzzle pieces from: {rectangular_dir}")
        
        piece_files = sorted([f for f in os.listdir(rectangular_dir) if f.startswith("piece_") and f.endswith(('.png', '.jpg'))])
        if not piece_files:
            continue

        # FIXED: Group pieces by puzzle correctly for "piece_4_97.jpg" format
        pieces_by_puzzle = {}
        for p_file in piece_files:
            parts = p_file.split('_')
            # For "piece_4_97.jpg": parts = ["piece", "4", "97.jpg"]
            if len(parts) >= 3:
                puzzle_id = parts[2].split('.')[0]  # Take "97" from "97.jpg"
                pieces_by_puzzle.setdefault(puzzle_id, []).append(p_file)
        
        for puzzle_id, pieces in pieces_by_puzzle.items():
            # FIXED: Sort pieces correctly - use the middle number "4", "3", "2", "1"
            pieces.sort(key=lambda x: int(x.split('_')[1]))  # "piece_4_97.jpg" -> "4"
            
            print(f"\n--- 🧩 ANALYSIS: Puzzle {puzzle_id} ({len(pieces)} pieces) ---")
            print(f"   Pieces in order: {pieces}")
            
            # Detect grid size based on number of pieces
            num_pieces = len(pieces)
            if num_pieces == 4:
                N = 2  # 2x2 puzzle
            elif num_pieces == 16:
                N = 4  # 4x4 puzzle  
            elif num_pieces == 64:
                N = 8  # 8x8 puzzle
            else:
                N = int(np.sqrt(num_pieces))  # Try to guess
                print(f"   ⚠️ Unusual piece count: {num_pieces}, assuming {N}x{N}")
            
            total_pieces = N * N
            
            if len(pieces) != total_pieces:
                print(f"   ⚠️ Skipping: {len(pieces)} pieces, expected {total_pieces} for {N}x{N}")
                continue
            
            # Load all pieces and extract descriptors
            all_piece_data = []
            all_piece_images = []
            
            for piece_file in pieces:
                piece_path = os.path.join(rectangular_dir, piece_file)
                piece_img = cv2.imread(piece_path)
                if piece_img is not None:
                    raw_edges = extract_rectangular_edges(piece_img)
                    descriptors = {k: describe_edge_color_pattern(v) for k, v in raw_edges.items()}
                    all_piece_data.append(descriptors)
                    all_piece_images.append(piece_img)
                else:
                    print(f"   ❌ Failed to load: {piece_file}")
            
            if len(all_piece_data) < 2:
                print(f"   ⚠️ Not enough pieces loaded for analysis")
                continue
            
            # Compare all pieces
            all_comparisons = analyze_all_possible_matches(all_piece_data, pieces, N)
            
            # Show heatmap visualization
            print(f"\n   📈 Generating compatibility heatmaps...")
            horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                all_comparisons, pieces, N, f"Puzzle_{puzzle_id}"
            )
            
            # Visualize a few best matches
            print(f"\n   👀 Visualizing best match examples...")
            best_matches = sorted(all_comparisons, key=lambda x: x['score'])[:3]  # Top 3 matches
            
            for match in best_matches:
                piece1_idx, piece2_idx = match['piece1'], match['piece2']
                if (piece1_idx < len(all_piece_images) and piece2_idx < len(all_piece_images)):
                    print(f"   🎯 Showing: {match['label']} (Score: {match['score']:.4f})")
                    visualize_best_match_pair(
                        all_piece_images[piece1_idx],
                        all_piece_images[piece2_idx], 
                        all_piece_data[piece1_idx][match['edge1']],
                        all_piece_data[piece2_idx][match['edge2']],
                        match['score'],
                        match
                    )
            
            # Only process first puzzle for demo
            break
        
        # Only process first directory for demo  
        break
            
    print(f"\n" + "=" * 70)
    print("COMPARISON ANALYSIS COMPLETE!")
    print("✅ Compared ALL pieces against ALL other pieces") 
    print("✅ Showed compatibility heatmaps")
    print("✅ Displayed best matches visually")
    print("=" * 70)
else:
    print("❌ Previous steps not completed.")