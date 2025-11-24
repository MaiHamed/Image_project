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
# PART 6: EDGE DESCRIPTOR EXTRACTION & ANALYSIS
# ==============================================================
print("\n" + "🧬" * 10 + " PART 6: EDGE DESCRIPTOR EXTRACTION " + "🧬" * 10)

def extract_rectangular_edges(piece_img):
    """
    Extracts pixel values from the 4 borders of a rectangular piece.
    Returns a dictionary of arrays.
    """
    if piece_img is None: return {}
    
    h, w = piece_img.shape[:2]
    
    # Extract raw pixel strips
    edges = {
        'top': piece_img[0, :],           # Top row
        'bottom': piece_img[h-1, :],      # Bottom row
        'left': piece_img[:, 0],          # Left column
        'right': piece_img[:, w-1]        # Right column
    }
    return edges

def describe_edge_color_pattern(edge_pixels, target_length=100):
    """
    Converts a strip of colored pixels into a normalized 1D intensity descriptor.
    (Adapted from your code)
    """
    if len(edge_pixels) == 0: return np.array([])
    
    # Convert BGR to Grayscale Intensity if needed
    if len(edge_pixels.shape) > 1 and edge_pixels.shape[1] == 3:
        # Standard luminance formula: 0.299R + 0.587G + 0.114B
        # Note: OpenCV is BGR, so: 0.114*B + 0.587*G + 0.299*R
        intensities = 0.114 * edge_pixels[:, 0] + 0.587 * edge_pixels[:, 1] + 0.299 * edge_pixels[:, 2]
    else:
        intensities = edge_pixels.flatten()
        
    # Interpolate to fixed length (invariant to piece size)
    x_old = np.linspace(0, 1, len(intensities))
    x_new = np.linspace(0, 1, target_length)
    descriptor = np.interp(x_new, x_old, intensities)
    
    # Normalize values to 0-1 range
    if descriptor.max() > descriptor.min():
        descriptor = (descriptor - descriptor.min()) / (descriptor.max() - descriptor.min())
    else:
        descriptor = np.zeros_like(descriptor) # Handle flat color edges
        
    return descriptor

def compare_edges(desc1, desc2):
    """Calculates difference score (Lower is better match)"""
    if len(desc1) == 0 or len(desc2) == 0: return float('inf')
    
    # Mean Squared Error
    score = np.mean((desc1 - desc2) ** 2)
    return score

# ==============================================================
# PART 6: EDGE DESCRIPTOR EXTRACTION & ANALYSIS (CORRECTED MATCH VISUALIZATION)
# ==============================================================
print("\n" + "🧬" * 10 + " PART 6: EDGE DESCRIPTOR EXTRACTION & ANALYSIS " + "🧬" * 10)

# Re-using previously defined functions:
# extract_rectangular_edges, describe_edge_color_pattern, compare_edges

def visualize_piece_descriptors(piece_img, descriptors, filename):
    """
    Visualizes the piece and the descriptors for its 4 sides
    """
    plt.figure(figsize=(14, 8))
    
    # Center: The Piece Image
    ax_main = plt.subplot2grid((3, 3), (1, 1))
    ax_main.imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f"Piece: {filename}", fontweight='bold')
    ax_main.axis('off')
    
    # Top Descriptor
    ax_top = plt.subplot2grid((3, 3), (0, 1))
    ax_top.plot(descriptors['top'], 'r-')
    ax_top.set_title("Top Edge Signal")
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')
    
    # Bottom Descriptor
    ax_bottom = plt.subplot2grid((3, 3), (2, 1))
    ax_bottom.plot(descriptors['bottom'], 'r-')
    ax_bottom.set_title("Bottom Edge Signal")
    ax_bottom.set_ylim(0, 1)
    ax_bottom.invert_yaxis() 
    ax_bottom.axis('off')
    
    # Left Descriptor
    ax_left = plt.subplot2grid((3, 3), (1, 0))
    ax_left.plot(descriptors['left'], 'b-')
    ax_left.set_title("Left Edge")
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')

    # Right Descriptor
    ax_right = plt.subplot2grid((3, 3), (1, 2))
    ax_right.plot(descriptors['right'], 'b-')
    ax_right.set_title("Right Edge")
    ax_right.set_ylim(0, 1)
    ax_right.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_correct_match(piece1_img, piece2_img, desc1, desc2, score, match_type):
    """
    Plots two adjacent pieces and their corresponding matching edges and descriptors.
    """
    
    if match_type == 'Horizontal':
        edge1_name = "Right Edge (P1)"
        edge2_name = "Left Edge (P2)"
        title_tag = "Horizontal Match (P1 vs P2)"
    else: # Vertical
        edge1_name = "Bottom Edge (P1)"
        edge2_name = "Top Edge (P2)"
        title_tag = "Vertical Match (P1 vs P(1+N))"
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2, 1]})

    # 1. Piece 1 and Piece 2 Visualization
    combined_img = np.hstack((piece1_img, piece2_img))
    ax1.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"{match_type} Check (Score: {score:.4f})", fontweight='bold')
    ax1.axis('off')

    # 2. Descriptor Comparison Plot
    ax2.plot(desc1, 'b-', label=edge1_name, linewidth=2)
    
    # If Horizontal, reverse the second descriptor so the signals overlap correctly.
    if match_type == 'Horizontal':
        desc2_rev = desc2[::-1] # Reverse the signal for Left Edge
        ax2.plot(desc2_rev, 'r--', label=f'{edge2_name} (Reversed)', linewidth=2, alpha=0.7)
    else: # Vertical (Bottom-to-Top)
        ax2.plot(desc2, 'r--', label=edge2_name, linewidth=2, alpha=0.7)
        
    ax2.set_title(f"Descriptor Overlap: {title_tag}")
    ax2.set_xlabel("Normalized Position along Edge")
    ax2.set_ylabel("Normalized Intensity (0-1)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Simple Error/Difference Plot (Optional but helpful)
    if match_type == 'Horizontal':
        diff = np.abs(desc1 - desc2_rev)
    else:
        diff = np.abs(desc1 - desc2)
        
    ax3.plot(diff, 'g-', linewidth=1)
    ax3.fill_between(range(len(diff)), diff, color='g', alpha=0.1)
    ax3.set_title(f"Absolute Difference (Mean Error: {score:.4f})")
    ax3.set_xlabel("Position")
    ax3.set_ylim(0, 0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Run Part 6
if 'images_by_dir' in locals():
    print("🔍 Extracting descriptors from processed pieces...")
    
    processed_count = 0
    examples_to_show = 4
    examples_shown_desc = 0
    examples_shown_match = 0
    
    for dir_name, dir_images in images_by_dir.items():
        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")
        
        if not os.path.exists(rectangular_dir):
            continue
            
        print(f"\n📁 Analyzing pieces in: {rectangular_dir}")
        
        piece_files = sorted([f for f in os.listdir(rectangular_dir) if f.startswith("piece_") and f.endswith(('.png', '.jpg'))])
        if not piece_files:
            continue

        pieces_by_puzzle = {}
        for p_file in piece_files:
            parts = p_file.split('_')
            original_name = "_".join(parts[2:]) 
            pieces_by_puzzle.setdefault(original_name, []).append(p_file)
            
        # Process each puzzle group
        for puzzle_name, pieces in pieces_by_puzzle.items():
            
            pieces.sort(key=lambda x: int(x.split('_')[1]))
            
            # --- 1. DETECT GRID SIZE FOR SOLVING LOGIC ---
            N = 2 
            if '2x2' in dir_name.lower() or '4' in pieces[0]: N = 2
            if '4x4' in dir_name.lower() or '16' in pieces[0]: N = 4
            if '8x8' in dir_name.lower() or '64' in pieces[0]: N = 8

            # Try to guess N based on total pieces if possible
            if len(pieces) > 0:
                potential_N = int(np.sqrt(len(pieces)))
                if potential_N in [2, 3, 4, 8]: N = potential_N
            
            total_pieces = N * N
            
            print(f"\n--- 🧩 PUZZLE: {puzzle_name} ({N}x{N} Grid, {len(pieces)} pieces) ---")
            
            # --- 2. DESCRIPTOR VISUALIZATION (Piece 1) ---
            if examples_shown_desc < examples_to_show and pieces:
                first_piece_path = os.path.join(rectangular_dir, pieces[0])
                first_piece_img = cv2.imread(first_piece_path)
                
                if first_piece_img is not None:
                    print(f"   🧩 Visualizing all 4 descriptors for: {pieces[0]}")
                    raw_edges = extract_rectangular_edges(first_piece_img)
                    descriptors = {k: describe_edge_color_pattern(v) for k, v in raw_edges.items()}
                    visualize_piece_descriptors(first_piece_img, descriptors, pieces[0])
                    examples_shown_desc += 1


            # --- 3. CORRECT MATCH ANALYSIS & VISUALIZATION ---
            if len(pieces) == total_pieces and examples_shown_match < 4: # Only show one visual match example
                print(f"   📐 Visualizing Correct Sequential Matches (Piece 1 <-> Piece 2 and P1 <-> P(1+N))")
                
                # Check 2 pairs (Piece 1 -> Piece 2 and Piece N -> Piece N+1)
                comparison_indices = [
                    (0, 1, "Horizontal"),                         # Horizontal Check (P1 -> P2)
                    (0, N, "Vertical")                            # Vertical Check (P1 -> P(1+N))
                ]
                
                for idx1, idx2, match_type in comparison_indices:
                    if idx2 < len(pieces):
                        p1_file = pieces[idx1]
                        p2_file = pieces[idx2]
                        
                        p1_img = cv2.imread(os.path.join(rectangular_dir, p1_file))
                        p2_img = cv2.imread(os.path.join(rectangular_dir, p2_file))
                        
                        if p1_img is not None and p2_img is not None:
                            edges1 = extract_rectangular_edges(p1_img)
                            edges2 = extract_rectangular_edges(p2_img)
                            
                            if match_type == "Horizontal":
                                # Compare P1 Right to P2 Left (The correct sequential match)
                                desc1 = describe_edge_color_pattern(edges1['right'])
                                desc2 = describe_edge_color_pattern(edges2['left'])
                                score = compare_edges(desc1, desc2[::-1]) # Reverse desc2 for true comparison
                                
                                print(f"      • P{idx1+1} Right <-> P{idx2+1} Left (Correct Match): Score: {score:.4f}")
                                visualize_correct_match(p1_img, p2_img, desc1, desc2, score, match_type)
                                
                            elif match_type == "Vertical":
                                # Compare P1 Bottom to P2 Top (The correct sequential match)
                                desc1 = describe_edge_color_pattern(edges1['bottom'])
                                desc2 = describe_edge_color_pattern(edges2['top'])
                                score = compare_edges(desc1, desc2)
                                
                                print(f"      • P{idx1+1} Bottom <-> P{idx2+1} Top (Correct Match): Score: {score:.4f}")
                                visualize_correct_match(p1_img, p2_img, desc1, desc2, score, match_type)

                            examples_shown_match += 1
                
                print("   ---")
            else:
                if len(pieces) != total_pieces:
                    print(f"   ⚠️ Skipping sequential check: Found {len(pieces)} pieces, expected {total_pieces}.")
                
            processed_count += 1
            if processed_count >= 5: break # Limit output for log cleanliness
            
    print(f"\n" + "=" * 70)
    print("PART 6 COMPLETE: Descriptors extracted and visually verified using correct adjacent pieces.")
    print("=" * 70)
else:
    print("❌ Previous steps not completed. Variables missing.")
