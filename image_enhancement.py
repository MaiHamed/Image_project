import cv2
import os
import zipfile
import numpy as np
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
print("\n🔍 Finding all images (skipping Mac system files)...")
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

                # Save the enhanced image
                output_path = os.path.join(puzzle_output_dir, f"enhanced_{filename}")
                cv2.imwrite(output_path, enhanced)

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
        import matplotlib.pyplot as plt
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
        plt.title("Edges (BW)")
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

#Descriptor demo - FOR SLIDING TILE PUZZLE
#IGNORE

def extract_edge_pixels(image, edge_mask):
    """
    Extract color/intensity values along a puzzle piece edge
    For sliding tiles, we match edges based on color patterns, not shapes!
    """
    # edge_mask should be a binary mask showing which pixels are on the edge
    edge_pixels = image[edge_mask > 0]
    return edge_pixels

def describe_edge_color_pattern(edge_pixels, target_length=100):
    """
    Describe an edge using its COLOR/INTENSITY pattern
    This is what we use for sliding tile puzzles!
    """
    if len(edge_pixels) == 0:
        return np.array([])
    
    # Convert to grayscale if it's color
    if len(edge_pixels.shape) > 1 and edge_pixels.shape[1] == 3:
        # Convert RGB to grayscale
        intensities = 0.299 * edge_pixels[:, 0] + 0.587 * edge_pixels[:, 1] + 0.114 * edge_pixels[:, 2]
    else:
        intensities = edge_pixels
    
    # Resample to fixed length
    x_old = np.linspace(0, 1, len(intensities))
    x_new = np.linspace(0, 1, target_length)
    
    # Interpolate to get consistent descriptor length
    descriptor = np.interp(x_new, x_old, intensities)
    
    # Normalize to [0, 1] range
    if descriptor.max() > descriptor.min():
        descriptor = (descriptor - descriptor.min()) / (descriptor.max() - descriptor.min())
    
    return descriptor

def create_test_sliding_tile_edges():
    """
    Create synthetic edge INTENSITY patterns for testing
    Simulates different color patterns along straight edges
    """
    test_edges = {}
    
    # Edge 1: Smooth gradient (light to dark)
    x = np.linspace(0, 100, 100)
    intensity = np.linspace(1.0, 0.2, 100)  # Smooth gradient
    test_edges['gradient'] = intensity
    
    # Edge 2: High contrast pattern (like an object boundary)
    intensity = 0.3 + 0.7 * (np.sin(8 * np.pi * x / 100) > 0)  # Striped pattern
    test_edges['high_contrast'] = intensity
    
    # Edge 3: Medium pattern (texture)
    intensity = 0.5 + 0.3 * np.sin(4 * np.pi * x / 100)  # Wavy pattern
    test_edges['medium_pattern'] = intensity
    
    # Edge 4: Uniform color (flat area)
    intensity = np.full(100, 0.7)  # Constant color
    test_edges['uniform'] = intensity
    
    return test_edges

def visualize_sliding_tile_edge(pattern, descriptor, edge_name):
    """
    Plot the intensity pattern and its descriptor
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot original intensity pattern along the edge
    ax1.plot(pattern, 'b-', linewidth=2)
    ax1.set_title(f'Original Intensity Pattern\n({edge_name} Edge)')
    ax1.set_xlabel('Position along edge')
    ax1.set_ylabel('Intensity (0-1)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot the normalized descriptor
    ax2.plot(descriptor, 'r-', linewidth=2)
    ax2.set_title('Color Pattern Descriptor')
    ax2.set_xlabel('Position along edge (normalized)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def compare_sliding_tile_edges(desc1, desc2):
    """
    Compare two edge descriptors for sliding tiles
    Low score = good color match
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return float('inf')
    
    # Simple Mean Squared Error
    score = np.mean((desc1 - desc2) ** 2)
    
    # Also try reversed (for opposite edges)
    reversed_score = np.mean((desc1 - desc2[::-1]) ** 2)
    
    return min(score, reversed_score)

def interactive_sliding_tile_demo():
    """
    Demo for SLIDING TILE puzzle edges - based on COLOR PATTERNS
    """
    print("\n" + "🧩" * 10 + " SLIDING TILE DESCRIPTOR DEMO " + "🧩" * 10)
    print("🔍 For SLIDING TILES - we match edges based on COLOR PATTERNS!")
    print("   (Not shapes like jigsaw puzzles)")
    
    # Create test intensity patterns
    test_edges = create_test_sliding_tile_edges()
    
    print(f"\n📊 Created {len(test_edges)} edge INTENSITY patterns:")
    for edge_type in test_edges.keys():
        print(f"   - {edge_type}")
    
    # Ask user how many examples to show
    try:
        max_possible = len(test_edges)
        num_to_show = int(input(f"\nHow many pattern types do you want to see? (1-{max_possible}): "))
        num_to_show = max(1, min(num_to_show, max_possible))
        print(f"✅ Will show {num_to_show} intensity pattern examples")
    except ValueError:
        num_to_show = min(3, len(test_edges))
        print(f"⚠️ Invalid input. Showing {num_to_show} examples by default")
    
    # Show selected examples
    edge_items = list(test_edges.items())
    
    print(f"\n🎨 Intensity Pattern Analysis:")
    print("   Left graph: Original color pattern along edge")
    print("   Right graph: Normalized descriptor for matching")
    
    for i, (edge_type, intensity_pattern) in enumerate(edge_items[:num_to_show]):
        print(f"\n📐 Example {i+1}/{num_to_show}: {edge_type.upper()} pattern")
        
        # The intensity pattern IS our "raw data" - just normalize it
        descriptor = describe_edge_color_pattern(intensity_pattern)
        
        print(f"   Pattern range: [{intensity_pattern.min():.2f} to {intensity_pattern.max():.2f}]")
        print(f"   Descriptor length: {len(descriptor)} points")
        
        # Show visualization
        visualize_sliding_tile_edge(intensity_pattern, descriptor, edge_type)
    
    # Demonstrate matching
    print(f"\n🔗 Pattern Matching Demo:")
    patterns_list = list(test_edges.items())
    
    # Compare a few patterns
    comparisons = [
        (0, 1, "gradient vs high-contrast"),
        (0, 0, "gradient vs itself (should match perfectly)"),
        (2, 3, "medium pattern vs uniform")
    ]
    
    for idx1, idx2, description in comparisons:
        if idx1 < len(patterns_list) and idx2 < len(patterns_list):
            desc1 = describe_edge_color_pattern(patterns_list[idx1][1])
            desc2 = describe_edge_color_pattern(patterns_list[idx2][1])
            score = compare_sliding_tile_edges(desc1, desc2)
            
            match_quality = "PERFECT" if score < 0.01 else "GOOD" if score < 0.1 else "POOR"
            print(f"   {description}: score = {score:.4f} ({match_quality} match)")
    
    print(f"\n✅ Sliding tile descriptor system ready!")
    print("🎯 For real images, we'll extract color patterns from piece edges!")
    print("💡 Matching principle: Similar color patterns = likely neighbors!")

# Actually run the SLIDING TILE descriptor demo
interactive_sliding_tile_demo()