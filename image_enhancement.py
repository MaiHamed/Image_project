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

    # Process each pixel (skip borders)
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

def enhance_image(img):
    enhanced = img.copy()

    if len(enhanced.shape) == 3:  
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)

    lap = cv2.Laplacian(enhanced, cv2.CV_32F)
    lap -= lap.mean()
    sharpened = enhanced - 0.2 * lap
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)


    return sharpened



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
                # Apply selective median filter first
                denoised = selective_median_filter(img, threshold=50)

                # Apply enhancement (contrast + sharpening)
                enhanced = enhance_image(denoised)

                # Save the enhanced image
                output_path = os.path.join(puzzle_output_dir, f"enhanced_{filename}")
                cv2.imwrite(output_path, enhanced)

                # Store examples for preview
                
                examples.append({
                        'original': img,
                        'denoised': denoised,
                        'enhanced': enhanced,
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
        plt.figure(figsize=(15, 6))

        plt.subplot(1,3,1)
        plt.imshow(cv2.cvtColor(example['original'], cv2.COLOR_BGR2RGB)); plt.title("Original")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(cv2.cvtColor(example['denoised'], cv2.COLOR_BGR2RGB)); plt.title("Denoised")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(example['enhanced'], cv2.COLOR_BGR2RGB)); plt.title("Enhanced")
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

#Descriptor demo
#IGNORE

def normalize_descriptor(descriptor, target_length=100):
    #Normalize descriptor to fixed length and scale
    #This makes descriptors comparable regardless of original size/point count
    if len(descriptor) == 0:
        return np.array([])
    
    # Resample to target length using interpolation
    x_old = np.linspace(0, 1, len(descriptor))
    x_new = np.linspace(0, 1, target_length)
    
    # Linear interpolation to get fixed number of points
    normalized = np.interp(x_new, x_old, descriptor)
    
    # Normalize scale to [-1, 1] range
    max_abs = np.max(np.abs(normalized))
    if max_abs > 0:
        normalized = normalized / max_abs
    
    return normalized

def create_test_edges():
    "Create synthetic edge contours for testing"
    test_edges = {}
    
    # Flat edge (straight line)
    x = np.linspace(0, 100, 50)
    y = np.zeros(50)
    test_edges['flat'] = np.column_stack((x, y))
    
    # Tab (outward bulge)
    x = np.linspace(0, 100, 50)
    y = 20 * np.sin(np.pi * x / 100)  # Sine wave bulge
    test_edges['tab'] = np.column_stack((x, y))
    
    # Blank (inward curve)
    x = np.linspace(0, 100, 50)
    y = -20 * np.sin(np.pi * x / 100)  # Negative sine wave
    test_edges['blank'] = np.column_stack((x, y))
    
    return test_edges

def describe_edge(edge_points):
    #Describe an edge using distance from centerline
    #This is exactly what you'll use for real data later!

    # 1. Get endpoints
    p1 = edge_points[0]
    p2 = edge_points[-1]
    
    # 2. Calculate centerline
    centerline_vector = p2 - p1
    centerline_length = np.linalg.norm(centerline_vector)
    
    # 3. Calculate distances for each point
    distances = []
    for point in edge_points:
        # Vector from p1 to current point
        v = point - p1
        # Projection onto centerline
        t = np.dot(v, centerline_vector) / (centerline_length ** 2)
        t = np.clip(t, 0, 1)  # Clamp to segment
        # Closest point on centerline
        closest_point = p1 + t * centerline_vector
        # Signed distance to centerline
        distance = np.linalg.norm(point - closest_point)
        # Determine sign (which side of centerline)
        cross_product = np.cross(centerline_vector, point - p1)
        if cross_product < 0:
            distance = -distance
        distances.append(distance)
    
    # 4. Normalize and resample (you'll implement this)
    normalized_descriptor = normalize_descriptor(distances)
    return normalized_descriptor

def visualize_edge_description(edge_points, descriptor):
    #Plot the original edge and its descriptor
    # This helps you debug and understand what your code is doing
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot original edge
    ax1.plot(edge_points[:, 0], edge_points[:, 1], 'b.-')
    ax1.set_title('Original Edge')
    ax1.axis('equal')
    
    # Plot descriptor
    ax2.plot(descriptor)
    ax2.set_title('Shape Descriptor')
    ax2.set_xlabel('Point index')
    ax2.set_ylabel('Normalized Distance')
    
    plt.show()

def interactive_descriptor_demo():
    """Let user choose how many descriptor examples to see"""
    print("\n" + "🧩" * 10 + " INTERACTIVE DESCRIPTOR DEMO " + "🧩" * 10)
    
    # Create test edges
    test_edges = create_test_edges()
    
    print(f"📊 Created {len(test_edges)} edge types for demonstration")
    print("Edge types: FLAT, TAB (bump), BLANK (hole)")
    
    # Ask user how many examples to show
    try:
        max_possible = len(test_edges)  # This is 3: flat, tab, blank
        num_to_show = int(input(f"\nHow many edge types do you want to visualize? (1-{max_possible}): "))
        num_to_show = max(1, min(num_to_show, max_possible))
        print(f"✅ Will show {num_to_show} edge type examples")
    except ValueError:
        num_to_show = min(2, len(test_edges))  # Default to 2 if input invalid
        print(f"⚠️ Invalid input. Showing {num_to_show} examples by default")
    
    # Show selected examples
    edge_items = list(test_edges.items())
    
    for i, (edge_type, edge_points) in enumerate(edge_items[:num_to_show]):
        print(f"\n📐 Example {i+1}/{num_to_show}: {edge_type.upper()} edge")
        
        # Get descriptor
        descriptor = describe_edge(edge_points)
        
        print(f"   Descriptor pattern: [{descriptor.min():.1f} to {descriptor.max():.1f}]")
        
        # Simple classification
        if descriptor.max() > 0.5:
            classification = "TAB (should fit with a BLANK)"
        elif descriptor.min() < -0.5:
            classification = "BLANK (should fit with a TAB)" 
        else:
            classification = "FLAT EDGE"
        
        print(f"   Classification: {classification}")
        
        # Show visualization
        visualize_edge_description(edge_points, descriptor)
    
    # Summary
    print(f"\n✅ Demonstrated {num_to_show} edge types")
    print("🎯 Ready to process real puzzle pieces when contours are available!")
    print(f"💡 For 110 real images, we'll process ALL of them automatically!")

# Actually run the interactive descriptor demo
interactive_descriptor_demo()