import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ---------------- IMPORT FUNCTIONS ----------------
from functions import (
    selective_median_filter, enhance_image, estimate_noise,
    detect_grid_size, extract_generic_grid_pieces,
    extract_rectangular_edges, describe_edge_color_pattern,
    analyze_all_possible_matches_paper_based,  # NEW: paper-based analysis
    analyze_all_possible_matches_rotation_aware  # For compatibility
)

from visualize import (
    show_examples,
    visualize_generic_grid,
    visualize_comparison_heatmap,
    visualize_matches_with_lines,
    visualize_best_match_pair
)

from paper_algorithms import PaperPuzzleSolver  

def main():
    # 1️⃣ UPLOAD ZIP
    print("\n📦 SELECT PUZZLE ZIP FOLDER")
    Tk().withdraw() 
    zip_file = askopenfilename(title="Select ZIP file", filetypes=[("ZIP files", "*.zip")])

    if not zip_file:
        print("❌ No file selected!")
        return

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

    # 2️⃣ FIND IMAGES
    print("\n🔍 Finding all images...")
    image_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.startswith("._"):
                continue
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append({
                    'full_path': os.path.join(root, file),
                    'filename': file,
                    'directory': os.path.relpath(root, extract_dir)
                })

    if not image_files:
        print("❌ No images found in the extracted folder!")
        return

    # Organize by directory
    images_by_dir = {}
    for img in image_files:
        dir_name = img['directory'] if img['directory'] != '.' else 'root'
        images_by_dir.setdefault(dir_name, []).append(img)

        # Sort images inside each directory numerically if possible
        for dir_name, dir_images in images_by_dir.items():
            images_by_dir[dir_name] = sorted(
                dir_images,
                key=lambda x: int(os.path.splitext(x['filename'])[0])
                if os.path.splitext(x['filename'])[0].isdigit()
                else x['filename']
            )


    print(f"\n📊 Found {len(image_files)} images in {len(images_by_dir)} directories")
    print("="*50)

    # 3️⃣ APPLY FILTERS
    print("\n🔧 APPLYING FILTERS TO ALL IMAGES 🔧")
    output_dir = os.path.join(os.path.dirname(zip_file), "processed_puzzles")
    os.makedirs(output_dir, exist_ok=True)

    examples = []
    total_successful = 0

    for dir_name, dir_images in images_by_dir.items():
        print(f"\n📁 Processing directory: {dir_name} ({len(dir_images)} images)")

        puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
        orig_dir = os.path.join(puzzle_output_dir, "original")
        denoise_dir = os.path.join(puzzle_output_dir, "denoised")
        enhance_dir = os.path.join(puzzle_output_dir, "enhanced")
        edges_dir = os.path.join(puzzle_output_dir, "edges")

        for d in [orig_dir, denoise_dir, enhance_dir, edges_dir]:
            os.makedirs(d, exist_ok=True)

        for i, img_info in enumerate(dir_images, 1):
            img_path = img_info['full_path']
            filename = img_info['filename']

            img = cv2.imread(img_path)
            if img is None:
                print(f"   ❌ Failed to load {filename}")
                continue

            # Estimate noise
            noise_level = estimate_noise(img)

            if noise_level > 150:
                denoise_threshold = 130
                apply_denoise = True
            else:
                denoise_threshold = 0
                apply_denoise = False

            # 2. Apply selective median filter ONCE (if needed)
            if apply_denoise:
                # This filter will now only touch the most severe outliers due to the high threshold (60)
                denoised = selective_median_filter(img, threshold=denoise_threshold)
            else:
                denoised = img.copy()

            # 3. Enhance image (using the gamma value from the previous fix)
            GAMMA_VALUE = 0.9
            enhanced, edges_bw = enhance_image(denoised, apply_denoise, denoise_threshold,GAMMA_VALUE, 0.08, low_threshold=50, high_threshold=150)

            # 4. Save outputs
            cv2.imwrite(os.path.join(orig_dir, filename), img)
            cv2.imwrite(os.path.join(denoise_dir, filename), denoised)
            cv2.imwrite(os.path.join(enhance_dir, filename), enhanced)
            cv2.imwrite(os.path.join(edges_dir, f"edges_{filename}"), edges_bw)

            examples.append({
                'original': img,
                'denoised': denoised,
                'enhanced': enhanced,
                'edges_bw': edges_bw,
                'filename': filename,
                'directory': dir_name
            })
            total_successful += 1


            if i <= 5 or i == len(dir_images):
                print(f"   ✅ [{i}/{len(dir_images)}] {filename}")
            elif i == 6:
                print(f"   ... processing {len(dir_images)-5} more images ...")

    print(f"\n✅ Filters applied to {total_successful} images")
    print("="*50)
    #show examples
    show_examples(examples, images_by_dir, output_dir)

    # 4️⃣ GENERIC GRID CROPPING
    if 'image_files' in locals() and image_files:
        print("\n🔍 Executing Generic Grid Cropping...")
        print("   Method: Auto-detecting 2x2, 4x4, or 8x8 based on folder/filenames")

        total_images = sum(len(v) for v in images_by_dir.values())

        # Ask user how many examples to visualize
        try:
            examples_to_show = int(input(f"\nHow many grid-cropping examples do you want to see? (1–{total_images}): "))
            examples_to_show = max(1, min(examples_to_show, total_images))
            print(f"✅ Will show {examples_to_show} grid examples")
        except ValueError:
            examples_to_show = min(3, total_images)
            print(f"⚠ Invalid input. Showing {examples_to_show} examples by default")

        total_pieces_extracted = 0
        processed_count = 0

        if images_by_dir:
            for dir_name, dir_images in images_by_dir.items():
                print(f"\n📁 Processing directory: {dir_name} ({len(dir_images)} images)")

                # Create directories
                puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
                rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")
                os.makedirs(rectangular_dir, exist_ok=True)

                examples_shown = 0

                for i, img_info in enumerate(dir_images, 1):
                    img_path = img_info['full_path']
                    filename = img_info['filename']

                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"   ❌ Failed to load {filename}")
                        continue

                    # 1. Detect grid size
                    N = detect_grid_size(filename, dir_name, default_n=2)

                    # 2. Extract pieces
                    pieces = extract_generic_grid_pieces(img, N=N)

                    # Save pieces
                    for p_idx, piece in enumerate(pieces):
                        piece_filename = f"piece_{p_idx+1}_{filename}"
                        cv2.imwrite(os.path.join(rectangular_dir, piece_filename), piece)

                    total_pieces_extracted += len(pieces)
                    processed_count += 1

                    # Show example visualizations
                    if examples_shown < examples_to_show:
                        print(f"\n🧩 Visualizing {N}x{N} crop {examples_shown + 1}/{examples_to_show}: {filename}")
                        viz_img = visualize_generic_grid(img, pieces, N, filename)
                        cv2.imwrite(os.path.join(rectangular_dir, f"grid_cut_{filename}"), viz_img)
                        examples_shown += 1

                    # Print progress
                    if i <= 5:
                        print(f"   ✅ [{i}/{len(dir_images)}] {filename} -> {N}x{N} ({len(pieces)} pieces)")
                    elif i == 6:
                        print(f"   ... processing {len(dir_images)-5} more images ...")

            print(f"\n" + "=" * 70)
            print(f"CROPPING COMPLETE: {total_pieces_extracted} pieces extracted from {processed_count} images.")
            print("=" * 70)
        else:
            print("❌ No images found for cropping.")
    
    # 5️⃣ RUN PAPER-BASED COMPARISON ANALYSIS
    if 'images_by_dir' in locals():
        print("\n" + "="*70)
        print("🔍 ANALYZING PIECE COMPATIBILITY - Using Paper's Algorithms")
        print("="*70)

        for dir_name, dir_images in images_by_dir.items():
            puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
            rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")

            if not os.path.exists(rectangular_dir):
                continue

            print(f"\n📁 Analyzing puzzle pieces from: {rectangular_dir}")

            # load piece files
            piece_files = sorted(
                [f for f in os.listdir(rectangular_dir) if f.startswith("piece_") and f.endswith(('.png', '.jpg'))]
            )
            if not piece_files:
                continue

            # group by puzzle id (piece_4_97.jpg → id = 97)
            pieces_by_puzzle = {}
            for p_file in piece_files:
                parts = p_file.split('_')
                if len(parts) >= 3:
                    puzzle_id = parts[2].split('.')[0]
                    pieces_by_puzzle.setdefault(puzzle_id, []).append(p_file)

            # For demo, only process first puzzle
            for puzzle_id, pieces in pieces_by_puzzle.items():
                pieces.sort(key=lambda x: int(x.split('_')[1]))
                
                # Load piece images
                all_piece_images = []
                for piece_file in pieces:
                    img = cv2.imread(os.path.join(rectangular_dir, piece_file))
                    if img is None:
                        print(f"   ❌ Failed to load: {piece_file}")
                    else:
                        all_piece_images.append(img)

                if len(all_piece_images) < 4:  # Need at least 2x2
                    print(f"   ⚠️ Not enough pieces ({len(all_piece_images)})")
                    continue

                # detect puzzle grid size
                num_pieces = len(all_piece_images)
                N = int(np.sqrt(num_pieces))
                if N * N != num_pieces:
                    print(f"   ⚠️ Puzzle skipped: expected {N*N} pieces, found {num_pieces}")
                    continue

                print(f"\n--- 🧩 PAPER SOLVER: Puzzle {puzzle_id} ({N}x{N}, {num_pieces} pieces) ---")
                print(f"   Using prediction-based compatibility with p=0.3, q=1/16")

                # Run paper-based analysis
                all_comparisons, all_piece_rotations, final_grid, best_buddies = \
                    analyze_all_possible_matches_paper_based(all_piece_images, pieces, N)

                # heatmaps (using original visualization)
                print("\n   📈 Generating compatibility heatmaps...")
                horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                    all_comparisons[:N*N*4], pieces, N, f"Puzzle_{puzzle_id}"
                )

                # visualize best matches
                print("\n   👀 Visualizing best matches...")
                visualize_matches_with_lines(all_piece_images, all_comparisons, top_n=5)

                # Assemble puzzle using paper's grid
                print("\n   🧩 Assembling puzzle from paper solver...")
                
                # Build assembled image from final_grid
                piece_height, piece_width = all_piece_images[0].shape[:2]
                assembled_height = N * piece_height
                assembled_width = N * piece_width
                assembled = np.zeros((assembled_height, assembled_width, 3), dtype=np.uint8)
                
                for r in range(N):
                    for c in range(N):
                        piece_idx = final_grid[r][c]
                        if piece_idx is not None:
                            y_start = r * piece_height
                            x_start = c * piece_width
                            assembled[y_start:y_start+piece_height, 
                                     x_start:x_start+piece_width] = all_piece_images[piece_idx]
                
                # Display assembly
                plt.figure(figsize=(8, 8))
                plt.imshow(cv2.cvtColor(assembled, cv2.COLOR_BGR2RGB))
                plt.title(f"Paper Solver Assembly: Puzzle {puzzle_id} ({N}x{N})", 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
                # Save assembled image
                save_path = os.path.join(puzzle_output_dir, f"paper_assembled_{puzzle_id}.jpg")
                cv2.imwrite(save_path, assembled)
                print(f"   💾 Saved to: {save_path}")
                
                # Save best buddies info
                bb_file = os.path.join(puzzle_output_dir, f"best_buddies_{puzzle_id}.txt")
                with open(bb_file, 'w') as f:
                    f.write(f"Best Buddies for Puzzle {puzzle_id} ({N}x{N})\n")
                    f.write("="*50 + "\n")
                    for i, bb in enumerate(best_buddies[:20]):
                        f.write(f"{i+1:2d}. P{bb['piece1']+1} ↔ P{bb['piece2']+1} ")
                        f.write(f"({bb['relation']}, Score: {bb['score']:.6f})\n")
                
                print(f"   📝 Best buddies saved to: {bb_file}")
                
                # only first puzzle for demo
                break

            # only first directory for demo
            break

        print("\n" + "=" * 70)
        print("✅ PAPER SOLVER COMPLETE!")
        print("✅ Used prediction-based compatibility (p=0.3, q=1/16)")
        print("✅ Found best buddies and assembled puzzle")
        print("=" * 70)

    else:
        print("❌ Previous steps not completed.")
        
# In the paper-based analysis section of run.py:
    try:
        # Run paper-based analysis
        all_comparisons, all_piece_rotations, final_grid, best_buddies = \
            analyze_all_possible_matches_paper_based(all_piece_images, pieces, N)
        
        # Check if results are valid
        if not all_comparisons:
            print("   ⚠️ No comparisons generated, using fallback method")
            # Fallback to rotation-aware method
            all_comparisons, all_piece_rotations = \
                analyze_all_possible_matches_rotation_aware(all_piece_images, pieces, N)
            final_grid = None
            best_buddies = []
            
    except Exception as e:
        print(f"   ❌ Paper solver failed: {e}")
        print("   🔄 Falling back to rotation-aware method")
        
        # Fallback to the old method
        all_comparisons, all_piece_rotations = \
            analyze_all_possible_matches_rotation_aware(all_piece_images, pieces, N)
        final_grid = None
        best_buddies = []
    # Assemble puzzle using paper's grid (if available)
    if final_grid is not None:
        print("\n   🧩 Assembling puzzle from paper solver...")
        
        # Build assembled image from final_grid
        piece_height, piece_width = all_piece_images[0].shape[:2]
        assembled_height = N * piece_height
        assembled_width = N * piece_width
        assembled = np.zeros((assembled_height, assembled_width, 3), dtype=np.uint8)
        
        for r in range(N):
            for c in range(N):
                piece_idx = final_grid[r][c]
                if piece_idx is not None:
                    y_start = r * piece_height
                    x_start = c * piece_width
                    assembled[y_start:y_start+piece_height, 
                            x_start:x_start+piece_width] = all_piece_images[piece_idx]
        
        # Display assembly
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(assembled, cv2.COLOR_BGR2RGB))
        plt.title(f"Paper Solver Assembly: Puzzle {puzzle_id} ({N}x{N})", 
                fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save assembled image
        save_path = os.path.join(puzzle_output_dir, f"paper_assembled_{puzzle_id}.jpg")
        cv2.imwrite(save_path, assembled)
        print(f"   💾 Saved to: {save_path}")
    else:
        print("   ⚠️ No grid assembly available from paper solver")        
            
if __name__ == "__main__":
        main()