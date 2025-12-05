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
    analyze_all_possible_matches_rotation_aware
)

from visualize import (
    show_examples,
    visualize_generic_grid,
    visualize_comparison_heatmap,
    visualize_best_match_pair
)


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
    
    # 5️⃣ RUN COMPARISON ANALYSIS (Rotation-Aware)
    if 'images_by_dir' in locals():
        print("🔍 ANALYZING PIECE COMPATIBILITY - Rotation-aware matching enabled...")

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

                # sort by middle number (piece_4_97.jpg → sort by 4)
                pieces.sort(key=lambda x: int(x.split('_')[1]))
                print(f"\n--- 🧩 ANALYSIS: Puzzle {puzzle_id} ({len(pieces)} pieces) ---")
                print(f"   Pieces in order: {pieces}")

                # detect puzzle grid size
                num_pieces = len(pieces)
                N = int(np.sqrt(num_pieces))
                if N * N != num_pieces:
                    print(f"   ⚠️ Puzzle skipped: expected {N*N} pieces, found {num_pieces}")
                    continue

                # load images
                all_piece_images = []
                for piece_file in pieces:
                    img = cv2.imread(os.path.join(rectangular_dir, piece_file))
                    if img is None:
                        print(f"   ❌ Failed to load: {piece_file}")
                    all_piece_images.append(img)

                if len(all_piece_images) < 2:
                    print("   ⚠️ Not enough pieces loaded for analysis")
                    continue

                print("   🌀 Running rotation-aware edge comparison...")
                all_comparisons, all_piece_rotations = analyze_all_possible_matches_rotation_aware(
                    all_piece_images, pieces, N
                )

                # heatmaps
                print("\n   📈 Generating compatibility heatmaps...")
                horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                    all_comparisons, pieces, N, f"Puzzle_{puzzle_id}"
                )

                # visualize first 3 images in folder as demo
                print("\n   👀 Visualizing demo matches for FIRST 3 PIECES...")
                demo_indices = list(range(min(3, len(all_piece_images))))

                for idx in demo_indices:
                    # find best match involving this piece
                    matches_for_piece = [m for m in all_comparisons if m['piece1'] == idx]
                    if not matches_for_piece:
                        continue

                    best = sorted(matches_for_piece, key=lambda x: x['score'])[0]
                    p1_idx, p2_idx = best['piece1'], best['piece2']
                    rot_angle = best.get('rotation_of_piece2', 0)

                    desc1 = all_piece_rotations[p1_idx][0]['descriptors'][best['edge1']]
                    desc2 = all_piece_rotations[p2_idx][rot_angle]['descriptors'][best['edge2']]
                    desc2_plot = desc2[::-1] if best['edge1'] == 'right' and best['edge2'] == 'left' else desc2

                    print(f"   🎯 Demo: Piece {p1_idx+1} best match → Piece {p2_idx+1} "
                        f"(Score: {best['score']:.4f}, Rot={rot_angle}°)")

                    visualize_best_match_pair(
                        all_piece_rotations[p1_idx][0]['image'],
                        all_piece_rotations[p2_idx][rot_angle]['image'],
                        desc1,
                        desc2_plot,
                        best['score'],
                        best
                    )

                # only first puzzle for demo
                break

            # only first directory for demo
            break

        print("\n" + "=" * 70)
        print("COMPARISON ANALYSIS COMPLETE!")
        print("✅ Rotation-aware matching used")
        print("✅ Heatmaps generated")
        print("✅ Demo visualizations for first three pieces shown")
        print("=" * 70)

    else:
        print("❌ Previous steps not completed.")

        # 6️⃣ ASSEMBLE PUZZLE - FIXED VERSION
    if 'images_by_dir' in locals():
        print("\n" + "="*70)
        print("🧩 ASSEMBLING PUZZLE FROM MATCHES")
        print("="*70)
        
        # Import the new matching functions
        try:
            from Matching import (
                assemble_puzzle_from_matches,
                visualize_matches_with_lines,
                visualize_assembly
            )
            print("✅ Successfully imported matching functions")
        except ImportError as e:
            print(f"❌ Failed to import matching functions: {e}")
            return
        
        for dir_name, dir_images in images_by_dir.items():
            puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
            rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")
            
            if not os.path.exists(rectangular_dir):
                print(f"⚠️ No rectangular pieces found in {rectangular_dir}")
                continue
            
            piece_files = sorted(
                [f for f in os.listdir(rectangular_dir) if f.startswith("piece_") and f.endswith(('.png', '.jpg'))]
            )
            
            if not piece_files:
                print(f"⚠️ No piece files found in {rectangular_dir}")
                continue
            
            # Group by puzzle (piece_4_97.jpg → puzzle id = 97)
            pieces_by_puzzle = {}
            for p_file in piece_files:
                parts = p_file.split('_')
                if len(parts) >= 3:
                    puzzle_id = parts[2].split('.')[0]
                    pieces_by_puzzle.setdefault(puzzle_id, []).append(p_file)
            
            # Process each puzzle
            for puzzle_id, pieces in pieces_by_puzzle.items():
                pieces.sort(key=lambda x: int(x.split('_')[1]))
                
                # Load piece images
                piece_images = []
                valid_pieces = []
                for piece_file in pieces:
                    img_path = os.path.join(rectangular_dir, piece_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        piece_images.append(img)
                        valid_pieces.append(piece_file)
                    else:
                        print(f"⚠️ Failed to load: {piece_file}")
                
                if len(piece_images) < 4:  # Need at least 2x2
                    print(f"⚠️ Puzzle {puzzle_id}: Only {len(piece_images)} pieces, need at least 4")
                    continue
                print(f"   Loaded {len(piece_images)} pieces")
                print(f"   Piece sizes: {[img.shape for img in piece_images]}")
                print(f"   Number of comparisons generated: {len(all_comparisons)}")
                
                # Determine grid size
                num_pieces = len(piece_images)
                N = int(np.sqrt(num_pieces))
                if N * N != num_pieces:
                    print(f"⚠️ Puzzle {puzzle_id}: {num_pieces} pieces not a perfect square")
                    continue
                
                print(f"\n" + "-"*60)
                print(f"🧩 ASSEMBLING: Puzzle {puzzle_id} ({N}x{N}, {num_pieces} pieces)")
                print(f"   Pieces: {valid_pieces}")
                print("-"*60)
                
                # Get matches using the original analysis function
                print("🔍 Running edge comparison analysis...")
                all_comparisons, all_piece_rotations = analyze_all_possible_matches_rotation_aware(
                    piece_images, valid_pieces, N
                )
                
                if not all_comparisons:
                    print("❌ No matches found!")
                    continue
                
                # Visualize top matches with lines
                print("\n📊 Visualizing best matches with connecting lines...")
                visualize_matches_with_lines(piece_images, all_comparisons, top_n=10)
                
                # Assemble puzzle
                print("\n🔧 Attempting to assemble puzzle...")
                assembled_grid = assemble_puzzle_from_matches(all_comparisons, piece_images, N)
                
                if assembled_grid is None:
                    print("❌ Failed to assemble puzzle")
                    continue
                
                # Visualize assembly
                print("\n🎨 Displaying assembled puzzle...")
                assembled_img = visualize_assembly(assembled_grid, piece_images, N, 
                                                   f"Assembled Puzzle {puzzle_id} ({N}x{N})")
                
                # Save assembled image
                if assembled_img is not None:
                    save_path = os.path.join(puzzle_output_dir, f"assembled_{puzzle_id}.jpg")
                    cv2.imwrite(save_path, assembled_img)
                    print(f"💾 Saved assembled image to: {save_path}")
                
                # Save match results to file
                matches_file = os.path.join(puzzle_output_dir, f"matches_{puzzle_id}.txt")
                with open(matches_file, 'w') as f:
                    f.write(f"Match Results for Puzzle {puzzle_id} ({N}x{N})\n")
                    f.write("="*50 + "\n")
                    for i, match in enumerate(sorted(all_comparisons, key=lambda x: x['score'])[:20]):
                        f.write(f"{i+1:2d}. P{match['piece1']+1} {match['edge1']} <-> P{match['piece2']+1} {match['edge2']}\n")
                        f.write(f"    Rotation: {match['rotation_of_piece2']}°, Score: {match['score']:.6f}\n")
                        f.write("-"*40 + "\n")
                
                print(f"📝 Match details saved to: {matches_file}")
                
                # Only process first puzzle for demo
                break
            
            # Only process first directory for demo
            break
        
        print("\n" + "="*70)
        print("✅ ASSEMBLY COMPLETE!")
        print("="*70)
        
if __name__ == "__main__":
    main()
