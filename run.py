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
    selective_median_filter, enhance_image,
    detect_grid_size, extract_generic_grid_pieces,
    extract_rectangular_edges, describe_edge_color_pattern,
    analyze_all_possible_matches
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

            denoised = selective_median_filter(img, threshold=50)
            enhanced, edges_bw = enhance_image(denoised, low_threshold=50, high_threshold=150)

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


        examples_to_show = 2  # number of visualizations
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
    
    # 5️⃣ RUN COMPARISON ANALYSIS
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

            # Group pieces by puzzle correctly
            pieces_by_puzzle = {}
            for p_file in piece_files:
                parts = p_file.split('_')
                if len(parts) >= 3:
                    puzzle_id = parts[2].split('.')[0]
                    pieces_by_puzzle.setdefault(puzzle_id, []).append(p_file)
            
            for puzzle_id, pieces in pieces_by_puzzle.items():
                pieces.sort(key=lambda x: int(x.split('_')[1]))  # sort by piece number
                
                print(f"\n--- 🧩 ANALYSIS: Puzzle {puzzle_id} ({len(pieces)} pieces) ---")
                print(f"   Pieces in order: {pieces}")
                
                # Detect grid size
                num_pieces = len(pieces)
                if num_pieces == 4:
                    N = 2
                elif num_pieces == 16:
                    N = 4
                elif num_pieces == 64:
                    N = 8
                else:
                    N = int(np.sqrt(num_pieces))
                    print(f"   ⚠️ Unusual piece count: {num_pieces}, assuming {N}x{N}")
                
                total_pieces = N * N
                if len(pieces) != total_pieces:
                    print(f"   ⚠️ Skipping: {len(pieces)} pieces, expected {total_pieces} for {N}x{N}")
                    continue
                
                # Load pieces and extract descriptors
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
                
                # Show heatmap
                print(f"\n   📈 Generating compatibility heatmaps...")
                horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                    all_comparisons, pieces, N, f"Puzzle_{puzzle_id}"
                )
                
                # Show top matches
                print(f"\n   👀 Visualizing best match examples...")
                best_matches = sorted(all_comparisons, key=lambda x: x['score'])[:3]
                
                for match in best_matches:
                    piece1_idx, piece2_idx = match['piece1'], match['piece2']
                    if piece1_idx < len(all_piece_images) and piece2_idx < len(all_piece_images):
                        print(f"   🎯 Showing: {match['label']} (Score: {match['score']:.4f})")
                        visualize_best_match_pair(
                            all_piece_images[piece1_idx],
                            all_piece_images[piece2_idx],
                            all_piece_data[piece1_idx][match['edge1']],
                            all_piece_data[piece2_idx][match['edge2']],
                            match['score'],
                            match
                        )
                
                # Only first puzzle for demo
                break
            
            # Only first directory for demo
            break
                
        print(f"\n" + "=" * 70)
        print("COMPARISON ANALYSIS COMPLETE!")
        print("✅ Compared ALL pieces against ALL other pieces") 
        print("✅ Showed compatibility heatmaps")
        print("✅ Displayed best matches visually")
        print("=" * 70)
    else:
        print("❌ Previous steps not completed.")

        
if __name__ == "__main__":
    main()
