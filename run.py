import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from functions import analyze_all_possible_matches_paper_based, assemble_grid_from_pieces, choose_best_orientation_hybrid, detect_grid_size, enhance_image, estimate_noise, evaluate_corner_compatibility_direct, evaluate_grid_compatibility_direct, extract_generic_grid_pieces, reassemble_grid_all_orientations, selective_median_filter, visualize_piece_relationships
from visualize import show_examples, visualize_generic_grid, visualize_orientation_comparison

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

    # Sort images inside each directory
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
            apply_denoise = noise_level > 150
            denoise_threshold = 130 if apply_denoise else 0

            # Apply denoising if needed
            if apply_denoise:
                denoised = selective_median_filter(img, threshold=denoise_threshold)
            else:
                denoised = img.copy()

            # Enhance image
            GAMMA_VALUE = 0.9
            enhanced, edges_bw = enhance_image(denoised, apply_denoise, denoise_threshold, 
                                             GAMMA_VALUE, 0.08, 50, 150)

            # Save outputs
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
    
    # Show examples
    show_examples(examples, images_by_dir, output_dir)

    # 4️⃣ GENERIC GRID CROPPING
    if images_by_dir:
        print("\n🔍 Executing Generic Grid Cropping...")
        print("   Method: Auto-detecting 2x2, 4x4, or 8x8 based on folder/filenames")

        total_images = sum(len(v) for v in images_by_dir.values())
        
        # Ask user how many examples to visualize
        try:
            examples_to_show = int(input(f"\nHow many grid-cropping examples do you want to see? (1-{total_images}): "))
            examples_to_show = max(1, min(examples_to_show, total_images))
            print(f"✅ Will show {examples_to_show} grid examples")
        except ValueError:
            examples_to_show = min(3, total_images)
            print(f"⚠ Invalid input. Showing {examples_to_show} examples by default")

        total_pieces_extracted = 0

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
                if img is None:
                    print(f"   ❌ Failed to load {filename}")
                    continue

                # Detect grid size
                N = detect_grid_size(filename, dir_name, default_n=2)

                # Extract pieces
                pieces = extract_generic_grid_pieces(img, N=N)

                # Save pieces
                for p_idx, piece in enumerate(pieces):
                    piece_filename = f"piece_{p_idx+1}_{filename}"
                    cv2.imwrite(os.path.join(rectangular_dir, piece_filename), piece)

                total_pieces_extracted += len(pieces)

                # Show example visualizations
                if examples_shown < examples_to_show:
                    print(f"\n🧩 Visualizing {N}x{N} crop {examples_shown + 1}/{examples_to_show}: {filename}")
                    viz_img = visualize_generic_grid(img, pieces, N, filename)
                    cv2.imwrite(os.path.join(rectangular_dir, f"grid_cut_{filename}"), viz_img)
                    examples_shown += 1

                if i <= 5:
                    print(f"   ✅ [{i}/{len(dir_images)}] {filename} -> {N}x{N} ({len(pieces)} pieces)")
                elif i == 6:
                    print(f"   ... processing {len(dir_images)-5} more images ...")

        print(f"\n" + "=" * 70)
        print(f"CROPPING COMPLETE: {total_pieces_extracted} pieces extracted")
        print("=" * 70)

    # 5️⃣ RUN PAPER-BASED ANALYSIS
    if 'images_by_dir' in locals():
        print("\n" + "="*70)
        print("🔍 PAPER ALGORITHM - ORIENTATION ANALYSIS")
        print("="*70)

        for dir_name, dir_images in images_by_dir.items():
            puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
            rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")

            if not os.path.exists(rectangular_dir):
                continue

            print(f"\n📁 Analyzing puzzle pieces from: {rectangular_dir}")

            # Load piece files
            piece_files = sorted(
                [f for f in os.listdir(rectangular_dir) if f.startswith("piece_") and f.endswith(('.png', '.jpg'))]
            )
            if not piece_files:
                continue

            # Group by puzzle id
            pieces_by_puzzle = {}
            for p_file in piece_files:
                parts = p_file.split('_')
                if len(parts) >= 3:
                    puzzle_id = parts[2].split('.')[0]
                    pieces_by_puzzle.setdefault(puzzle_id, []).append(p_file)

            # Ask user how many puzzles to assemble
            total_puzzles = len(pieces_by_puzzle)
            print(f"\n📊 Found {total_puzzles} complete puzzles")
            print("Puzzle IDs found:", list(pieces_by_puzzle.keys())[:min(10, total_puzzles)])
            if total_puzzles > 10:
                print(f"... and {total_puzzles - 10} more")
            
            try:
                puzzles_to_assemble = int(input(f"\nHow many puzzles do you want to assemble? (1-{total_puzzles}): "))
                puzzles_to_assemble = max(1, min(puzzles_to_assemble, total_puzzles))
                print(f"✅ Will assemble {puzzles_to_assemble} puzzles")
            except ValueError:
                puzzles_to_assemble = min(3, total_puzzles)
                print(f"⚠ Invalid input. Assembling {puzzles_to_assemble} puzzles by default")

            # Sort puzzle IDs for consistent processing
            sorted_puzzle_ids = sorted(pieces_by_puzzle.keys(), 
                                     key=lambda x: int(x) if x.isdigit() else x)
            
            processed_puzzles = 0
            assembly_results = []

            for puzzle_id in sorted_puzzle_ids:
                if processed_puzzles >= puzzles_to_assemble:
                    break
                    
                pieces = sorted(pieces_by_puzzle[puzzle_id], key=lambda x: int(x.split('_')[1]))
                
                # Load piece images
                all_piece_images = []
                for piece_file in pieces:
                    img = cv2.imread(os.path.join(rectangular_dir, piece_file))
                    if img is None:
                        print(f"   ❌ Failed to load: {piece_file}")
                    else:
                        all_piece_images.append(img)

                if len(all_piece_images) < 4:  # Need at least 2x2
                    print(f"   ⚠️ Puzzle {puzzle_id}: Not enough pieces ({len(all_piece_images)})")
                    continue

                # Detect puzzle grid size
                num_pieces = len(all_piece_images)
                N = int(np.sqrt(num_pieces))
                if N * N != num_pieces:
                    print(f"   ⚠️ Puzzle {puzzle_id}: Expected {N*N} pieces, found {num_pieces}")
                    continue

                processed_puzzles += 1
                print(f"\n" + "="*60)
                print(f"🧩 ASSEMBLING PUZZLE {puzzle_id} ({N}x{N}, {num_pieces} pieces)")
                print("="*60)
                
                # ========== PAPER ALGORITHM - ORIENTATION ANALYSIS ==========
                print(f"\n📘 PAPER ALGORITHM - Testing All Orientations")
                
                # Initialize variables
                assembled_paper = None
                final_grid_paper = None
                best_orientation = None
                corner_score = 0
                grid_score = 0
                combined_score = 0
                
                try:
                    all_comparisons_paper, all_piece_rotations_paper, final_grid_paper, best_buddies_paper = \
                        analyze_all_possible_matches_paper_based(all_piece_images, pieces, N)
                    
                    if final_grid_paper is not None:
                        # Test all 4 orientations
                        orientations = reassemble_grid_all_orientations(final_grid_paper, N)
                        
                        print("   🎯 Original grid arrangement:")
                        visualize_piece_relationships(all_piece_images, final_grid_paper, N, 
                                                    f"Puzzle {puzzle_id}: Original Paper Grid ({N}x{N})")
                        
                        # Use hybrid evaluation
                        print("   📊 Evaluating all orientations...")
                        best_orientation, corner_score, grid_score = choose_best_orientation_hybrid(
                            all_piece_images, orientations, N, all_comparisons_paper
                        )
                        
                        combined_score = 0.3 * corner_score + 0.7 * grid_score
                        
                        # Visualize all orientations
                        visualize_orientation_comparison(all_piece_images, orientations, N, puzzle_id)
                        
                        if best_orientation:
                            # Show why this orientation is the best
                            print(f"\n   📊 Selected Orientation: {best_orientation['name']}")
                            print(f"   📈 Scores:")
                            print(f"      - Corner compatibility: {corner_score:.3f}")
                            print(f"      - Grid compatibility: {grid_score:.3f}")
                            print(f"      - Combined score: {combined_score:.3f}")
                            
                            # Show all scores for comparison
                            print(f"   📊 All orientation scores:")
                            for orientation in orientations:
                                c_score = evaluate_corner_compatibility_direct(all_piece_images, orientation['grid'], N)
                                g_score = evaluate_grid_compatibility_direct(all_piece_images, orientation['grid'], N)
                                o_combined = 0.3 * c_score + 0.7 * g_score
                                
                                if orientation['name'] == best_orientation['name']:
                                    print(f"      → {orientation['name']}: {o_combined:.3f} (SELECTED)")
                                else:
                                    diff = combined_score - o_combined
                                    print(f"      - {orientation['name']}: {o_combined:.3f} ({'+' if diff > 0 else ''}{diff:.3f})")
                            
                            # Assemble with the best orientation
                            assembled_paper = assemble_grid_from_pieces(
                                all_piece_images, 
                                best_orientation['grid'], 
                                N=N
                            )
                            
                            # Visualize the final assembly
                            if assembled_paper is not None:
                                plt.figure(figsize=(8, 8))
                                plt.imshow(cv2.cvtColor(assembled_paper, cv2.COLOR_BGR2RGB))
                                
                                # Color code based on score
                                if combined_score > 0.7:
                                    color = "green"
                                    quality = "Excellent"
                                elif combined_score > 0.5:
                                    color = "orange"
                                    quality = "Good"
                                elif combined_score > 0.3:
                                    color = "yellow"
                                    quality = "Fair"
                                else:
                                    color = "red"
                                    quality = "Poor"
                                
                                plt.title(f"Puzzle {puzzle_id}: Final Assembly ({N}x{N})\nOrientation: {best_orientation['name']}\nQuality: {quality} ({combined_score:.3f})", 
                                         fontsize=14, fontweight='bold', color=color)
                                plt.axis('off')
                                
                                # Add score annotations
                                plt.figtext(0.5, 0.02, 
                                           f"Corner: {corner_score:.3f} | Grid: {grid_score:.3f} | Combined: {combined_score:.3f}",
                                           ha='center', fontsize=10, 
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                                
                                plt.tight_layout()
                                plt.show()
                            
                            # Save paper assembly
                            paper_save_path = os.path.join(puzzle_output_dir, f"paper_assembled_{puzzle_id}_{best_orientation['name'].replace('→', '_')}.jpg")
                            if assembled_paper is not None:
                                cv2.imwrite(paper_save_path, assembled_paper)
                                print(f"   💾 Saved paper assembly to: {paper_save_path}")
                                
                            # Save all orientations for reference
                            for orientation in orientations:
                                assembled = assemble_grid_from_pieces(all_piece_images, orientation['grid'], N=N)
                                if assembled is not None:
                                    orient_path = os.path.join(puzzle_output_dir, f"paper_{puzzle_id}_{orientation['name'].replace('→', '_')}.jpg")
                                    cv2.imwrite(orient_path, assembled)
                        else:
                            assembled_paper = None
                            print("   ⚠️ Could not select best orientation")
                    else:
                        assembled_paper = None
                        print("   ⚠️ No grid assembly from paper solver")
                        
                except Exception as e:
                    print(f"   ❌ Paper solver failed: {e}")
                    import traceback
                    traceback.print_exc()
                    assembled_paper = None
                
                # ========== SAVE RESULTS SUMMARY ==========
                results_file = os.path.join(puzzle_output_dir, f"results_summary_{puzzle_id}.txt")
                with open(results_file, 'w') as f:
                    f.write(f"PUZZLE ASSEMBLY RESULTS - Puzzle {puzzle_id} ({N}x{N})\n")
                    f.write("="*60 + "\n\n")
                    
                    f.write("PAPER ALGORITHM - ORIENTATION ANALYSIS:\n")
                    if best_orientation:
                        f.write(f"  • Selected orientation: {best_orientation['name']}\n")
                        f.write(f"  • Description: {best_orientation['description']}\n")
                        f.write(f"  • Corner compatibility score: {corner_score:.3f}\n")
                        f.write(f"  • Grid compatibility score: {grid_score:.3f}\n")
                        f.write(f"  • Combined score: {combined_score:.3f}\n")
                        f.write(f"  • All orientations tested:\n")
                        for orientation in orientations:
                            c_score = evaluate_corner_compatibility_direct(all_piece_images, orientation['grid'], N)
                            g_score = evaluate_grid_compatibility_direct(all_piece_images, orientation['grid'], N)
                            o_combined = 0.3 * c_score + 0.7 * g_score
                            f.write(f"    - {orientation['name']}: Corner={c_score:.3f}, Grid={g_score:.3f}, Combined={o_combined:.3f}\n")
                    
                    if assembled_paper is not None:
                        f.write(f"\n3. FINAL ASSEMBLY:\n")
                        f.write(f"   Grid (selected): {best_orientation['grid']}\n")
                        f.write(f"   Saved to: {paper_save_path}\n")
                    
                print(f"   📝 Results summary saved to: {results_file}")
                
                # Store assembly results
                assembly_results.append({
                    'puzzle_id': puzzle_id,
                    'grid_size': N,
                    'num_pieces': num_pieces,
                    'paper_assembly': assembled_paper is not None,
                    'paper_orientation': best_orientation['name'] if assembled_paper is not None else None,
                    'paper_score': combined_score if assembled_paper is not None else 0
                })
                
                print(f"\n✅ Puzzle {puzzle_id} assembly completed [{processed_puzzles}/{puzzles_to_assemble}]")
            
            # Show final summary
            print("\n" + "=" * 80)
            print("📊 ASSEMBLY SUMMARY")
            print("=" * 80)
            
            if assembly_results:
                for result in assembly_results:
                    status = '✓' if result['paper_assembly'] else '✗'
                    print(f"🧩 Puzzle {result['puzzle_id']} ({result['grid_size']}x{result['grid_size']}, {result['num_pieces']} pieces):")
                    if result['paper_assembly']:
                        print(f"   Paper Algorithm: {status} ({result['paper_orientation']}, score: {result['paper_score']:.3f})")
                    else:
                        print(f"   Paper Algorithm: {status}")
                    print()
            else:
                print("No puzzles were successfully assembled.")
            
            print(f"✅ Total puzzles assembled: {processed_puzzles}/{puzzles_to_assemble}")
            print("=" * 80)

if __name__ == "__main__":
    main()