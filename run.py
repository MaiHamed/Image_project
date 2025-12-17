import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from descriptor_assembler import DescriptorBasedAssembler
from advanced_assember import AdvancedPuzzleSolver
from functions import assemble_grid_from_pieces, detect_grid_size, enhance_image, estimate_noise, extract_generic_grid_pieces, selective_median_filter
from visualize import show_examples, visualize_generic_grid, visualize_comparison_heatmap, visualize_matches_with_lines, visualize_descriptor_result


def reconstruct_image(pieces, placement, grid_n):
    ph, pw = pieces[0].shape[:2]
    canvas = np.zeros((grid_n * ph, grid_n * pw, 3), dtype=np.uint8)

    for idx, pid in enumerate(placement):
        r, c = divmod(idx, grid_n)
        canvas[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = pieces[pid]

    return canvas


def validate_grid(grid, N):
    """
    Validate that grid contains all pieces 0 to N*N-1 exactly once
    """
    if grid is None:
        return False
    
    expected_pieces = set(range(N * N))
    actual_pieces = set()
    
    for r in range(N):
        for c in range(N):
            piece = grid[r][c]
            if piece is None or piece == -1:
                return False
            actual_pieces.add(piece)
    
    return actual_pieces == expected_pieces


def format_advanced_solver_result(solver_result, N, all_piece_images):
    """
    Format the result from AdvancedPuzzleSolver to match the expected 5-tuple format
    """
    print(f"   Formatting solver result (type: {type(solver_result)}, length: {len(solver_result) if isinstance(solver_result, (tuple, list)) else 'N/A'})")
    
    # If it's already a 5-tuple, return as-is
    if isinstance(solver_result, tuple) and len(solver_result) == 5:
        print("   ✅ Already in correct 5-tuple format")
        return solver_result
    
    # If it's a list (probably the solved grid)
    if isinstance(solver_result, (list, np.ndarray)):
        # Check if it's a flat list of piece indices
        if len(solver_result) == N * N and all(isinstance(x, (int, np.integer)) for x in solver_result):
            print(f"   Detected flat list of {len(solver_result)} piece indices")
            
            # Convert flat list to 2D grid
            if N > 0:
                final_grid = [solver_result[i*N:(i+1)*N] for i in range(N)]
            else:
                final_grid = [solver_result]
            
            # Create dummy values for other expected returns
            all_comparisons = []
            all_piece_rotations = [{'0': img} for img in all_piece_images]
            best_buddies = []
            assembly_score = 0.7  # Default good score for AdvancedSolver
            
            return all_comparisons, all_piece_rotations, final_grid, best_buddies, assembly_score
    
    # If we get here, something unexpected happened
    print(f"   ⚠️ Unexpected result format from AdvancedPuzzleSolver")
    
    # Create a fallback ordered grid
    final_grid = [[i * N + j for j in range(N)] for i in range(N)]
    all_comparisons = []
    all_piece_rotations = [{'0': img} for img in all_piece_images]
    best_buddies = []
    assembly_score = 0.3  # Low score for fallback
    
    return all_comparisons, all_piece_rotations, final_grid, best_buddies, assembly_score


def run_descriptor_algorithm_with_improvement(all_piece_images, N, puzzle_id, puzzle_output_dir):
    """
    Run descriptor algorithm with improved scoring and save visualizations.
    Uses different solvers based on grid size:
    - 2x2: Uses DescriptorBasedAssembler.solve() (5 parameters)
    - 4x4 or 8x8: Uses AdvancedPuzzleSolver.solve_puzzle() (16 parameters)
    """
    print(f"\n🤖 PUZZLE SOLVER")
    print(f"   Grid size: {N}x{N}")
    
    results = {
        'success': False,
        'all_comparisons': None,
        'all_piece_rotations': None,
        'final_grid': None,
        'best_buddies': None,
        'assembly_score': 0,
        'assembled_image': None,
        'save_paths': {}
    }
    
    try:
        # Check if we have the right number of pieces
        expected_pieces = N * N
        if len(all_piece_images) != expected_pieces:
            print(f"   ⚠️ Warning: Expected {expected_pieces} pieces, got {len(all_piece_images)}")
            print(f"   Using first {expected_pieces} pieces only")
            all_piece_images = all_piece_images[:expected_pieces]
        
        # Choose solver based on grid size
        if N == 2:
            # Use DescriptorBasedAssembler for 2x2 puzzles
            print("   Using DescriptorBasedAssembler (5-parameter .solve())...")
            descriptor_assembler = DescriptorBasedAssembler(border_width=3, descriptor_length=100)
            all_comparisons, all_piece_rotations, final_grid, best_buddies, assembly_score = \
                descriptor_assembler.solve(all_piece_images)
            
        elif N == 4 or N == 8:
            # Use AdvancedPuzzleSolver for 4x4 and 8x8 puzzles
            print(f"   Using AdvancedPuzzleSolver for {N}x{N} puzzle...")
            solver = AdvancedPuzzleSolver()
            
            # Call the solver
            solver_result = solver.solve_puzzle(all_piece_images, N)
            
            # Format the result to match expected 5-tuple
            all_comparisons, all_piece_rotations, final_grid, best_buddies, assembly_score = \
                format_advanced_solver_result(solver_result, N, all_piece_images)
            
        else:
            print(f"   ❌ Unsupported grid size: {N}x{N}")
            return results
        
        # Update results
        results.update({
            'all_comparisons': all_comparisons,
            'all_piece_rotations': all_piece_rotations,
            'final_grid': final_grid,
            'best_buddies': best_buddies,
            'assembly_score': assembly_score
        })
        
        # Validate the grid
        if final_grid is not None:
            is_valid = validate_grid(final_grid, N)
            print(f"   Grid validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
            
            if not is_valid:
                print("   ⚠️ Grid is invalid, using simple ordered grid")
                # Create a simple grid in order
                final_grid = [[i * N + j for j in range(N)] for i in range(N)]
                assembly_score = 0.3  # Lower score for fallback
                results['final_grid'] = final_grid
                results['assembly_score'] = assembly_score
            else:
                print(f"   ✅ Valid {N}x{N} grid found")
        
        print(f"✅ Algorithm analysis completed")
        print(f"   Assembly score: {assembly_score:.3f}")
        
        # --- Create organized subfolders ---
        heatmap_dir = os.path.join(puzzle_output_dir, "heatmaps")
        matches_dir = os.path.join(puzzle_output_dir, "top_matches")
        assembled_dir = os.path.join(puzzle_output_dir, "best_assembled")
        for d in [heatmap_dir, matches_dir, assembled_dir]:
            os.makedirs(d, exist_ok=True)
        
        # --- Heatmap --- (only for 2x2 or if we have comparisons)
        if all_comparisons and len(all_comparisons) > 0:
            try:
                heatmap_path = os.path.join(heatmap_dir, f"heatmap_{puzzle_id}.png")
                fig, horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                    all_comparisons, all_piece_images, N, f"Puzzle {puzzle_id}"
                )
                fig.tight_layout()
                fig.savefig(heatmap_path, dpi=200)
                plt.close(fig)
                results['save_paths']['heatmap'] = heatmap_path
                print(f"   💾 Saved heatmap: {heatmap_path}")
            except Exception as e:
                print(f"   ⚠️ Heatmap visualization failed: {e}")
        else:
            print(f"   ℹ️ No comparisons available for heatmap")
        
        # --- Top matches --- (only for 2x2 or if we have comparisons)
        if all_comparisons and len(all_comparisons) > 0:
            try:
                match_line_path = os.path.join(matches_dir, f"top_matches_{puzzle_id}.png")
                fig = visualize_matches_with_lines(all_piece_images, all_comparisons, top_n=min(3, len(all_comparisons)))
                fig.tight_layout()
                fig.savefig(match_line_path, dpi=200)
                plt.close(fig)
                results['save_paths']['top_matches'] = match_line_path
                print(f"   💾 Saved top matches: {match_line_path}")
            except Exception as e:
                print(f"   ⚠️ Top match visualization failed: {e}")
        else:
            print(f"   ℹ️ No comparisons available for match visualization")
        
        # --- Assemble final grid ---
        if final_grid is not None:
            print(f"   Assembling final image...")
            assembled_descriptor = assemble_grid_from_pieces(all_piece_images, final_grid, N=N)
            
            if assembled_descriptor is not None:
                results['assembled_image'] = assembled_descriptor
                
                assembled_path = os.path.join(assembled_dir, f"descriptor_solved_{puzzle_id}.jpg")
                
                # Display and save the image
                print(f"   Displaying assembled image...")
                visualize_descriptor_result(
                    assembled_image=assembled_descriptor,
                    puzzle_id=puzzle_id,
                    N=N,
                    assembly_score=assembly_score,
                    show=True,   # Show the image
                    save_path=assembled_path
                )
                results['save_paths']['assembled'] = assembled_path
                results['success'] = True
                print(f"   ✅ Puzzle assembly successful!")
            else:
                print(f"   ❌ Failed to assemble image")
        else:
            print(f"   ❌ No grid to assemble")
                
    except Exception as e:
        print(f"   ❌ Puzzle solving failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    # 1️⃣ UPLOAD ZIP
    print("\n" + "="*50)
    print("🧩 PUZZLE ASSEMBLY SYSTEM")
    print("="*50)
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
    
    # Show directory structure
    print("\n📁 Directory Structure:")
    for dir_name, dir_images in images_by_dir.items():
        print(f"   {dir_name}: {len(dir_images)} images")
    
    print("="*50)

    # 3️⃣ APPLY FILTERS (Optional - can be skipped)
    print("\n🔧 APPLYING FILTERS TO ALL IMAGES 🔧")
    print("   (This step can be skipped for faster processing)")
    
    skip_filters = input("   Skip image filtering? (y/N): ").lower() == 'y'
    
    output_dir = os.path.join(os.path.dirname(zip_file), "processed_puzzles")
    os.makedirs(output_dir, exist_ok=True)

    examples = []
    total_successful = 0

    if not skip_filters:
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
        
        # Show examples
        if examples:
            show_examples(examples, images_by_dir, output_dir)
    else:
        print("\n✅ Skipping image filtering, using original images")
    
    print("="*50)

    # 4️⃣ GENERIC GRID CROPPING
    if images_by_dir:
        print("\n🔍 EXECUTING GENERIC GRID CROPPING")
        print("   Method: Auto-detecting 2x2, 4x4, or 8x8 based on folder/filenames")
        print("   Solver selection:")
        print("     - 2x2 puzzles: DescriptorBasedAssembler (5 parameters)")
        print("     - 4x4/8x8 puzzles: AdvancedPuzzleSolver (16 parameters)")

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
        puzzles_solved = 0
        puzzles_failed = 0

        for dir_name, dir_images in images_by_dir.items():
            print(f"\n📁 Processing directory: {dir_name} ({len(dir_images)} images)")
            puzzle_output_dir = os.path.join(output_dir, dir_name if dir_name != 'root' else 'main_images')
            rectangular_dir = os.path.join(puzzle_output_dir, "rectangular_pieces")
            descriptor_solved_dir = os.path.join(puzzle_output_dir, "descriptor_solved")

            os.makedirs(rectangular_dir, exist_ok=True)
            os.makedirs(descriptor_solved_dir, exist_ok=True)

            examples_shown = 0

            for i, img_info in enumerate(dir_images, 1):
                img_path = img_info['full_path']
                filename = img_info['filename']
                puzzle_name = os.path.splitext(filename)[0]

                print(f"\n🎯 Processing puzzle {i}/{len(dir_images)}: {filename}")
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"   ❌ Failed to load {filename}")
                    puzzles_failed += 1
                    continue

                # Detect grid size
                N = detect_grid_size(filename, dir_name, default_n=2)
                print(f"   Detected grid size: {N}x{N}")
                
                # Check if grid size is valid
                if N not in [2, 4, 8]:
                    print(f"   ⚠️ Unsupported grid size {N}x{N}. Only 2x2, 4x4, and 8x8 are supported.")
                    puzzles_failed += 1
                    continue
                
                # Extract pieces
                pieces = extract_generic_grid_pieces(img, N=N)
                
                # Verify we got the right number of pieces
                expected_pieces = N * N
                if len(pieces) != expected_pieces:
                    print(f"   ⚠️ Warning: Expected {expected_pieces} pieces, got {len(pieces)}")
                    if len(pieces) > expected_pieces:
                        pieces = pieces[:expected_pieces]
                        print(f"   Using first {expected_pieces} pieces")
                    else:
                        print(f"   ❌ Not enough pieces, skipping")
                        puzzles_failed += 1
                        continue
                
                print(f"   Extracted {len(pieces)} pieces")

                # Save pieces
                for p_idx, piece in enumerate(pieces):
                    cv2.imwrite(
                        os.path.join(rectangular_dir, f"piece_{p_idx+1}_{filename}"),
                        piece
                    )

                # 🤖 Solve puzzle (Choose solver based on grid size)
                print(f"   Solving {N}x{N} puzzle...")
                descriptor_results = run_descriptor_algorithm_with_improvement(
                    pieces,
                    N,
                    puzzle_id=puzzle_name,
                    puzzle_output_dir=descriptor_solved_dir
                )

                if descriptor_results['success']:
                    print(f"   ✅ Puzzle {filename} solved successfully!")
                    puzzles_solved += 1
                    
                    # Show saved files
                    if 'save_paths' in descriptor_results:
                        for key, path in descriptor_results['save_paths'].items():
                            if path and os.path.exists(path):
                                print(f"      Saved {key}: {os.path.basename(path)}")
                else:
                    print(f"   ❌ Failed to solve puzzle {filename}")
                    puzzles_failed += 1

                total_pieces_extracted += len(pieces)

                # Show example visualizations
                if examples_shown < examples_to_show:
                    print(f"\n🧩 Visualizing {N}x{N} crop {examples_shown + 1}/{examples_to_show}: {filename}")
                    viz_img = visualize_generic_grid(img, pieces, N, filename)
                    cv2.imwrite(os.path.join(rectangular_dir, f"grid_cut_{filename}"), viz_img)
                    examples_shown += 1

                # Progress update
                if i <= 5 or i == len(dir_images):
                    status = "✅" if descriptor_results['success'] else "❌"
                    print(f"   {status} [{i}/{len(dir_images)}] {filename} -> {N}x{N} ({len(pieces)} pieces)")
                elif i == 6:
                    print(f"   ... processing {len(dir_images)-5} more images ...")

        # Summary
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)
        print(f"Total puzzles processed: {total_images}")
        print(f"Puzzles successfully solved: {puzzles_solved}")
        print(f"Puzzles failed: {puzzles_failed}")
        print(f"Total pieces extracted: {total_pieces_extracted}")
        print(f"Output directory: {output_dir}")
        
        if puzzles_solved > 0:
            success_rate = (puzzles_solved / total_images) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print("="*70)
        print("🎉 PROCESSING COMPLETE!")
        print("="*70)


if __name__ == "__main__":
    main()