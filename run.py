import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from functions import  assemble_grid_from_pieces, detect_grid_size, enhance_image, estimate_noise, extract_generic_grid_pieces, selective_median_filter
from paper_algorithms import PaperPuzzleSolver
from visualize import show_examples, visualize_generic_grid, visualize_orientation_comparison, visualize_comparison_heatmap, visualize_matches_with_lines
from descriptor_assembler import DescriptorBasedAssembler

def run_paper_algorithm_with_fix(all_piece_images, piece_files, N, puzzle_id, puzzle_output_dir):
    """
    Run paper algorithm with simpler but more effective assembly
    """
    print(f"\n📘 PAPER ALGORITHM")
    print("   Method: Pomeranz et al. (2011) - Simple Greedy Assembly")
    
    results = {
        'success': False,
        'all_comparisons': None,
        'all_piece_rotations': None,
        'final_grid': None,
        'best_buddies': None,
        'assembled_image': None,
        'save_path': None,
        'grid_score': 0,
        'combined_score': 0
    }
    
    try:
        # Initialize solver
        solver = PaperPuzzleSolver(p=0.3, q=1/16, use_prediction=True, border_width=10)
        
        # Step 1: Compute pairwise comparisons
        print("   Step 1: Computing pairwise comparisons...")
        all_comparisons, all_piece_rotations, best_buddies = solver.solve_for_comparisons(all_piece_images)

        # Step 2: Build compatibility matrix
        print("   Step 2: Building compatibility matrix...")
        compatibility_matrix = solver.build_compatibility_matrix(all_piece_images)
        
        # Step 3: Use the solver's own assembly method
        print("   Step 3: Assembling puzzle with improved greedy algorithm...")
        
        try:
            # Use the improved greedy assembly
            final_grid = solver.greedy_assemble(all_piece_images, compatibility_matrix, N)
            print(f"   ✓ Assembly completed")
        except Exception as e:
            print(f"   ⚠️ Improved assembly failed: {e}")
            # Fallback to simple sequential placement
            final_grid = []
            piece_idx = 0
            for r in range(N):
                row = []
                for c in range(N):
                    if piece_idx < len(all_piece_images):
                        row.append(piece_idx)
                        piece_idx += 1
                    else:
                        row.append(None)
                final_grid.append(row)
            print(f"   ⚠️ Using fallback sequential placement")

        results['all_comparisons'] = all_comparisons
        results['all_piece_rotations'] = all_piece_rotations
        results['final_grid'] = final_grid
        results['best_buddies'] = best_buddies
        
        print(f"\n✅ Paper Algorithm analysis completed")
        print(f"   Found {len(all_comparisons)} comparisons")
        print(f"   Found {len(best_buddies)} best-buddy pairs")
        print(f"   Grid size: {N}x{N}")
        
        # Show top matches
        if all_comparisons:
            sorted_comparisons = sorted(all_comparisons, key=lambda x: x['score'], reverse=True)
            print(f"\n🏆 TOP 5 MATCHES:")
            for idx in range(min(5, len(sorted_comparisons))):
                match = sorted_comparisons[idx]
                print(f"   {idx+1}. P{match['piece1']+1} {match['edge1']} ↔ P{match['piece2']+1} {match['edge2']} "
                      f"(Score: {match['score']:.4f})")
        
        # ========== SIMPLE ASSEMBLY AND SCORING ==========
        if final_grid is not None:
            print(f"\n🎯 ASSEMBLING FINAL GRID")
            
            try:
                # Assemble the grid
                assembled_paper = assemble_grid_from_pieces(all_piece_images, final_grid, N=N)
                
                if assembled_paper is not None:
                    results['assembled_image'] = assembled_paper
                    
                    # Calculate simple score
                    if all_comparisons:
                        # Evaluate grid quality
                        grid_score = solver.evaluate_assembly(final_grid, compatibility_matrix, N)
                        
                        # Calculate average of top scores
                        top_scores = [c['score'] for c in sorted_comparisons[:min(10, len(sorted_comparisons))]]
                        top_score_avg = np.mean(top_scores) if top_scores else 0.5
                        
                        # Combined score: 60% grid score, 40% top matches
                        combined_score = 0.6 * grid_score + 0.4 * top_score_avg
                    else:
                        grid_score = 0.5
                        combined_score = 0.5
                    
                    results['grid_score'] = grid_score
                    results['combined_score'] = combined_score
                    
                    print(f"\n   📊 Assembly Statistics:")
                    print(f"   📈 Scores:")
                    print(f"      - Grid compatibility: {grid_score:.3f}")
                    print(f"      - Top match quality: {top_score_avg:.3f}")
                    print(f"      - Combined score: {combined_score:.3f}")
                    
                    # Visualize the assembled result
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
                    
                    plt.title(f"Paper Algorithm - Puzzle {puzzle_id} ({N}x{N})\n"
                             f"Quality: {quality} ({combined_score:.3f})", 
                             fontsize=14, fontweight='bold', color=color)
                    plt.axis('off')
                    
                    # Add score annotations
                    plt.figtext(0.5, 0.02, 
                               f"Grid Score: {grid_score:.3f} | Combined: {combined_score:.3f}",
                               ha='center', fontsize=10, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Save paper assembly
                    paper_save_path = os.path.join(puzzle_output_dir, f"paper_assembled_{puzzle_id}.jpg")
                    cv2.imwrite(paper_save_path, assembled_paper)
                    results['save_path'] = paper_save_path
                    print(f"   💾 Saved paper assembly to: {paper_save_path}")
                    
                    results['success'] = True
                    
                else:
                    print("   ⚠️ Could not assemble the grid")
                    
            except Exception as e:
                print(f"   ⚠️ Assembly failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("   ⚠️ No grid assembly from paper solver")
            
    except Exception as e:
        print(f"   ❌ Paper Algorithm failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def run_descriptor_algorithm_with_improvement(all_piece_images, N, puzzle_id, puzzle_output_dir):
    """
    Run descriptor algorithm with improved scoring
    """
    print(f"\n🤖 DESCRIPTOR-BASED ALGORITHM")
    print("   Method: Enhanced edge descriptors with better discrimination")
    
    results = {
        'success': False,
        'all_comparisons': None,
        'all_piece_rotations': None,
        'final_grid': None,
        'best_buddies': None,
        'assembly_score': 0,
        'assembled_image': None,
        'save_path': None
    }
    
    try:
        # Initialize descriptor-based assembler
        descriptor_assembler = DescriptorBasedAssembler(
            border_width=8,
            descriptor_length=100
        )
        
        # Solve using descriptor-based algorithm
        all_comparisons, all_piece_rotations, final_grid, best_buddies, assembly_score = \
            descriptor_assembler.solve(all_piece_images)
        
        results.update({
            'all_comparisons': all_comparisons,
            'all_piece_rotations': all_piece_rotations,
            'final_grid': final_grid,
            'best_buddies': best_buddies,
            'assembly_score': assembly_score
        })
        
        print(f"✅ Descriptor Algorithm analysis completed (Score: {assembly_score:.3f})")
        
        # =====================================
        # VISUALIZE ALL COMPARISONS (Heatmap)
        # =====================================
        if all_comparisons:
            print(f"\n📊 Visualizing all descriptor matches (Heatmap)")
            try:
                horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                    all_comparisons, all_piece_images, N, f"Descriptor Algorithm - Puzzle {puzzle_id}"
                )
            except Exception as e:
                print(f"   ⚠️ Heatmap visualization failed: {e}")
        
        # =====================================
        # VISUALIZE TOP 3 MATCHES
        # =====================================
        if all_comparisons:
            print(f"\n🔗 Visualizing top 3 descriptor matches")
            try:
                visualize_matches_with_lines(all_piece_images, all_comparisons, top_n=3)
            except Exception as e:
                print(f"   ⚠️ Top matches visualization failed: {e}")
        
        # =====================================
        # ASSEMBLE FINAL GRID
        # =====================================
        if final_grid is not None:
            print(f"\n🎯 Assembling descriptor grid")
            try:
                assembled_descriptor = assemble_grid_from_pieces(all_piece_images, final_grid, N=N)
                
                if assembled_descriptor is not None:
                    results['assembled_image'] = assembled_descriptor
                    
                    # Visualize assembled image
                    plt.figure(figsize=(8, 8))
                    plt.imshow(cv2.cvtColor(assembled_descriptor, cv2.COLOR_BGR2RGB))
                    
                    # Color code quality
                    if assembly_score > 0.3:
                        color, quality = "green", "Excellent"
                    elif assembly_score > 0.2:
                        color, quality = "orange", "Good"
                    elif assembly_score > 0.1:
                        color, quality = "yellow", "Fair"
                    else:
                        color, quality = "red", "Poor"
                    
                    plt.title(
                        f"Descriptor Algorithm - Puzzle {puzzle_id} ({N}x{N})\n"
                        f"Score: {assembly_score:.3f} | Quality: {quality}",
                        fontsize=14, fontweight='bold', color=color
                    )
                    plt.axis('off')
                    
                    plt.figtext(
                        0.5, 0.02,
                        f"Assembly Score: {assembly_score:.3f}",
                        ha='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
                    )
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Save assembled image
                    descriptor_save_path = os.path.join(puzzle_output_dir, f"descriptor_assembled_{puzzle_id}.jpg")
                    cv2.imwrite(descriptor_save_path, assembled_descriptor)
                    results['save_path'] = descriptor_save_path
                    results['success'] = True
                    print(f"   💾 Saved descriptor assembly to: {descriptor_save_path}")
                else:
                    print("   ⚠️ Could not assemble descriptor grid")
            except Exception as e:
                print(f"   ⚠️ Descriptor assembly failed: {e}")
        else:
            print("   ⚠️ No grid from descriptor solver")
    
    except Exception as e:
        print(f"   ❌ Descriptor Algorithm failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results

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

    # 5️⃣ RUN BOTH ALGORITHMS
    if 'images_by_dir' in locals():
        print("\n" + "="*70)
        print("🔍 DUAL ALGORITHM COMPARISON")
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
                
                # ========== RUN PAPER ALGORITHM ==========
                paper_results = run_paper_algorithm_with_fix(
                    all_piece_images, pieces, N, puzzle_id, puzzle_output_dir
                )
                
                # ========== RUN DESCRIPTOR ALGORITHM ==========
                descriptor_results = run_descriptor_algorithm_with_improvement(
                    all_piece_images, N, puzzle_id, puzzle_output_dir
                )
                
                # ========== COMPARE BOTH ALGORITHMS ==========
                print(f"\n🔀 ALGORITHM COMPARISON")
                print("="*60)
                
                if paper_results['success'] and descriptor_results['success']:
                    print("📊 Both algorithms succeeded!")
                    print(f"   Paper Algorithm Score: {paper_results.get('combined_score', 0):.3f}")
                    print(f"   Descriptor Algorithm Score: {descriptor_results.get('assembly_score', 0):.3f}")
                    
                    # Determine which algorithm performed better
                    paper_score = paper_results.get('combined_score', 0)
                    descriptor_score = descriptor_results.get('assembly_score', 0)
                    
                    if paper_score > descriptor_score:
                        diff = paper_score - descriptor_score
                        print(f"   🏆 Winner: Paper Algorithm (by {diff:.3f})")
                        winner = "Paper"
                    elif descriptor_score > paper_score:
                        diff = descriptor_score - paper_score
                        print(f"   🏆 Winner: Descriptor Algorithm (by {diff:.3f})")
                        winner = "Descriptor"
                    else:
                        print(f"   ⚖️ Tie: Both algorithms performed equally")
                        winner = "Tie"
                    
                    # Create side-by-side comparison visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Paper algorithm result
                    ax1.imshow(cv2.cvtColor(paper_results['assembled_image'], cv2.COLOR_BGR2RGB))
                    ax1.set_title(f"Paper Algorithm\nScore: {paper_score:.3f}", 
                                 fontsize=14, fontweight='bold',
                                 color='green' if winner == "Paper" else 'black')
                    ax1.axis('off')
                    
                    # Descriptor algorithm result
                    ax2.imshow(cv2.cvtColor(descriptor_results['assembled_image'], cv2.COLOR_BGR2RGB))
                    ax2.set_title(f"Descriptor Algorithm\nScore: {descriptor_score:.3f}", 
                                 fontsize=14, fontweight='bold',
                                 color='green' if winner == "Descriptor" else 'black')
                    ax2.axis('off')
                    
                    plt.suptitle(f"Algorithm Comparison - Puzzle {puzzle_id} ({N}x{N})", 
                                fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.show()
                    
                elif paper_results['success']:
                    print("📊 Only Paper Algorithm succeeded")
                elif descriptor_results['success']:
                    print("📊 Only Descriptor Algorithm succeeded")
                else:
                    print("📊 Both algorithms failed to assemble the puzzle")
                
                # ========== SAVE COMPREHENSIVE RESULTS SUMMARY ==========
                results_file = os.path.join(puzzle_output_dir, f"dual_algorithm_results_{puzzle_id}.txt")
                with open(results_file, 'w') as f:
                    f.write(f"DUAL ALGORITHM COMPARISON - Puzzle {puzzle_id} ({N}x{N})\n")
                    f.write("="*60 + "\n\n")
                    
                    f.write("PAPER ALGORITHM RESULTS:\n")
                    if paper_results['success']:
                        f.write(f"  • Success: ✓\n")
                        f.write(f"  • Combined Score: {paper_results.get('combined_score', 0):.3f}\n")
                        
                    else:
                        f.write(f"  • Success: ✗\n")
                    
                    f.write(f"\nDESCRIPTOR ALGORITHM RESULTS:\n")
                    if descriptor_results['success']:
                        f.write(f"  • Success: ✓\n")
                        f.write(f"  • Assembly Score: {descriptor_results.get('assembly_score', 0):.3f}\n")
                        f.write(f"  • Total Comparisons: {len(descriptor_results.get('all_comparisons', []))}\n")
                        if descriptor_results['save_path']:
                            f.write(f"  • Saved to: {descriptor_results['save_path']}\n")
                    else:
                        f.write(f"  • Success: ✗\n")
                    
                    f.write(f"\nCOMPARISON:\n")
                    if paper_results['success'] and descriptor_results['success']:
                        if paper_score > descriptor_score:
                            f.write(f"  • Winner: Paper Algorithm (by {paper_score - descriptor_score:.3f})\n")
                        elif descriptor_score > paper_score:
                            f.write(f"  • Winner: Descriptor Algorithm (by {descriptor_score - paper_score:.3f})\n")
                        else:
                            f.write(f"  • Tie: Both algorithms performed equally\n")
                    elif paper_results['success']:
                        f.write(f"  • Only Paper Algorithm succeeded\n")
                    elif descriptor_results['success']:
                        f.write(f"  • Only Descriptor Algorithm succeeded\n")
                    else:
                        f.write(f"  • Both algorithms failed\n")
                
                print(f"   📝 Comprehensive results saved to: {results_file}")
                
                # Store assembly results
                result_entry = {
                    'puzzle_id': puzzle_id,
                    'grid_size': N,
                    'num_pieces': num_pieces,
                    'paper_success': paper_results['success'],
                    'descriptor_success': descriptor_results['success'],
                    'paper_score': paper_results.get('combined_score', 0),
                    'descriptor_score': descriptor_results.get('assembly_score', 0)
                }
                
                assembly_results.append(result_entry)
                
                print(f"\n✅ Puzzle {puzzle_id} assembly completed [{processed_puzzles}/{puzzles_to_assemble}]")
            
            # Show final summary
            print("\n" + "=" * 80)
            print("📊 DUAL ALGORITHM SUMMARY")
            print("=" * 80)
            
            if assembly_results:
                paper_success = sum(1 for r in assembly_results if r['paper_success'])
                descriptor_success = sum(1 for r in assembly_results if r['descriptor_success'])
                both_success = sum(1 for r in assembly_results if r['paper_success'] and r['descriptor_success'])
                
                print(f"\n📈 Success Rates:")
                print(f"   Paper Algorithm: {paper_success}/{processed_puzzles} ({paper_success/processed_puzzles*100:.1f}%)")
                print(f"   Descriptor Algorithm: {descriptor_success}/{processed_puzzles} ({descriptor_success/processed_puzzles*100:.1f}%)")
                print(f"   Both Algorithms: {both_success}/{processed_puzzles} ({both_success/processed_puzzles*100:.1f}%)")
                
                print(f"\n📊 Average Scores (for successful assemblies):")
                if paper_success > 0:
                    avg_paper_score = np.mean([r['paper_score'] for r in assembly_results if r['paper_success']])
                    print(f"   Paper Algorithm: {avg_paper_score:.3f}")
                else:
                    print(f"   Paper Algorithm: N/A (no successful assemblies)")
                
                if descriptor_success > 0:
                    avg_descriptor_score = np.mean([r['descriptor_score'] for r in assembly_results if r['descriptor_success']])
                    print(f"   Descriptor Algorithm: {avg_descriptor_score:.3f}")
                else:
                    print(f"   Descriptor Algorithm: N/A (no successful assemblies)")
                
                print(f"\n🧩 Detailed Results:")
                for result in assembly_results:
                    paper_status = '✓' if result['paper_success'] else '✗'
                    descriptor_status = '✓' if result['descriptor_success'] else '✗'
                    
                    print(f"   Puzzle {result['puzzle_id']} ({result['grid_size']}x{result['grid_size']}):")
                    if result['paper_success']:
                        print(f"      Paper: {paper_status} (score: {result['paper_score']:.3f})")
                    if result['descriptor_success']:
                        print(f"      Descriptor: {descriptor_status} (score: {result['descriptor_score']:.3f})")
                    print()
            else:
                print("No puzzles were successfully assembled.")
            
            print(f"✅ Total puzzles processed: {processed_puzzles}/{puzzles_to_assemble}")
            print("=" * 80)

if __name__ == "__main__":
    main()
