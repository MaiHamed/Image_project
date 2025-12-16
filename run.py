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
from functions import  assemble_grid_from_pieces, detect_grid_size, enhance_image, estimate_noise, extract_generic_grid_pieces, selective_median_filter
from visualize import show_examples, visualize_generic_grid, visualize_orientation_comparison, visualize_comparison_heatmap, visualize_matches_with_lines, visualize_descriptor_result


def reconstruct_image(pieces, placement, grid_n):
    ph, pw = pieces[0].shape[:2]
    canvas = np.zeros((grid_n * ph, grid_n * pw, 3), dtype=np.uint8)

    for idx, pid in enumerate(placement):
        r, c = divmod(idx, grid_n)
        canvas[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = pieces[pid]

    return canvas
def run_descriptor_algorithm_with_improvement(all_piece_images, N, puzzle_id, puzzle_output_dir):
    """
    Run descriptor algorithm with improved scoring and save visualizations.
    Saves outputs in organized subfolders: heatmaps, top_matches, best_assembled
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
        'save_paths': {}
    }
    
    try:
        # Solve puzzle
        descriptor_assembler = DescriptorBasedAssembler(border_width=8, descriptor_length=100)
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
        
        # --- Create organized subfolders ---
        heatmap_dir = os.path.join(puzzle_output_dir, "heatmaps")
        matches_dir = os.path.join(puzzle_output_dir, "top_matches")
        assembled_dir = os.path.join(puzzle_output_dir, "best_assembled")
        for d in [heatmap_dir, matches_dir, assembled_dir]:
            os.makedirs(d, exist_ok=True)
        
        # --- Heatmap ---
        if all_comparisons:
            try:
                heatmap_path = os.path.join(heatmap_dir, f"heatmap_{puzzle_id}.png")
                # Let the function create the figure and return it
                fig, horizontal_scores, vertical_scores = visualize_comparison_heatmap(
                    all_comparisons, all_piece_images, N, f"Puzzle {puzzle_id}"
                )
                fig.tight_layout()
                fig.savefig(heatmap_path, dpi=200)
                plt.close(fig)
                results['save_paths']['heatmap'] = heatmap_path
            except Exception as e:
                print(f"   ⚠️ Heatmap visualization failed: {e}")

        # --- Top matches ---
        try:
            match_line_path = os.path.join(matches_dir, f"top_matches_{puzzle_id}.png")
            # Return figure from the function
            fig = visualize_matches_with_lines(all_piece_images, all_comparisons, top_n=3)
            fig.tight_layout()
            fig.savefig(match_line_path, dpi=200)
            plt.close(fig)
            results['save_paths']['top_matches'] = match_line_path
        except Exception as e:
            print(f"   ⚠️ Top match visualization failed: {e}")

        
        # --- Assemble final grid ---
        if final_grid is not None:
            assembled_descriptor = assemble_grid_from_pieces(all_piece_images, final_grid, N=N)
            results['assembled_image'] = assembled_descriptor
            
            assembled_path = os.path.join(assembled_dir, f"descriptor_solved_{puzzle_id}.jpg")
            visualize_descriptor_result(
                assembled_image=assembled_descriptor,
                puzzle_id=puzzle_id,
                N=N,
                assembly_score=assembly_score,
                show=False,   # Do not pop up window
                save_path=assembled_path
            )
            results['save_paths']['assembled'] = assembled_path
            results['success'] = True
                
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
            solved_dir = os.path.join(puzzle_output_dir, "solved")
            descriptor_solved_dir = os.path.join(puzzle_output_dir, "descriptor_solved")

            os.makedirs(rectangular_dir, exist_ok=True)
            os.makedirs(solved_dir, exist_ok=True)
            os.makedirs(descriptor_solved_dir, exist_ok=True)

            examples_shown = 0

            for i, img_info in enumerate(dir_images, 1):
                img_path = img_info['full_path']
                filename = img_info['filename']

                img = cv2.imread(img_path)
                if img is None:
                    print(f"   ❌ Failed to load {filename}")
                    continue

                N = detect_grid_size(filename, dir_name, default_n=2)
                pieces = extract_generic_grid_pieces(img, N=N)

                for p_idx, piece in enumerate(pieces):
                    cv2.imwrite(
                        os.path.join(rectangular_dir, f"piece_{p_idx+1}_{filename}"),
                        piece
                    )

                # 🤖 Solve puzzle (Descriptor-based)
                descriptor_results = run_descriptor_algorithm_with_improvement(
                    pieces,
                    N,
                    puzzle_id=filename.split('.')[0],
                    puzzle_output_dir=descriptor_solved_dir
                )

                if descriptor_results['success']:
                    print(f"   ✅ Puzzle {filename} solved!")
                    for key, path in descriptor_results['save_paths'].items():
                        print(f"      Saved {key}: {path}")


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


if __name__ == "__main__":
    main()

