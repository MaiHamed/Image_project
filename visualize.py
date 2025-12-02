
import matplotlib.pyplot as plt
import cv2
import numpy as np  
import os

def show_examples(examples, images_by_dir, output_dir):
    """
    Display before/after images with metrics and available processed folders.
    """
    print("\n" + "🔍" * 10 + " SHOW EXAMPLES & METRICS " + "🔍" * 10)

    if not examples:
        print("❌ No images were processed! Please run processing first.")
        return

    # Show processing summary
    print(f"🎯 You processed {sum(len(v) for v in images_by_dir.values())} images from {len(images_by_dir)} locations")

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
        print(f"⚠ Invalid input. Showing {num_examples} examples by default")

    # Show examples
    for i, example in enumerate(examples[:num_examples], 1):
        print(f"\n🖼 Example {i}/{num_examples}: {example['filename']}")
        print(f"📁 Location: {example['directory']}")

        plt.figure(figsize=(18, 6))
        plt.suptitle(f"{example['filename']}", fontsize=16, fontweight='bold')  # <-- filename at top

        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(example['original'], cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(example['denoised'], cv2.COLOR_BGR2RGB))
        plt.title("Denoised")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(example['enhanced'], cv2.COLOR_BGR2RGB))
        plt.title("Enhanced")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(example['edges_bw'], cmap='gray')
        plt.title("Edges")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
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
            processed_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
            print(f"   {i}. {subdir} ({processed_count} images)")

    print("\n" + "=" * 60)
    print("EXAMPLES DISPLAY COMPLETE")
    print("=" * 60)

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
    plt.suptitle(f"{filename}", fontsize=16, fontweight='bold')  # <-- filename at top
    
    # Left: The cut lines
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cut_viz, cv2.COLOR_BGR2RGB))
    plt.title(f"Grid Slicing ({N}x{N})", fontweight='bold')
    plt.axis('off')
    
    # Right: The extracted pieces in a grid
    plt.subplot(1, 2, 2)
    
    # Create a display grid
    gap = 5
    piece_h, piece_w = pieces[0].shape[:2]
    
    grid_h = N * piece_h + (N-1) * gap
    grid_w = N * piece_w + (N-1) * gap
    
    display_grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # White bg
    
    idx = 0
    for row in range(N):
        for col in range(N):
            if idx < len(pieces):
                y = row * (piece_h + gap)
                x = col * (piece_w + gap)
                p = cv2.resize(pieces[idx], (piece_w, piece_h))
                display_grid[y:y+piece_h, x:x+piece_w] = p
                idx += 1
    
    plt.imshow(cv2.cvtColor(display_grid, cv2.COLOR_BGR2RGB))
    plt.title(f"Extracted {N*N} Pieces", fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()
    
    return cut_viz

def visualize_comparison_heatmap(all_comparisons, piece_files, N, puzzle_name):
    num = len(piece_files)
    horizontal_scores = np.full((num, num), np.nan)
    vertical_scores = np.full((num, num), np.nan)

    for match in all_comparisons:
        i, j = match['piece1'], match['piece2']
        if match['edge1'] == 'right' and match['edge2'] == 'left':
            horizontal_scores[i, j] = match['score']
        elif match['edge1'] == 'bottom' and match['edge2'] == 'top':
            vertical_scores[i, j] = match['score']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(horizontal_scores, cmap='RdYlGn_r', aspect='auto')
    ax1.set_title("Horizontal Match Scores")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(vertical_scores, cmap='RdYlGn_r', aspect='auto')
    ax2.set_title("Vertical Match Scores")
    plt.colorbar(im2, ax=ax2)

    # Add text on squares
    hmin, hmax = np.nanmin(horizontal_scores), np.nanmax(horizontal_scores)
    vmin, vmax = np.nanmin(vertical_scores), np.nanmax(vertical_scores)

    for i in range(num):
        for j in range(num):
            if not np.isnan(horizontal_scores[i, j]):
                threshold = hmin + 0.5 * (hmax - hmin)
                ax1.text(j, i, f"{horizontal_scores[i, j]:.4f}",
                         ha='center', va='center', fontsize=8,
                         color='white' if horizontal_scores[i, j] > threshold else 'black')
            if not np.isnan(vertical_scores[i, j]):
                threshold = vmin + 0.5 * (vmax - vmin)
                ax2.text(j, i, f"{vertical_scores[i, j]:.4f}",
                         ha='center', va='center', fontsize=8,
                         color='white' if vertical_scores[i, j] > threshold else 'black')

    plt.suptitle(f"Puzzle Match Analysis: {puzzle_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

    return horizontal_scores, vertical_scores


def visualize_best_match_pair(piece1_img, piece2_img, desc1, desc2, score, info):
    p1 = info['piece1'] + 1
    p2 = info['piece2'] + 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.imshow(cv2.cvtColor(piece1_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Piece {p1}\n{info['edge1']}")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(piece2_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Piece {p2}\n{info['edge2']} (rot {info['rotation_of_piece2']}°)")
    ax2.axis('off')

    if info['edge1'] == 'right' and info['edge2'] == 'left':
        desc2_plot = desc2[::-1]
    else:
        desc2_plot = desc2

    ax3.plot(desc1, label="Piece 1")
    ax3.plot(desc2_plot, label="Piece 2")
    ax3.set_title(f"Score = {score:.4f}")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
