import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io  # Import io module


def load_image(path):
    """Loads an image, handles potential read errors, and converts it from BGR to RGB."""
    try:
        # Try reading with cv2.imread directly first
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # If cv2.imread fails, try reading as raw bytes and then decode
            with open(path, 'rb') as f:
                img_bytes = f.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Failed to load image from {path} even after byte-level attempt.")
            return None

        # OpenCV loads images in BGR format, matplotlib expects RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"An error occurred while loading image {path}: {e}")
        return None


def visualize_single_image_comparison(
        image_name,
        gt_folder, lr_folder, bicubic_folder, hat_psnr_folder, real_hat_gan_folder,
        output_save_dir='./results/visual_comparisons'
):
    """
    Visualizes comparison for a single image across different SR methods.

    Args:
        image_name (str): Base name of the image (e.g., 'baby.png').
        gt_folder (str): Path to the high-resolution ground truth images.
        lr_folder (str): Path to the low-resolution input images.
        bicubic_folder (str): Path to the bicubic upscaled images.
        hat_psnr_folder (str): Path to the HAT PSNR-optimized model outputs.
        real_hat_gan_folder (str): Path to the Real-HAT-GAN model outputs.
        output_save_dir (str): Directory to save the comparison figures.
    """
    os.makedirs(output_save_dir, exist_ok=True)

    base_name_without_ext = os.path.splitext(image_name)[0]

    # --- Construct paths for each image type ---
    # GT: 'baby.png'
    path_gt = os.path.join(gt_folder, image_name)

    # LR: 'babyx4.png' (Input to models)
    path_lr = os.path.join(lr_folder, base_name_without_ext + 'x4.png')

    # Bicubic: 'babyx4_bic.png' (Output from your generate_bicubic.py)
    path_bicubic = os.path.join(bicubic_folder, base_name_without_ext + 'x4_bic.png')

    # HAT PSNR: 'babyx4_HAT_SRx4.png' (Output from HAT_SRx4_ImageNet-pretrain.yml)
    path_hat_psnr = os.path.join(hat_psnr_folder, base_name_without_ext + 'x4_HAT_SRx4.png')

    # Real-HAT-GAN: 'babyx4_HAT_GAN_Real_SRx4.png' (Output from HAT_GAN_Real_SRx4.yml)
    path_real_hat_gan = os.path.join(real_hat_gan_folder, base_name_without_ext + 'x4_HAT_GAN_Real_SRx4.png')

    # Load all images
    img_gt = load_image(path_gt)
    img_lr = load_image(path_lr)  # Low-resolution input
    img_bicubic = load_image(path_bicubic)
    img_hat_psnr = load_image(path_hat_psnr)
    img_real_hat_gan = load_image(path_real_hat_gan)

    # Check if all critical images are loaded
    if any(img is None for img in [img_gt, img_lr, img_bicubic, img_hat_psnr, img_real_hat_gan]):
        print(f"Skipping visualization for {image_name} due to missing image files.")
        return

    # --- Create the plot ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 1 row, 5 columns
    titles = ['Low Resolution Input', 'Bicubic Upscaled', 'HAT PSNR-Optimized', 'Real-HAT-GAN', 'Ground Truth HR']
    images = [img_lr, img_bicubic, img_hat_psnr, img_real_hat_gan, img_gt]

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')  # Hide axes ticks

    plt.suptitle(f"Super-Resolution Comparison for {image_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # Save the figure
    save_path_full = os.path.join(output_save_dir, f"{base_name_without_ext}_comparison.png")
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    print(f"Saved comparison figure: {save_path_full}")
    plt.close(fig)  # Close the figure to free up memory

    # --- Create a detailed zoomed-in plot (Crucial for showing differences) ---
    # Select a region to zoom in. You might need to adjust these coordinates.
    # A common approach is to pick a central region or a region with fine details.
    # For Set5/baby.png, for instance, faces or hair are good regions.
    # Assuming images are large enough (e.g., 512x512 for GT, Bicubic, HAT outputs)
    # Let's pick a 128x128 pixel region (adjust as needed)

    zoom_size = 128
    # Try to center the zoom region or pick a known detailed spot.
    # For a general approach, let's pick a center region.
    h, w, _ = img_gt.shape
    start_y = h // 2 - zoom_size // 2
    end_y = start_y + zoom_size
    start_x = w // 2 - zoom_size // 2
    end_x = start_x + zoom_size

    # Ensure coordinates are within bounds
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(h, end_y)
    end_x = min(w, end_x)

    # Extract zoomed regions
    zoom_gt = img_gt[start_y:end_y, start_x:end_x]
    zoom_lr = img_lr[start_y:end_y, start_x:end_x]  # LR is small, so this crop would be tiny if not for imshow scaling
    # For LR zoom, it's better to show the LR original, then the upscaled ones
    zoom_bicubic = img_bicubic[start_y:end_y, start_x:end_x]
    zoom_hat_psnr = img_hat_psnr[start_y:end_y, start_x:end_x]
    zoom_real_hat_gan = img_real_hat_gan[start_y:end_y, start_x:end_x]

    # If LR is very small, we might want to zoom into its corresponding region directly,
    # or just show the upscaled zoomed regions for clarity.
    # Let's show the upscaled results' zoomed regions for direct comparison of details.

    fig_zoom, axes_zoom = plt.subplots(1, 4, figsize=(16, 4))  # Only compare upscaled versions here
    zoom_titles = ['Bicubic Upscaled (Zoom)', 'HAT PSNR-Optimized (Zoom)', 'Real-HAT-GAN (Zoom)',
                   'Ground Truth HR (Zoom)']
    zoom_images = [zoom_bicubic, zoom_hat_psnr, zoom_real_hat_gan, zoom_gt]

    for i, ax_zoom in enumerate(axes_zoom):
        ax_zoom.imshow(zoom_images[i])
        ax_zoom.set_title(zoom_titles[i], fontsize=10)
        ax_zoom.axis('off')

    plt.suptitle(f"Super-Resolution Detailed View for {image_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path_zoom = os.path.join(output_save_dir, f"{base_name_without_ext}_zoom_comparison.png")
    plt.savefig(save_path_zoom, dpi=300, bbox_inches='tight')
    print(f"Saved zoomed comparison figure: {save_path_zoom}")
    plt.close(fig_zoom)


# --- Main execution ---
if __name__ == '__main__':
    # --- Configuration (Adjust these paths) ---
    # Make sure you run this script from the HAT-main project root directory.

    gt_folder_path = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/datasets/GTmod4/'
    lr_input_folder = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/datasets/LRbicx4/X4'  # Folder containing actual LR input images

    bicubic_output_folder = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/results/Set5/bicubic_upscaled'
    hat_psnr_output_folder = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/results/HAT_SRx4/visualization/Set5/'
    real_hat_gan_output_folder = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/results/HAT_GAN_Real_SRx4/visualization/Set5/'

    output_comparison_figures_dir = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/results/comparison_figures'  # Where to save all generated plots

    # Images to visualize (these should be present in your GT_folder_path)
    images_to_visualize = [
        'baby.png',
        'bird.png',
        'butterfly.png',
        'head.png',
        'woman.png'
    ]

    for img_name in images_to_visualize:
        print(f"\nGenerating visualizations for {img_name}...")
        visualize_single_image_comparison(
            img_name,
            gt_folder_path,
            lr_input_folder,
            bicubic_output_folder,
            hat_psnr_output_folder,
            real_hat_gan_output_folder,
            output_comparison_figures_dir
        )
    print("\nAll visualizations generated. Check the folder:", output_comparison_figures_dir)