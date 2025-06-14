import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# --- Metric Calculation Functions ---

def calculate_image_metrics(img_path_hr, img_path_test):
    """
    Calculates PSNR and SSIM between a test image and a high-resolution ground truth image.

    Args:
        img_path_hr (str): Path to the high-resolution ground truth image.
        img_path_test (str): Path to the test image (e.g., bicubic, HAT, Real-HAT-GAN output).

    Returns:
        tuple: (PSNR value, SSIM value) or (None, None) if images cannot be loaded or SSIM fails.
    """
    img_hr = cv2.imread(img_path_hr, cv2.IMREAD_COLOR)
    img_test = cv2.imread(img_path_test, cv2.IMREAD_COLOR)

    if img_hr is None:
        print(f"Error: Could not load HR image from {img_path_hr}")
        return None, None
    if img_test is None:
        print(f"Error: Could not load test image from {img_path_test}")
        return None, None

    if img_hr.shape != img_test.shape:
        print(
            f"Warning: Image shapes do not match for {os.path.basename(img_path_hr)} and {os.path.basename(img_path_test)}")
        print(f"  HR shape: {img_hr.shape}, Test shape: {img_test.shape}. Resizing test image.")
        img_test = cv2.resize(img_test, (img_hr.shape[1], img_hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        print(f"  Test image resized to {img_test.shape} for metric calculation.")

    img_hr = img_hr.astype(np.float64)
    img_test = img_test.astype(np.float64)

    current_psnr = psnr(img_hr, img_test, data_range=255)

    try:
        current_ssim = ssim(img_hr, img_test,
                            data_range=255,
                            multichannel=True,
                            channel_axis=2,
                            win_size=7)
    except ValueError as e:
        print(f"SSIM calculation failed for {os.path.basename(img_path_hr)} (shape: {img_hr.shape}): {e}")
        return current_psnr, None

    return current_psnr, current_ssim


# --- Visualization Functions ---

def plot_per_image_metrics(image_names, psnr_data_per_method, ssim_data_per_method, methods_names, save_dir):
    """
    Plots PSNR and SSIM for all images, with separate lines for each method.
    X-axis: Image names, Y-axis: Metrics, with data labels.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x_pos = np.arange(len(image_names))  # Positions for image names on x-axis

    # --- PSNR Plot ---
    fig_psnr, ax_psnr = plt.subplots(figsize=(12, 7))

    for method_key, method_name in zip(['bicubic', 'hat_psnr', 'real_hat_gan'], methods_names):
        psnrs = psnr_data_per_method[method_key]
        # Filter out NaN values for plotting line, but keep positions for labels
        valid_x = [x_pos[i] for i, val in enumerate(psnrs) if not np.isnan(val)]
        valid_psnrs = [val for val in psnrs if not np.isnan(val)]

        ax_psnr.plot(valid_x, valid_psnrs, marker='o', linestyle='-', label=method_name)

        # Add data labels
        for i, val in enumerate(psnrs):
            if not np.isnan(val):
                ax_psnr.annotate(f'{val:.2f}', (x_pos[i], val), textcoords="offset points", xytext=(0, 10), ha='center',
                                 fontsize=8)

    ax_psnr.set_xlabel('Image Name', fontsize=12)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=12)
    ax_psnr.set_title('PSNR Comparison Across Images on Set5', fontsize=14)
    ax_psnr.set_xticks(x_pos)
    ax_psnr.set_xticklabels(image_names, rotation=45, ha='right', fontsize=10)
    ax_psnr.grid(axis='y', linestyle='--', alpha=0.7)
    ax_psnr.legend(loc='lower left')
    plt.tight_layout()
    save_path_psnr = os.path.join(save_dir, 'overall_psnr_comparison_line_chart.png')
    plt.savefig(save_path_psnr, dpi=300)
    print(f"Overall PSNR line chart saved to: {save_path_psnr}")
    plt.close(fig_psnr)

    # --- SSIM Plot ---
    fig_ssim, ax_ssim = plt.subplots(figsize=(12, 7))

    for method_key, method_name in zip(['bicubic', 'hat_psnr', 'real_hat_gan'], methods_names):
        ssims = ssim_data_per_method[method_key]
        valid_x = [x_pos[i] for i, val in enumerate(ssims) if not np.isnan(val)]
        valid_ssims = [val for val in ssims if not np.isnan(val)]

        ax_ssim.plot(valid_x, valid_ssims, marker='x', linestyle='--', label=method_name)

        # Add data labels
        for i, val in enumerate(ssims):
            if not np.isnan(val):
                ax_ssim.annotate(f'{val:.4f}', (x_pos[i], val), textcoords="offset points", xytext=(0, -15),
                                 ha='center', fontsize=8)

    ax_ssim.set_xlabel('Image Name', fontsize=12)
    ax_ssim.set_ylabel('SSIM', fontsize=12)
    ax_ssim.set_title('SSIM Comparison Across Images on Set5', fontsize=14)
    ax_ssim.set_xticks(x_pos)
    ax_ssim.set_xticklabels(image_names, rotation=45, ha='right', fontsize=10)
    ax_ssim.set_ylim(0.0, 1.0)  # SSIM ranges from 0 to 1
    ax_ssim.grid(axis='y', linestyle='--', alpha=0.7)
    ax_ssim.legend(loc='lower left')
    plt.tight_layout()
    save_path_ssim = os.path.join(save_dir, 'overall_ssim_comparison_line_chart.png')
    plt.savefig(save_path_ssim, dpi=300)
    print(f"Overall SSIM line chart saved to: {save_path_ssim}")
    plt.close(fig_ssim)


def plot_average_metrics(avg_psnr_data, avg_ssim_data, methods_names, save_dir):
    """
    Plots average PSNR and SSIM for different super-resolution methods using a bar chart.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bar_width = 0.35
    index = np.arange(len(methods_names))

    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_psnr = ax1.bar(index - bar_width / 2, avg_psnr_data, bar_width, label='Average PSNR (dB)', color='skyblue')
    ax1.set_xlabel('Super-Resolution Method', fontsize=12)
    ax1.set_ylabel('Average PSNR (dB)', color='skyblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_ylim(np.nanmin(avg_psnr_data) - 2, np.nanmax(avg_psnr_data) + 2)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    bar_ssim = ax2.bar(index + bar_width / 2, avg_ssim_data, bar_width, label='Average SSIM', color='lightcoral')
    ax2.set_ylabel('Average SSIM', color='lightcoral', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    ax2.set_ylim(np.nanmin(avg_ssim_data) - 0.1, 1.0)

    ax1.set_xticks(index)
    ax1.set_xticklabels(methods_names, fontsize=10)

    plt.title('Average PSNR and SSIM Comparison on Set5 Dataset', fontsize=14)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.05, 0.95))

    def autolabel_dual_axis(bars, ax_obj, is_psnr=True):
        for bar in bars:
            height = bar.get_height()
            ax_obj.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if is_psnr else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if is_psnr else 'top',
                            fontsize=9, color='black')

    autolabel_dual_axis(bar_psnr, ax1, is_psnr=True)
    autolabel_dual_axis(bar_ssim, ax2, is_psnr=False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'average_metrics_comparison_bar_chart.png')
    plt.savefig(save_path, dpi=300)
    print(f"Average metrics bar chart saved to: {save_path}")
    plt.show()
    plt.close(fig)


# --- Main Evaluation and Visualization Logic ---
def run_full_evaluation_and_visualization(
        gt_folder, bicubic_folder, hat_psnr_folder, real_hat_gan_folder,
        metrics_plots_output_dir='./results/metric_figures'
):
    # Ensure output directory for plots exists
    os.makedirs(metrics_plots_output_dir, exist_ok=True)

    image_names = sorted([
        f for f in os.listdir(gt_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    if not image_names:
        print(f"No ground truth images found in {gt_folder}. Exiting evaluation.")
        return

    # To store individual image results, structured for new plot type
    # These will store lists of PSNRs/SSIMs, where each list is for one method across all images
    all_psnrs_for_methods = {'bicubic': [], 'hat_psnr': [], 'real_hat_gan': []}
    all_ssims_for_methods = {'bicubic': [], 'hat_psnr': [], 'real_hat_gan': []}

    methods_for_plot_names = ['Bicubic', 'HAT PSNR', 'Real HAT Gan']  # For plot labels
    methods_for_internal_keys = ['bicubic', 'hat_psnr', 'real_hat_gan']  # For dictionary keys

    print("\n--- Starting Full Metric Calculation and Visualization ---")
    print(f"Evaluating {len(image_names)} images.")
    print("-" * 50)

    for img_name in image_names:
        print(f"Processing image: {img_name}")

        base_name_without_ext = os.path.splitext(img_name)[0]

        path_hr = os.path.join(gt_folder, img_name)
        path_bicubic = os.path.join(bicubic_folder, base_name_without_ext + 'x4_bic.png')
        path_hat_psnr_full = os.path.join(hat_psnr_folder, base_name_without_ext + 'x4_HAT_SRx4.png')
        path_real_hat_gan_full = os.path.join(real_hat_gan_folder, base_name_without_ext + 'x4_HAT_GAN_Real_SRx4.png')

        # Calculate metrics for each method
        psnr_bic, ssim_bic = calculate_image_metrics(path_hr, path_bicubic)
        psnr_hat, ssim_hat = calculate_image_metrics(path_hr, path_hat_psnr_full)
        psnr_gan, ssim_gan = calculate_image_metrics(path_hr, path_real_hat_gan_full)

        # Store for overall average and line plotting later
        all_psnrs_for_methods['bicubic'].append(psnr_bic if psnr_bic is not None else np.nan)
        all_ssims_for_methods['bicubic'].append(ssim_bic if ssim_bic is not None else np.nan)

        all_psnrs_for_methods['hat_psnr'].append(psnr_hat if psnr_hat is not None else np.nan)
        all_ssims_for_methods['hat_psnr'].append(ssim_hat if ssim_hat is not None else np.nan)

        all_psnrs_for_methods['real_hat_gan'].append(psnr_gan if psnr_gan is not None else np.nan)
        all_ssims_for_methods['real_hat_gan'].append(ssim_gan if ssim_gan is not None else np.nan)

        # Print individual image metrics to console
        psnr_bic_str = f"{psnr_bic:.4f}" if psnr_bic is not None else 'N/A'
        ssim_bic_str = f"{ssim_bic:.4f}" if ssim_bic is not None else 'N/A'
        psnr_hat_str = f"{psnr_hat:.4f}" if psnr_hat is not None else 'N/A'
        ssim_hat_str = f"{ssim_hat:.4f}" if ssim_hat is not None else 'N/A'
        psnr_gan_str = f"{psnr_gan:.4f}" if psnr_gan is not None else 'N/A'
        ssim_gan_str = f"{ssim_gan:.4f}" if ssim_gan is not None else 'N/A'

        print(f"  Bicubic: PSNR={psnr_bic_str}, SSIM={ssim_bic_str}")
        print(f"  HAT_PSNR: PSNR={psnr_hat_str}, SSIM={ssim_hat_str}")
        print(f"  Real_HAT_GAN: PSNR={psnr_gan_str}, SSIM={ssim_gan_str}")
        print("-" * 20)

    # Plot overall line charts for PSNR and SSIM
    # These will be the two plots you requested, showing all images on X-axis
    plot_per_image_metrics(image_names, all_psnrs_for_methods, all_ssims_for_methods, methods_for_plot_names,
                           metrics_plots_output_dir)

    # Calculate and plot average metrics at the end (bar chart)
    avg_psnr_list = [np.nanmean(all_psnrs_for_methods[key]) for key in methods_for_internal_keys]
    avg_ssim_list = [np.nanmean(all_ssims_for_methods[key]) for key in methods_for_internal_keys]

    # Filter out NaN values from methods list for plotting if some methods failed entirely
    valid_avg_psnr = [val for val in avg_psnr_list if not np.isnan(val)]
    valid_avg_ssim = [val for val in avg_ssim_list if not np.isnan(val)]
    valid_methods_for_avg_plot = [methods_for_plot_names[i] for i, val in enumerate(avg_psnr_list) if not np.isnan(val)]

    print("\n--- Average Metrics ---")
    print("-" * 50)
    for i, method_name in enumerate(methods_for_plot_names):
        avg_psnr_str = f"{avg_psnr_list[i]:.4f}" if not np.isnan(avg_psnr_list[i]) else 'N/A'
        avg_ssim_str = f"{avg_ssim_list[i]:.4f}" if not np.isnan(avg_ssim_list[i]) else 'N/A'
        print(f"{method_name}: Average PSNR={avg_psnr_str} dB, Average SSIM={avg_ssim_str}")
    print("-" * 50)

    if valid_methods_for_avg_plot:  # Only plot if there is valid data
        plot_average_metrics(valid_avg_psnr, valid_avg_ssim, valid_methods_for_avg_plot, metrics_plots_output_dir)
    else:
        print("No valid average metrics to plot.")


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    # Adjust these paths to match your actual folder structure.
    # Make sure you run this script from the HAT-main project root directory.

    gt_folder_path = './datasets/GTmod4'
    bicubic_folder_path = './results/Set5/bicubic_upscaled'
    hat_psnr_output_folder = './results/HAT_SRx4/visualization/Set5/'
    real_hat_gan_output_folder = './results/HAT_GAN_Real_SRx4/visualization/Set5/'

    # Output directory for all generated metric plots
    metrics_plots_output_dir = './results/metric_figures'

    run_full_evaluation_and_visualization(
        gt_folder_path,
        bicubic_folder_path,
        hat_psnr_output_folder,
        real_hat_gan_output_folder,
        metrics_plots_output_dir
    )