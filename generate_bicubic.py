import cv2
import os
import sys


def generate_bicubic_upscaled(lr_folder, hr_gt_folder, output_folder, scale_factor=4):

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing images from LR folder: {lr_folder}")
    print(f"Looking for GT images in HR folder: {hr_gt_folder}")
    print(f"Saving bicubic upscaled images to: {output_folder}")
    print("-" * 50)

    # Get a list of all LR image files
    lr_image_files = sorted([f for f in os.listdir(lr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if not lr_image_files:
        print(f"No image files found in LR folder: {lr_folder}")
        return

    for lr_filename in lr_image_files:
        lr_path = os.path.join(lr_folder, lr_filename)
        lr_img = cv2.imread(lr_path)

        if lr_img is None:
            print(f"Warning: Could not read LR image {lr_path}. Skipping.")
            continue

        # --- Modified Logic to infer GT filename from LR filename ---
        # Assuming lr_filename is 'basenamexN.png' (e.g., 'babyx4.png')
        # We need to transform it to 'basename.png' (e.g., 'baby.png')

        base_name_without_ext, ext = os.path.splitext(lr_filename)

        # Remove the 'xN' suffix if present
        # This regex checks for 'x' followed by 1 or more digits at the end of the base name
        import re
        match = re.search(r'x\d+$', base_name_without_ext)
        if match:
            # If 'xN' is found, remove it
            gt_base_name = base_name_without_ext[:match.start()]
        else:
            # If no 'xN' suffix, assume LR filename is already the base name
            gt_base_name = base_name_without_ext

        # Reconstruct the GT filename with its original extension (assuming .png or .jpg from common datasets)
        # For simplicity, we assume GT images are also .png, but can be adjusted if GT are .jpg etc.
        gt_filename = gt_base_name + '.png'  # Most common case for HR GT. Adjust if your GTs are .jpg
        # If the GT images maintain their original extension, use 'ext' here:
        # gt_filename = gt_base_name + ext
        # However, for standard datasets like Set5, HR are often .png regardless of LR's original format.
        # So, .png is often a safe bet for GT.

        gt_path = os.path.join(hr_gt_folder, gt_filename)
        gt_img = cv2.imread(gt_path)

        if gt_img is None:
            print(
                f"Warning: Could not find GT image at expected path '{gt_path}' (derived from LR '{lr_filename}'). Skipping.")
            continue

        hr_height, hr_width, _ = gt_img.shape

        # Perform bicubic interpolation to upscale LR image to HR GT dimensions
        bicubic_upscaled_img = cv2.resize(lr_img, (hr_width, hr_height), interpolation=cv2.INTER_CUBIC)

        # Construct output filename for the bicubic result
        # Example: 'babyx4.png' -> 'babyx4_bic.png' to avoid conflict with GT or LR
        output_filename = os.path.splitext(lr_filename)[0] + '_bic.png'
        output_path = os.path.join(output_folder, output_filename)

        # Save the upscaled image
        cv2.imwrite(output_path, bicubic_upscaled_img)
        print(f"Saved bicubic upscaled image: {output_path}")

    print("-" * 50)
    print("\nBicubic upscaling complete. Check the images in:", output_folder)
    print("Please verify the generated images visually and if the GT image finding logic was correct for your dataset.")


if __name__ == '__main__':
    # --- Configuration ---
    # Make sure you are running this script from the HAT-main project root directory.

    # Low-resolution input images folder (e.g., './datasets/Set5/LRBicx4/X4')
    # This should be the folder containing files like 'babyx4.png', 'headx4.png'
    lr_input_folder = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/datasets/LRbicx4/X4/'

    # High-resolution Ground Truth images folder (e.g., './datasets/Set5/GTmod4')
    # This should be the folder containing files like 'baby.png', 'head.png'
    hr_gt_folder = '/data_C/sdb1/zyk_runs/subgraph_removal2/HAT-main/datasets/GTmod4/'

    # Output folder for the generated bicubic upscaled images
    bicubic_output_folder = './results/Set5/bicubic_upscaled'

    # Super-resolution scale factor (e.g., 4 for 4x) - Used for reference, not directly in resize here
    # as we use GT dimensions. But important for context.
    scale_factor = 4

    # --- Run the function ---
    generate_bicubic_upscaled(lr_input_folder, hr_gt_folder, bicubic_output_folder, scale_factor)