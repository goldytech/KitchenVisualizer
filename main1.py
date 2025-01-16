import cv2
import numpy as np
from ultralytics import YOLO
from skimage.exposure import match_histograms
import os

def extract_high_frequency_details(src, sigma=5.0):
    """
    Extracts high-frequency details from an image by subtracting a Gaussian-blurred version from the original.

    Args:
        src (np.ndarray): Source image.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: High-frequency detail layer.
    """
    src_f = src.astype(np.float32)
    blur_f = cv2.GaussianBlur(src_f, (0,0), sigmaX=sigma, sigmaY=sigma)
    detail = src_f - blur_f
    return detail

def add_high_frequency_details(base, detail):
    """
    Adds high-frequency details to a base image and clamps the result to [0..255].

    Args:
        base (np.ndarray): Base image.
        detail (np.ndarray): Detail layer to add.

    Returns:
        np.ndarray: Image with added details.
    """
    out = base + detail
    return np.clip(out, 0, 255)

def replace_variant_all_instances(
        input_image_path: str,
        model_path: str,
        class_name: str,
        texture_path: str,
        variant_name: str,
        output_dir: str = ".",
        apply_histogram_matching: bool = False,
        apply_reflection: bool = False,
        morph_close_ksize: int = 5,
        morph_close_iter: int = 1,
        morph_open_ksize: int = 5,
        morph_open_iter: int = 1,
        dilation_ksize: int = 5,
        dilation_iter: int = 1,
        feather_ksize: int = 7,
        reflection_sigma: float = 5.0
):
    """
    Replaces all detected instances of a specified class in an image with a given texture.

    Args:
        input_image_path (str): Path to the input image.
        model_path (str): Path to the YOLO model.
        class_name (str): Class name to replace.
        texture_path (str): Path to the texture image.
        variant_name (str): Name for the output variant.
        output_dir (str): Directory to save the output image.
        apply_histogram_matching (bool): Whether to apply histogram matching.
        apply_reflection (bool): Whether to apply reflection.
        morph_close_ksize (int): Kernel size for morphological closing.
        morph_close_iter (int): Number of iterations for morphological closing.
        morph_open_ksize (int): Kernel size for morphological opening.
        morph_open_iter (int): Number of iterations for morphological opening.
        dilation_ksize (int): Kernel size for dilation.
        dilation_iter (int): Number of iterations for dilation.
        feather_ksize (int): Kernel size for feathering.
        reflection_sigma (float): Sigma for reflection detail extraction.

    Raises:
        RuntimeError: If no detection results are found.
        FileNotFoundError: If the input image or texture image cannot be loaded.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) YOLO Inference
    model = YOLO(model_path)
    results = model(input_image_path)
    if len(results) == 0:
        raise RuntimeError("No detection results.")

    # 2) Load Original
    original_img = cv2.imread(input_image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not read {input_image_path}")
    H, W, _ = original_img.shape

    # We'll keep a float copy for final compositing
    replaced_img = original_img.astype(np.float32)

    # 3) Loop over each instance
    #    Instead of merging masks, we process them one by one
    instance_count = 0

    for (mask_tensor, cls_idx) in zip(results[0].masks.data, results[0].boxes.cls):
        detected_class = model.names[int(cls_idx)]
        if detected_class != class_name:
            continue  # skip other classes

        instance_count += 1

        # Convert YOLO mask [0..1] -> [0..255]
        mask_np = mask_tensor.cpu().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        # Resize mask to match original image
        mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Morphological close + open
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close, iterations=morph_close_iter)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open, iterations=morph_open_iter)

        # Optional Dilation to expand edges
        if dilation_ksize > 1 and dilation_iter > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_ksize, dilation_ksize))
            mask_bin = cv2.dilate(mask_bin, kernel_dilate, iterations=dilation_iter)

        # Find contours for this instance
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue  # nothing to replace for this instance

        # For each connected region in this instance mask
        # (Sometimes a single instance can have multiple disjoint parts, though it's rare.)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # skip tiny noise
                continue

            # Approx polygon
            approx_eps = 0.01 * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, approx_eps, True)

            # boundingRect
            x, y, w, h = cv2.boundingRect(approx_poly)
            if w < 2 or h < 2:
                continue

            # 4) Prepare the texture for this boundingRect
            texture_img = cv2.imread(texture_path)
            if texture_img is None:
                raise FileNotFoundError(f"Could not load texture: {texture_path}")

            # Resize texture to match boundingRect
            texture_resized = cv2.resize(texture_img, (w, h), interpolation=cv2.INTER_CUBIC)

            replaced_roi_f32 = texture_resized.astype(np.float32)

            # Optional reflection
            if apply_reflection:
                original_roi = original_img[y:y+h, x:x+w]
                detail_layer = extract_high_frequency_details(original_roi, sigma=reflection_sigma)
                detail_layer_f32 = detail_layer.astype(np.float32)
                if detail_layer_f32.shape[:2] != replaced_roi_f32.shape[:2]:
                    detail_layer_f32 = cv2.resize(detail_layer_f32, (w, h), interpolation=cv2.INTER_LINEAR)
                replaced_roi_f32 = add_high_frequency_details(replaced_roi_f32, detail_layer_f32)

            # Optional histogram matching
            if apply_histogram_matching:
                original_roi = original_img[y:y+h, x:x+w]
                replaced_roi_u8 = replaced_roi_f32.astype(np.uint8)
                replaced_roi_rgb = cv2.cvtColor(replaced_roi_u8, cv2.COLOR_BGR2RGB)
                original_roi_rgb = cv2.cvtColor(original_roi, cv2.COLOR_BGR2RGB)
                matched_rgb = match_histograms(replaced_roi_rgb, original_roi_rgb, channel_axis=-1)
                matched_bgr = cv2.cvtColor(matched_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                replaced_roi_f32 = matched_bgr.astype(np.float32)

            # Warp boundingRect to boundingRect in final image
            src_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
            dst_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            warped_full = cv2.warpPerspective(
                replaced_roi_f32,
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Build polygon mask for approx_poly
            poly_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(poly_mask, [approx_poly], -1, color=255, thickness=-1)

            # Feather edges
            if feather_ksize > 1:
                poly_mask = cv2.GaussianBlur(poly_mask, (feather_ksize, feather_ksize), 0)

            alpha = (poly_mask.astype(np.float32)/255.0)[...,None]  # shape (H,W,1)

            # Alpha blend
            replaced_img = alpha * warped_full + (1 - alpha) * replaced_img

    # 5) After processing all instances, save final
    if instance_count == 0:
        # Means we didn't find the class at all
        print(f"No '{class_name}' found. Saving original.")
        out_path = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
        cv2.imwrite(out_path, original_img)
        return

    final_out = np.clip(replaced_img, 0, 255).astype(np.uint8)
    out_path = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
    cv2.imwrite(out_path, final_out)
    print(f"Replaced {instance_count} '{class_name}' instance(s). Saved final: {out_path}")


if __name__ == "__main__":
    """
    Example usage:
      - If YOLO detects two separate floor areas, this script
        processes them both in separate loops, so you'll see
        both replaced with the new texture.

    replace_variant_all_instances(
        input_image_path="kitchen.jpg",
        model_path="best.pt",
        class_name="floor",
        texture_path="new_floor_texture.jpg",
        variant_name="multi_floor"
    )
    """
    replace_variant_all_instances(
        input_image_path="countertop_multi_floor.png",
        model_path="best.pt",
        class_name="floor",
        texture_path="black_floor.jpg",
        variant_name="multi_floor",
        output_dir="outputs_multi_floor",
        apply_histogram_matching=False,
        apply_reflection=False,
        morph_close_ksize=7,
        morph_close_iter=1,
        morph_open_ksize=7,
        morph_open_iter=1,
        dilation_ksize=7,
        dilation_iter=2,
        feather_ksize=7,
        reflection_sigma=5.0
    )