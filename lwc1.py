import cv2
import numpy as np
from ultralytics import YOLO
from skimage.exposure import match_histograms
import os

def extract_high_frequency_details(src, sigma=5.0):
    """ Extract high-frequency (detail) layer via Gaussian blur subtraction. """
    src_f = src.astype(np.float32)
    blurred = cv2.GaussianBlur(src_f, (0,0), sigmaX=sigma, sigmaY=sigma)
    detail = src_f - blurred
    return detail

def add_high_frequency_details(base, detail):
    """ Add detail layer to base, clip to [0..255]. """
    out = base + detail
    return np.clip(out, 0, 255)

def replace_variant_polygon_light_morph(
        input_image_path: str,
        model_path: str,
        class_name: str,
        texture_path: str,
        variant_name: str,
        output_dir: str = ".",
        apply_histogram_matching: bool = False,
        apply_reflection: bool = False,
        morph_kernel_size: int = 5,
        morph_close_iterations: int = 1,
        morph_open_iterations: int = 1,
        feather_ksize: int = 7
):
    """
    1) YOLO seg => merges mask for `class_name`.
    2) Light morphological close/open with small kernel (avoid big expansions).
    3) Find contour -> approximate polygon.
    4) Warp texture boundingRect -> boundingRect of that polygon.
    5) Blend only inside polygon with a moderate feather.
    6) Optionally apply histogram matching and reflection detail.

    Saves output as {class_name}_{variant_name}.png in output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1) YOLO inference
    model = YOLO(model_path)
    results = model(input_image_path)
    if len(results) == 0:
        raise RuntimeError("No detection results found.")

    # 2) Load original
    original_img = cv2.imread(input_image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not read {input_image_path}")
    H, W, _ = original_img.shape

    replaced_img = original_img.astype(np.float32)

    # Merge mask for class_name
    merged_mask = np.zeros((H, W), dtype=np.uint8)
    for (mask_tensor, cls_idx) in zip(results[0].masks.data, results[0].boxes.cls):
        cls_name = model.names[int(cls_idx)]
        if cls_name != class_name:
            continue
        mask_np = mask_tensor.cpu().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
        merged_mask = cv2.bitwise_or(merged_mask, mask_bin)

    # If no mask found, save original
    if not np.any(merged_mask):
        out_no_mask = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
        cv2.imwrite(out_no_mask, original_img)
        print(f"No {class_name} found. Saved original: {out_no_mask}")
        return

    # 3) Light morphological close/open
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations)

    # Debug save the refined mask
    debug_mask_path = os.path.join(output_dir, f"debug_mask_{class_name}.png")
    cv2.imwrite(debug_mask_path, merged_mask)
    print(f"Debug: saved refined mask to {debug_mask_path}")

    # 4) Find largest contour (assuming we want the biggest piece if YOLO missed edges)
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        out_no_contour = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
        cv2.imwrite(out_no_contour, original_img)
        print(f"No contour found. Saved original: {out_no_contour}")
        return
    c_main = max(contours, key=cv2.contourArea)

    # Optionally, if you have only one countertop, ignoring smaller ones:
    # c_main = max(contours, key=cv2.contourArea)

    approx_eps = 0.01 * cv2.arcLength(c_main, True)
    approx_poly = cv2.approxPolyDP(c_main, approx_eps, True)

    # 5) boundingRect around that polygon
    x, y, w, h = cv2.boundingRect(approx_poly)
    if w < 2 or h < 2:
        out_small = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
        cv2.imwrite(out_small, original_img)
        print(f"Polygon boundingRect too small. Saved original: {out_small}")
        return

    # 6) Load texture & do optional histogram matching / reflection
    texture_img = cv2.imread(texture_path)
    if texture_img is None:
        raise FileNotFoundError(f"Could not load texture at {texture_path}")

    texture_resized = cv2.resize(texture_img, (w, h), interpolation=cv2.INTER_CUBIC)

    original_roi = original_img[y:y+h, x:x+w].copy()

    # Reflection detail?
    replaced_roi_f32 = texture_resized.astype(np.float32)
    if apply_reflection:
        detail_layer = extract_high_frequency_details(original_roi, sigma=5.0)
        detail_layer_f32 = detail_layer.astype(np.float32)
        if detail_layer_f32.shape[:2] != replaced_roi_f32.shape[:2]:
            detail_layer_f32 = cv2.resize(detail_layer_f32, (w, h))
        replaced_roi_f32 = add_high_frequency_details(replaced_roi_f32, detail_layer_f32)

    # Histogram matching?
    if apply_histogram_matching:
        # Convert to RGB for skimage
        replaced_roi_rgb = cv2.cvtColor(replaced_roi_f32.astype(np.uint8), cv2.COLOR_BGR2RGB)
        original_roi_rgb = cv2.cvtColor(original_roi, cv2.COLOR_BGR2RGB)
        matched_rgb = match_histograms(
            replaced_roi_rgb,
            original_roi_rgb,
            channel_axis=-1
        )
        matched_bgr = cv2.cvtColor(matched_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        replaced_roi_f32 = matched_bgr.astype(np.float32)

    # 7) Warp boundingRect of replaced_roi_f32 -> boundingRect in final image
    src_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    dst_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_full = cv2.warpPerspective(
        replaced_roi_f32, M, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # 8) Build a polygon mask for approx_poly, feather it
    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(poly_mask, [approx_poly], -1, color=255, thickness=-1)

    poly_mask = cv2.GaussianBlur(poly_mask, (feather_ksize, feather_ksize), 0)
    alpha = (poly_mask.astype(np.float32) / 255.0)[...,None]

    # 9) Alpha blend
    replaced_img = alpha * warped_full + (1.0 - alpha) * replaced_img

    # 10) Save final
    out_path = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
    final_out = np.clip(replaced_img, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, final_out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    """
    Example usage:

    replace_variant_polygon_light_morph(
        input_image_path="kitchen.jpg",
        model_path="best.pt",
        class_name="countertop",
        texture_path="porttolo.jpg",
        variant_name="light_polygon",
        output_dir="outputs_polygon",
        apply_histogram_matching=False, # you can test True or False
        apply_reflection=False,         # test True or False
        morph_kernel_size=5,
        morph_close_iterations=1,
        morph_open_iterations=1,
        feather_ksize=7
    )
    """
    replace_variant_polygon_light_morph(
        input_image_path="kitchen.jpg",
        model_path="best.pt",
        class_name="countertop",
        texture_path="porrtolo.jpeg",
        variant_name="light_polygon",
        output_dir="outputs_polygon",
        apply_histogram_matching=False,
        apply_reflection=False,
        morph_kernel_size=5,
        morph_close_iterations=1,
        morph_open_iterations=1,
        feather_ksize=7
    )