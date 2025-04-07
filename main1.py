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
    Function performs mask extraction, processing, and texture replacement for each instance.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1) YOLO Inference - Load model and detect objects in the image
    model = YOLO(model_path)
    results = model(input_image_path)
    if len(results) == 0:
        raise RuntimeError("No detection results.")

    # 2) Load Original image using OpenCV
    # cv2.imread(): Reads an image from file into a numpy array in BGR color order
    original_img = cv2.imread(input_image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not read {input_image_path}")
    H, W, _ = original_img.shape  # Get dimensions for later processing

    # Convert to float32 for precise blending calculations (allows values outside 0-255 range during processing)
    replaced_img = original_img.astype(np.float32)

    # 3) Process each detected instance individually
    instance_count = 0

    for (mask_tensor, cls_idx) in zip(results[0].masks.data, results[0].boxes.cls):
        detected_class = model.names[int(cls_idx)]
        if detected_class != class_name:
            continue  # Skip objects that aren't our target class

        instance_count += 1

        # Convert YOLO mask tensor (values 0-1) to numpy array with values 0-255
        mask_np = mask_tensor.cpu().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # cv2.resize(): Resizes the mask to match image dimensions
        # INTER_NEAREST preserves hard edges without interpolation artifacts
        mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # cv2.threshold(): Converts grayscale mask to binary (only 0 or 255 values)
        # THRESH_BINARY: Values above threshold become 255, below become 0
        _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up the mask
        # cv2.getStructuringElement(): Creates a kernel of specified shape and size
        # MORPH_ELLIPSE: Elliptical kernel helps create more natural, rounded mask edges
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        
        # cv2.morphologyEx(MORPH_CLOSE): Performs dilation followed by erosion
        # Purpose: Closes small holes in the foreground, smoothing boundaries and filling gaps
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close, iterations=morph_close_iter)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        
        # cv2.morphologyEx(MORPH_OPEN): Performs erosion followed by dilation
        # Purpose: Removes small noise and protrusions in the mask
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open, iterations=morph_open_iter)

        # Optional Dilation to expand mask edges
        # cv2.dilate(): Expands white regions in the image
        # Purpose: Extends mask slightly beyond object boundaries for better blending
        # Dilation in image processing is a morphological operation that expands white regions (foreground) in images
        # by adding pixels to object boundaries.
        if dilation_ksize > 1 and dilation_iter > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_ksize, dilation_ksize))
            mask_bin = cv2.dilate(mask_bin, kernel_dilate, iterations=dilation_iter)

        # cv2.findContours(): Finds boundaries of connected regions in binary image
        # RETR_EXTERNAL: Only finds outermost contours
        # CHAIN_APPROX_SIMPLE: Compresses horizontal/vertical/diagonal segments to endpoints
        # In OpenCV, contours are curves that join continuous points along the boundary of shapes with the same color or intensity. 
        # They represent the outlines of objects or regions in an image.
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue  # Skip if no valid contours found

        # Process each connected region within this instance mask
        for contour in contours:
            # cv2.contourArea(): Calculates area enclosed by contour
            # Purpose: Filter out tiny noise regions that aren't worth processing
            area = cv2.contourArea(contour)
            if area < 100:  # Skip areas smaller than 100px²
                continue

            # cv2.arcLength(): Calculates perimeter of contour
            # True parameter means contour is closed
            approx_eps = 0.01 * cv2.arcLength(contour, True)
            
            # cv2.approxPolyDP(): Approximates contour with fewer vertices
            # Purpose: Simplifies the contour while preserving its shape
            approx_poly = cv2.approxPolyDP(contour, approx_eps, True)

            # cv2.boundingRect(): Gets the minimal upright rectangle containing the contour
            # Returns (x,y) of top-left corner, width and height
            x, y, w, h = cv2.boundingRect(approx_poly)
            if w < 2 or h < 2:  # Skip extremely small regions
                continue

            # 4) Prepare texture for replacement
            texture_img = cv2.imread(texture_path)
            if texture_img is None:
                raise FileNotFoundError(f"Could not load texture: {texture_path}")

            # Resize texture to match the bounding rectangle dimensions
            # INTER_CUBIC: High-quality interpolation for superior visual results
            texture_resized = cv2.resize(texture_img, (w, h), interpolation=cv2.INTER_CUBIC)

            replaced_roi_f32 = texture_resized.astype(np.float32)

            # Optional: Preserve reflections and details from original image
            if apply_reflection:
                # Extract region of interest from original image
                original_roi = original_img[y:y+h, x:x+w]
                
                # Extract high-frequency details (reflections, texture patterns)
                detail_layer = extract_high_frequency_details(original_roi, sigma=reflection_sigma)
                detail_layer_f32 = detail_layer.astype(np.float32)
                
                # Ensure dimensions match if needed
                if detail_layer_f32.shape[:2] != replaced_roi_f32.shape[:2]:
                    detail_layer_f32 = cv2.resize(detail_layer_f32, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Add high-frequency details to replacement texture for more realistic results
                replaced_roi_f32 = add_high_frequency_details(replaced_roi_f32, detail_layer_f32)

            # Optional: Match histogram to original for consistent lighting/color
            if apply_histogram_matching:
                original_roi = original_img[y:y+h, x:x+w]
                replaced_roi_u8 = replaced_roi_f32.astype(np.uint8)
                
                # Convert to RGB for histogram matching (skimage works with RGB)
                replaced_roi_rgb = cv2.cvtColor(replaced_roi_u8, cv2.COLOR_BGR2RGB)
                original_roi_rgb = cv2.cvtColor(original_roi, cv2.COLOR_BGR2RGB)
                
                # match_histograms(): Adjusts color and intensity distribution of texture to match original
                # Purpose: Maintains consistent lighting appearance between original and replacement
                matched_rgb = match_histograms(replaced_roi_rgb, original_roi_rgb, channel_axis=-1)
                matched_bgr = cv2.cvtColor(matched_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                replaced_roi_f32 = matched_bgr.astype(np.float32)

            # Prepare source and destination points for perspective transformation
            src_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
            dst_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
            
            # cv2.getPerspectiveTransform(): Calculates transformation matrix between two quadrilaterals
            # Purpose: Creates mapping from texture coordinates to target position in image
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # cv2.warpPerspective(): Applies perspective transformation to an image
            # INTER_LINEAR: Bilinear interpolation for smooth result
            # BORDER_REFLECT_101: Handles edges by mirroring pixels at boundaries
            warped_full = cv2.warpPerspective(
                replaced_roi_f32,
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Create mask from polygon for precise blending
            poly_mask = np.zeros((H, W), dtype=np.uint8)
            
            # cv2.drawContours(): Fills the polygon defined by approx_poly
            # -1 means fill all contours, color=255 is white fill, thickness=-1 means fill interior
            cv2.drawContours(poly_mask, [approx_poly], -1, color=255, thickness=-1)

            # Feather edges for smoother blending
            if feather_ksize > 1:
                # cv2.GaussianBlur(): Applies Gaussian smoothing to the mask
                # Purpose: Creates soft, graduated transitions at edges for natural blending
                poly_mask = cv2.GaussianBlur(poly_mask, (feather_ksize, feather_ksize), 0)

            # Convert mask to alpha channel (0.0-1.0 range, 3D shape for broadcasting)
            alpha = (poly_mask.astype(np.float32)/255.0)[...,None]  # shape (H,W,1)

            # Alpha blend: result = alpha*new + (1-alpha)*original
            replaced_img = alpha * warped_full + (1 - alpha) * replaced_img

    # After processing all instances, save final result
    if instance_count == 0:
        print(f"No '{class_name}' found. Saving original.")
        out_path = os.path.join(output_dir, f"{class_name}_{variant_name}.png")
        cv2.imwrite(out_path, original_img)
        return

    # cv2.imwrite(): Saves image to file
    # np.clip() ensures pixel values stay within valid 0-255 range
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
        input_image_path="MultisurfaceDemo.jpg",
        model_path="best.pt",
        class_name="countertop",
        texture_path="countertopvar1.png",
        variant_name="multi_countertop",
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