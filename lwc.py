import cv2
import numpy as np
from ultralytics import YOLO

def replace_variant_in_image(
        input_image_path: str,
        model_path: str,
        variant_to_replace: str,
        texture_path: str,
        output_image_path: str,
        do_color_adjustment: bool = True
):
    """
    1) Runs YOLO instance segmentation to get the mask(s) for 'variant_to_replace' (e.g. 'countertop').
    2) Merges multiple instance masks of that class.
    3) Finds contours (polygons) of the merged mask.
    4) For each polygon, warps the texture to fit that shape, then blends it onto the original image.
    5) Optionally adjusts the texture color to better match the average color of the old countertop region.
    6) Saves the final composited result to output_image_path.
    """

    # ------------------ 1. Load YOLO model & Run Inference ------------------
    model = YOLO(model_path)
    results = model(input_image_path)
    if len(results) == 0:
        raise RuntimeError("No detection results found.")

    # ------------------ 2. Load original image & texture -------------------
    input_img = cv2.imread(input_image_path)
    if input_img is None:
        raise FileNotFoundError(f"Could not load {input_image_path}")
    texture_img = cv2.imread(texture_path)
    if texture_img is None:
        raise FileNotFoundError(f"Could not load {texture_path}")

    H, W, _ = input_img.shape

    # ------------------ 3. Merge masks for the requested variant -----------
    class_masks = {}
    for (mask_tensor, cls_idx) in zip(results[0].masks.data, results[0].boxes.cls):
        cls_name = model.names[int(cls_idx)]
        if cls_name != variant_to_replace:
            continue  # only care about the variant we want to replace

        # Convert to numpy, resize to match original image
        mask_np = mask_tensor.cpu().numpy()  # [0..1]
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
        _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Merge into a single mask per class
        if variant_to_replace not in class_masks:
            class_masks[variant_to_replace] = np.zeros((H, W), dtype=np.uint8)
        class_masks[variant_to_replace] = cv2.bitwise_or(class_masks[variant_to_replace], mask_resized)

    # If no masks found for that variant, just return
    if variant_to_replace not in class_masks:
        print(f"No instances of '{variant_to_replace}' found.")
        cv2.imwrite(output_image_path, input_img)
        return

    merged_mask = class_masks[variant_to_replace]  # shape (H, W)

    # Make a copy of the original for final compositing
    replaced_img = input_img.copy()

    # -----------------------------------------------------------------------
    # 4. Find polygons (contours) from the merged mask
    #    We'll handle each polygon separately. Each polygon should represent
    #    one or more connected “countertop” region(s).
    # -----------------------------------------------------------------------
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Optional: approximate contours to reduce the number of points
    # e.g. use cv2.approxPolyDP(contour, epsilon=..., closed=True)

    # For each contour, we do:
    #   - Warping the texture to the bounding rectangle or polygon
    #   - Doing edge feathering
    #   - Optionally doing color matching
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            # skip tiny artifacts
            continue

        # This polygon might be any shape. For a simple approach:
        # 1) Use boundingRect or bounding quadrilateral
        # 2) Perspective transform the texture to fit that shape

        # Compute bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Crop that region from the input image to measure average color (optional color matching)
        countertop_region = input_img[y:y+h, x:x+w]

        #-------------------------------------------
        # (A) Basic color correction (optional)
        #-------------------------------------------
        if do_color_adjustment:
            texture_patch = texture_img.copy()

            # Compute average color in the old countertop region
            old_mean = cv2.mean(countertop_region)  # (B, G, R, alpha)
            old_bgr = old_mean[:3]
            # Compute average color in the full texture (or a patch)
            text_mean = cv2.mean(texture_patch)
            text_bgr = text_mean[:3]

            # We'll do a simple ratio-based adjustment to match the brightness
            # More advanced methods handle gamma, hue, etc.
            # Avoid division by zero if text_bgr is near 0
            scale_b = (old_bgr[0] + 1) / (text_bgr[0] + 1)
            scale_g = (old_bgr[1] + 1) / (text_bgr[1] + 1)
            scale_r = (old_bgr[2] + 1) / (text_bgr[2] + 1)

            # Scale the entire texture
            # This might produce out-of-range values, so we'll clamp
            texture_patch = texture_patch.astype(np.float32)
            texture_patch[..., 0] *= scale_b
            texture_patch[..., 1] *= scale_g
            texture_patch[..., 2] *= scale_r

            # Clip to [0, 255]
            texture_patch = np.clip(texture_patch, 0, 255).astype(np.uint8)

            # Use texture_patch for the next step
            used_texture = texture_patch
        else:
            used_texture = texture_img

        #-------------------------------------------
        # (B) Perspective transform
        #    We'll get the boundingRect corners and do a 4-point warp.
        #    This is approximate if the shape is not exactly rectangular.
        #-------------------------------------------
        # Destination corners = the contour bounding box in the output image
        dst_pts = np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h]
        ], dtype=np.float32)

        # Source corners = corners of the texture image
        # We'll just use the entire texture for demonstration,
        # But a real workflow might pick a sub-rectangle from the texture
        th, tw, _ = used_texture.shape
        src_pts = np.array([
            [0, 0],
            [tw, 0],
            [tw, th],
            [0,  th]
        ], dtype=np.float32)

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Warp the entire texture to the boundingRect size
        warped_texture = cv2.warpPerspective(used_texture, M, (W, H))  # full image size

        #-------------------------------------------
        # (C) Create a mask just for this contour (feathered edges)
        #-------------------------------------------
        contour_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, color=255, thickness=-1)

        # Feather the contour edges
        # For example, blur or erode/dilate:
        contour_mask = cv2.GaussianBlur(contour_mask, (7, 7), 0)

        # Make a 3-channel mask for blending
        contour_mask_3c = cv2.merge([contour_mask, contour_mask, contour_mask])

        # Convert to float for alpha blending
        alpha = contour_mask_3c.astype(np.float32) / 255.0  # range [0..1]

        #-------------------------------------------
        # (D) Blend warped texture with replaced_img
        #-------------------------------------------
        warped_texture_f32 = warped_texture.astype(np.float32)
        replaced_img_f32 = replaced_img.astype(np.float32)

        # alpha-blending formula:
        # out = alpha * newTexture + (1 - alpha) * old
        blended = alpha * warped_texture_f32 + (1.0 - alpha) * replaced_img_f32
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Only update replaced_img where the mask > 0
        replaced_img = blended

    # ------------------ 5. Save final output -------------------
    cv2.imwrite(output_image_path, replaced_img)
    print(f"Saved replaced image to: {output_image_path}")


if __name__ == "__main__":
    """
    Example usage:
    - Replace 'countertop' in 'kitchen.jpg' using the YOLO model 'best.pt'
    - Use 'porttolo.jpg' as the texture
    - Save final to 'kitchen_replaced.jpg'
    - do_color_adjustment = True to attempt a basic brightness match
    """
    replace_variant_in_image(
        input_image_path="kitchen.jpg",
        model_path="best.pt",
        variant_to_replace="countertop",
        texture_path="porrtolo.jpeg",
        output_image_path="kitchen_replaced.jpg",
        do_color_adjustment=True
    )