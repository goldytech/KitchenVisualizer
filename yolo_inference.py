import cv2
import numpy as np
from ultralytics import YOLO

def save_inference_result_image_no_boxes(
        input_image_path: str,
        model_path: str,
        output_image_path: str = "inference_result.jpg",
        confidence_threshold: float = 0.5
):
    """
    Runs YOLO inference on `input_image_path` using the YOLO model at `model_path`.
    Then saves an annotated image to `output_image_path` that:
      - Only shows class labels (and their confidence) if confidence > `confidence_threshold`
      - Does NOT draw bounding boxes or polygons around objects.
    """

    # 1) Load the model
    model = YOLO(model_path)

    # 2) Load the image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {input_image_path}")
    H, W, _ = img.shape

    # 3) Inference
    results = model(input_image_path)  # results[0] for single image

    if len(results) == 0:
        raise RuntimeError("No YOLO inference results returned.")

    # 4) Iterate over each detected box
    #    YOLOv8 typically has results[0].boxes.xyxy, .conf, .cls
    boxes = results[0].boxes  # Boxes object with xyxy, conf, cls
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if conf < confidence_threshold:
            # skip low-confidence detections
            continue

        class_name = model.names[cls_id]

        # Coordinates are [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 5) Put text at an appropriate location
        #    We'll choose top-left corner above the bounding box.
        label = f"{class_name}"
        font_scale = 0.5
        color = (0, 0, 255)  # Red color in BGR
        thickness = 1  # Bold text

        text_pos = (x1, max(0, y1 - 5))  # Slightly above the box

        cv2.putText(
            img,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA
        )

        # We do NOT draw any rectangle or mask here—only the text.

    # 6) Save the annotated image
    cv2.imwrite(output_image_path, img)
    print(f"Saved inference result (no boxes) to: {output_image_path}")


if __name__ == "__main__":
    # Example usage
    save_inference_result_image_no_boxes(
        input_image_path="kitchen.jpg",
        model_path="best.pt",
        output_image_path="inference_result.jpg",
        confidence_threshold=0.5
    )