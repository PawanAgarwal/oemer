import cv2
import numpy as np


def detect_notes(image_path: str):
    """Simple heuristic-based notehead detection."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = w / float(h)
        if area < 20:
            continue
        if 0.5 < aspect < 1.5 and 10 < w < 100 and 10 < h < 100:
            bboxes.append((x, y, w, h))

    for x, y, w, h in bboxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out_path = image_path.replace(".png", "_detected.png")
    cv2.imwrite(out_path, img)
    return out_path, bboxes


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python improved_note_detection.py <image>")

    res, boxes = detect_notes(sys.argv[1])
    print(f"Detected {len(boxes)} noteheads. Output saved to {res}")
