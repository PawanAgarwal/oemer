"""Very small heuristic-based note head detection using OpenCV.

This script was added as a quick experiment and isn't part of the main
``oemer`` package.  The logic attempts to clean up the input score image
using a few morphological operations and then extracts note heads by
finding contours with a roughly circular aspect ratio.  It still relies on
``cv2`` and ``numpy`` at runtime.
"""

import cv2
import numpy as np


def detect_notes(image_path: str):
    """Detect note heads in ``image_path``.

    The current heuristic removes staff lines, cleans noise and then looks for
    near-circular contours within a reasonable size range.  Detected
    bounding boxes are returned and also drawn on the output image.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarise using Otsu threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Remove most staff lines by opening with a long horizontal kernel
    staff_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    staff_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, staff_kernel, iterations=1)
    no_staff = cv2.subtract(thresh, staff_lines)

    # Smooth noise and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(no_staff, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = w / float(h)

        if area < 30:
            continue
        if not 0.7 <= aspect <= 1.3:
            continue
        if w < 5 or h < 5 or w > 120 or h > 120:
            continue

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
