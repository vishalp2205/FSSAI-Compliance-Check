"""
Image Preprocessing Module

Implements OpenCV-based preprocessing for food label images:
  1. Deskewing (rotation correction)
  2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. Denoising
  4. Resize normalization
"""

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for food label images.

    Args:
        image: BGR image (numpy array)

    Returns:
        Preprocessed image optimized for OCR
    """
    # Step 1: Resize if too large (keep max dimension at 2000px)
    image = _resize_if_needed(image, max_dim=2000)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Deskew
    gray = _deskew(gray)

    # Step 4: Apply CLAHE for contrast enhancement
    gray = _apply_clahe(gray)

    # Step 5: Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 6: Adaptive thresholding for cleaner text
    # We return the CLAHE image (not thresholded) as OCR engines handle this better
    return gray


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess specifically for OCR text extraction.
    Returns a clean grayscale image.
    """
    gray = preprocess_image(image)
    return gray


def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Preprocess for symbol/logo detection.
    Returns the color image with contrast enhancement.
    """
    image = _resize_if_needed(image, max_dim=1600)

    # Apply CLAHE on each channel separately
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def _resize_if_needed(image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """Resize image if any dimension exceeds max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Deskew (rotate) the image to correct for slight rotations.
    Uses Hough Line Transform to detect dominant line angle.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 8,
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        return gray

    # Calculate angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (±30 degrees)
        if -30 < angle < 30:
            angles.append(angle)

    if not angles:
        return gray

    # Use the median angle for rotation
    median_angle = np.median(angles)

    # Only correct if skew is noticeable but not extreme
    if abs(median_angle) < 0.5 or abs(median_angle) > 15:
        return gray

    # Rotate to correct the skew
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This significantly improves text visibility in uneven lighting conditions.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced
