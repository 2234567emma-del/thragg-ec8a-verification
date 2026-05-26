import os
import re

import cv2
import easyocr
import numpy as np
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


easy_reader = easyocr.Reader(["en"], gpu=False)
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)

DEBUG_DIR = "static"
EXPECTED_ROW_COUNT = 8
MIN_LOCALIZATION_CONFIDENCE = 0.58
MIN_ROW_CONFIDENCE = 0.52
MIN_TROCR_RESCUE_CONFIDENCE = 0.38
RAW_STRONG_CONFIDENCE = 0.62


def clean_number_text(text):
    text = str(text)
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = re.sub(r"[^0-9]", "", text)

    if text == "":
        return None

    return int(text)


def save_debug_image(filename, image):
    if image is not None and image.size > 0:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, filename), image)


def is_valid_row_number(value):
    return value is not None and 0 <= value <= 5000


def digit_len(value):
    if value is None:
        return 0

    return len(str(abs(value)))


def threshold_for_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )


def order_points(points):
    points = np.asarray(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


def four_point_warp(image, points):
    rect = order_points(points)
    tl, tr, br, bl = rect

    max_width = int(
        max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        )
    )
    max_height = int(
        max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        )
    )

    if max_width < 30 or max_height < 90:
        return None

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def contour_points(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")

    return cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")


def horizontal_line_positions(image):
    thresh = threshold_for_lines(image)
    h, w = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, w // 2), 1))
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= w * 0.45 and height <= max(7, h * 0.06):
            positions.append((y + height // 2, x, x + width))

    positions = sorted(positions)
    merged = []
    min_gap = max(4, int(h * 0.018))

    for y, x1, x2 in positions:
        if not merged or abs(y - merged[-1][0]) > min_gap:
            merged.append([y, x1, x2])
        else:
            merged[-1][0] = int(round((merged[-1][0] + y) / 2))
            merged[-1][1] = min(merged[-1][1], x1)
            merged[-1][2] = max(merged[-1][2], x2)

    return merged


def vertical_line_positions(image):
    thresh = threshold_for_lines(image)
    h, w = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 2)))
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if height >= h * 0.55 and width <= max(8, w * 0.08):
            positions.append(x + width // 2)

    positions = sorted(positions)
    merged = []
    min_gap = max(4, int(w * 0.03))

    for x in positions:
        if not merged or abs(x - merged[-1]) > min_gap:
            merged.append(x)

    return merged


def best_eight_row_window(line_positions):
    if len(line_positions) < EXPECTED_ROW_COUNT + 1:
        return None, 0.0

    best = None
    for index in range(len(line_positions) - EXPECTED_ROW_COUNT):
        window = line_positions[index : index + EXPECTED_ROW_COUNT + 1]
        ys = np.asarray([item[0] for item in window], dtype="float32")
        gaps = np.diff(ys)

        if np.any(gaps < 10):
            continue

        median_gap = float(np.median(gaps))
        regularity = max(0.0, 1.0 - float(np.std(gaps) / max(median_gap, 1.0)))
        height = float(ys[-1] - ys[0])
        score = regularity + min(height / 220.0, 1.0)

        if best is None or score > best[0]:
            best = (score, window)

    if best is None:
        return None, 0.0

    confidence = min(1.0, best[0] / 2.0)
    return best[1], confidence


def refine_to_number_rows(warped):
    lines = horizontal_line_positions(warped)
    window, row_confidence = best_eight_row_window(lines)

    if window is None:
        return warped, row_confidence

    h, w = warped.shape[:2]
    y1 = max(0, window[0][0] - 2)
    y2 = min(h, window[-1][0] + 2)
    rough = warped[y1:y2, :]
    verticals = vertical_line_positions(rough)

    if len(verticals) >= 2:
        x1 = max(0, verticals[0] - 2)
        x2 = min(w, verticals[-1] + 2)
    else:
        x1 = max(0, min(item[1] for item in window) - 2)
        x2 = min(w, max(item[2] for item in window) + 2)

    if y2 - y1 < 80 or x2 - x1 < 40:
        return warped, row_confidence * 0.5

    return warped[y1:y2, x1:x2], row_confidence


def score_number_box_candidate(image, points):
    warped = four_point_warp(image, points)
    if warped is None:
        return None

    image_h, image_w = image.shape[:2]
    h, w = warped.shape[:2]
    aspect = h / float(max(1, w))
    area_ratio = (h * w) / float(image_h * image_w)
    center_x = np.mean(np.asarray(points)[:, 0]) / float(image_w)
    center_y = np.mean(np.asarray(points)[:, 1]) / float(image_h)

    if center_x < 0.55 or center_y < 0.12 or center_y > 0.68:
        return None

    if aspect < 1.15 or aspect > 2.80:
        return None

    if area_ratio < 0.006 or area_ratio > 0.11:
        return None

    refined, row_confidence = refine_to_number_rows(warped)
    lines = horizontal_line_positions(refined)
    line_count = len(lines)
    verticals = vertical_line_positions(refined)

    line_score = max(0.0, 1.0 - abs(line_count - 9) / 5.0)
    vertical_score = 1.0 if len(verticals) >= 2 else 0.35
    aspect_score = max(0.0, 1.0 - abs(aspect - 1.55) / 1.2)
    right_score = min(1.0, max(0.0, (center_x - 0.55) / 0.35))
    area_score = min(1.0, area_ratio / 0.035)

    score = (
        line_score * 0.38
        + row_confidence * 0.22
        + vertical_score * 0.15
        + aspect_score * 0.12
        + right_score * 0.08
        + area_score * 0.05
    )

    return score, refined


def detect_number_box(image):
    image_h, image_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(30, image_h // 18)),
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, image_w // 18), 1),
    )

    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    table_mask = cv2.add(vertical, horizontal)
    table_mask = cv2.dilate(
        table_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if x + width < image_w * 0.55:
            continue

        if width < image_w * 0.055 or height < image_h * 0.10:
            continue

        scored = score_number_box_candidate(image, contour_points(contour))
        if scored is None:
            continue

        score, refined = scored
        if best is None or score > best[0]:
            best = (score, refined)

    if best is None:
        x1 = int(image_w * 0.68)
        x2 = int(image_w * 0.93)
        y1 = int(image_h * 0.18)
        y2 = int(image_h * 0.50)
        fallback = image[y1:y2, x1:x2]

        if fallback.size == 0:
            return None, 0.0

        refined, row_confidence = refine_to_number_rows(fallback)
        line_count = len(horizontal_line_positions(refined))
        line_score = max(0.0, 1.0 - abs(line_count - 9) / 5.0)
        aspect = refined.shape[0] / float(max(1, refined.shape[1]))
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.55) / 1.3)
        confidence = 0.15 + row_confidence * 0.45 + line_score * 0.25 + aspect_score * 0.15

        print("NUMBER BOX LOCALIZATION FALLBACK USED")
        return refined, min(1.0, confidence)

    return best[1], best[0]


def split_number_box_into_rows(number_box):
    lines = horizontal_line_positions(number_box)
    window, confidence = best_eight_row_window(lines)

    h, w = number_box.shape[:2]
    rows = []

    if window is not None:
        boundaries = [item[0] for item in window]

        for index in range(EXPECTED_ROW_COUNT):
            y1 = boundaries[index]
            y2 = boundaries[index + 1]
            height = y2 - y1
            pad_y = max(2, int(height * 0.12))
            x_pad = max(2, int(w * 0.025))
            y_start = max(0, y1 - pad_y)
            y_end = min(h, y2 + pad_y)

            rows.append(
                {
                    "image": number_box[
                        y_start:y_end,
                        max(0, x_pad) : min(w, w - x_pad),
                    ],
                    "y_start": y_start,
                    "y_end": y_end,
                }
            )

        return rows, confidence

    row_height = h / float(EXPECTED_ROW_COUNT)
    for index in range(EXPECTED_ROW_COUNT):
        y1 = int(round(index * row_height))
        y2 = int(round((index + 1) * row_height))
        pad_y = max(2, int((y2 - y1) * 0.10))
        x_pad = max(2, int(w * 0.025))
        y_start = max(0, y1 - pad_y)
        y_end = min(h, y2 + pad_y)
        rows.append(
            {
                "image": number_box[
                    y_start:y_end,
                    max(0, x_pad) : min(w, w - x_pad),
                ],
                "y_start": y_start,
                "y_end": y_end,
            }
        )

    return rows, 0.35


def neutralize_row_borders(row):
    if row is None or row.size == 0:
        return row

    cleaned = row.copy()

    if len(cleaned.shape) == 2:
        gray = cleaned
    else:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    dark_threshold = 115
    horizontal_band = max(3, int(h * 0.22))
    vertical_band = max(2, int(w * 0.08))

    for start_y, end_y in ((0, horizontal_band), (h - horizontal_band, h)):
        band = gray[start_y:end_y, :]
        dark_mask = band < dark_threshold
        coverage = dark_mask.sum(axis=1) / float(max(1, w))

        for offset, ratio in enumerate(coverage):
            if ratio < 0.48:
                continue

            y = start_y + offset
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            cleaned[y1:y2, :] = cv2.addWeighted(
                cleaned[y1:y2, :],
                0.35,
                np.full_like(cleaned[y1:y2, :], 255),
                0.65,
                0,
            )

    for start_x, end_x in ((0, vertical_band), (w - vertical_band, w)):
        band = gray[:, start_x:end_x]
        dark_mask = band < dark_threshold
        coverage = dark_mask.sum(axis=0) / float(max(1, h))

        for offset, ratio in enumerate(coverage):
            if ratio < 0.45:
                continue

            x = start_x + offset
            x1 = max(0, x - 1)
            x2 = min(w, x + 2)
            cleaned[:, x1:x2] = cv2.addWeighted(
                cleaned[:, x1:x2],
                0.35,
                np.full_like(cleaned[:, x1:x2], 255),
                0.65,
                0,
            )

    return cleaned


def remove_horizontal_borders_for_ocr(row):
    if row is None or row.size == 0:
        return row

    cleaned = row.copy()

    if len(cleaned.shape) == 2:
        gray = cleaned
    else:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    dark = gray < 130
    search_band = max(3, int(h * 0.28))

    for start_y, end_y in ((0, search_band), (h - search_band, h)):
        band = dark[start_y:end_y, :]
        if band.size == 0:
            continue

        coverage = band.sum(axis=1) / float(max(1, w))
        for offset, ratio in enumerate(coverage):
            if ratio < 0.35:
                continue

            y = start_y + offset
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            cleaned[y1:y2, :] = cv2.addWeighted(
                cleaned[y1:y2, :],
                0.45,
                np.full_like(cleaned[y1:y2, :], 255),
                0.55,
                0,
            )

    return cleaned


def pad_row_for_ocr(row):
    if row is None or row.size == 0:
        return row

    h, w = row.shape[:2]
    pad_top = max(2, int(h * 0.12))
    pad_bottom = max(3, int(h * 0.20))
    pad_x = max(3, int(w * 0.05))

    return cv2.copyMakeBorder(
        row,
        pad_top,
        pad_bottom,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def preprocess_raw_enhanced(row):
    if row is None or row.size == 0:
        return row

    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.convertScaleAbs(gray, alpha=1.2, beta=5)


def preprocess_alt_ocr(row):
    if row is None or row.size == 0:
        return row

    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (0, 0), 0.8)
    sharpened = cv2.addWeighted(gray, 1.25, blurred, -0.25, 0)
    return cv2.convertScaleAbs(sharpened, alpha=1.08, beta=3)


def preprocess_light_fallback_ocr(row):
    if row is None or row.size == 0:
        return row

    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.convertScaleAbs(gray, alpha=1.08, beta=2)


def clean_row_for_feature_extraction(row):
    if row is None or row.size == 0:
        return row

    cleaned = neutralize_row_borders(row)

    if len(cleaned.shape) == 2:
        gray = cleaned
    else:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    line_mask = cv2.inRange(gray, 0, 135)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, w // 3), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 2)))
    table_lines = cv2.add(
        cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, horizontal_kernel),
        cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, vertical_kernel),
    )

    contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        is_horizontal = width > w * 0.42 and height <= max(5, h * 0.12)
        is_vertical = height > h * 0.42 and width <= max(5, w * 0.08)

        if not (is_horizontal or is_vertical):
            continue

        cleaned[
            max(0, y - 2) : min(h, y + height + 2),
            max(0, x - 2) : min(w, x + width + 2),
        ] = 255

    return cleaned


def estimate_row_image_features(row):
    if row is None or row.size == 0:
        return {
            "digit_contours": 0,
            "ink_bbox_width": 0,
            "ink_density": 0.0,
            "visible_digit_span": 0.0,
            "visual_digit_count": "unknown",
        }

    feature_row = clean_row_for_feature_extraction(row)
    gray = cv2.cvtColor(feature_row, cv2.COLOR_BGR2GRAY)
    ink = cv2.inRange(gray, 0, 165)

    h, w = ink.shape[:2]
    edge_y = max(1, h // 10)
    edge_x = max(1, w // 22)
    ink[:edge_y, :] = 0
    ink[h - edge_y :, :] = 0
    ink[:, :edge_x] = 0
    ink[:, w - edge_x :] = 0

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < max(4, h * w * 0.001):
            continue

        if height < h * 0.16:
            continue

        if width > w * 0.55 and height < h * 0.24:
            continue

        digit_contours.append((x, y, width, height))

    if digit_contours:
        x1 = min(item[0] for item in digit_contours)
        y1 = min(item[1] for item in digit_contours)
        x2 = max(item[0] + item[2] for item in digit_contours)
        y2 = max(item[1] + item[3] for item in digit_contours)
        ink_bbox_width = x2 - x1
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        ink_density = cv2.countNonZero(ink[y1:y2, x1:x2]) / float(bbox_area)
    else:
        ink_bbox_width = 0
        ink_density = 0.0

    visible_digit_span = ink_bbox_width / float(max(1, w))
    contour_count = len(digit_contours)

    if contour_count <= 1 and visible_digit_span < 0.20:
        visual_digit_count = "1"
    elif contour_count <= 2 and visible_digit_span < 0.34:
        visual_digit_count = "2"
    else:
        visual_digit_count = "3+"

    return {
        "digit_contours": contour_count,
        "ink_bbox_width": int(ink_bbox_width),
        "ink_density": round(float(ink_density), 4),
        "visible_digit_span": round(float(visible_digit_span), 4),
        "visual_digit_count": visual_digit_count,
    }


def run_easyocr_on_image(image):
    if image is None or image.size == 0:
        return None, 0.0

    results = easy_reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
        text_threshold=0.25,
        low_text=0.15,
        link_threshold=0.10,
        decoder="greedy",
        mag_ratio=2,
    )

    if not results:
        return None, 0.0

    texts = []
    confidences = []
    for result in results:
        if len(result) >= 3:
            texts.append(result[1])
            confidences.append(float(result[2]))

    return clean_number_text("".join(texts)), min(confidences) if confidences else 0.0


def run_trocr_on_image(image):
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 2:
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image).convert("RGB")

    pixel_values = trocr_processor(
        images=pil_image,
        return_tensors="pt",
    ).pixel_values.to(device)

    generated_ids = trocr_model.generate(
        pixel_values,
        max_new_tokens=8,
        num_beams=4,
        early_stopping=True,
    )

    text = trocr_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return clean_number_text(text)


def row_group_expected_digits(row_index):
    if row_index in (4, 5, 6):
        return 1, 2

    return 2, 4


def row_visual_digit_limits(row_index, features):
    visual = features.get("visual_digit_count")

    if visual == "1":
        return 1, 1

    if visual == "2":
        return 1, 2

    if visual == "3+":
        return 3, 3

    return row_group_expected_digits(row_index)


def value_passes_row_sanity(row_index, value, previous_values):
    if not is_valid_row_number(value):
        return False

    low_digits, high_digits = row_group_expected_digits(row_index)
    digits = digit_len(value)

    if row_index == 1:
        return 50 <= value <= 5000 and low_digits <= digits <= high_digits

    registered = previous_values[0] if len(previous_values) >= 1 else None
    accredited = previous_values[1] if len(previous_values) >= 2 else None
    issued = previous_values[2] if len(previous_values) >= 3 else None

    if is_valid_row_number(registered) and value > registered:
        return False

    if row_index in (4, 5, 6) and (digits > 2 or value > 99):
        return False

    if row_index == 7 and is_valid_row_number(accredited) and value > accredited:
        return False

    if row_index == 8 and is_valid_row_number(issued) and value > issued:
        return False

    return digits <= high_digits


def visual_digit_match_score(value, features):
    if value is None:
        return 0.0

    digits = digit_len(value)
    visual = features.get("visual_digit_count")

    if visual == "1":
        return 1.0 if digits == 1 else 0.0

    if visual == "2":
        if digits == 2:
            return 1.0
        return 0.35 if digits == 1 else 0.10

    if visual == "3+":
        if digits == 3:
            return 1.0
        return 0.0

    return 0.5


def value_passes_visual_sanity(row_index, value, features):
    if value is None:
        return False

    low_digits, high_digits = row_visual_digit_limits(row_index, features)
    digits = digit_len(value)

    if not (low_digits <= digits <= high_digits):
        return False

    return True


def easyocr_output_is_suspicious(row_index, value, features):
    if value is None:
        return True

    visual = features.get("visual_digit_count")
    digits = digit_len(value)

    if value == 0 and visual in ("2", "3+"):
        return True

    if visual == "3+" and digits < 3:
        return True

    if visual == "3+" and digits > 3:
        return True

    if visual == "2" and digits > 2:
        return True

    if visual == "1" and digits > 1:
        return True

    if row_index in (4, 5, 6) and (digits > 2 or value > 99):
        return True

    return False


def ocr_uncertainty_reason(row_index, value, features):
    if value is None:
        return "OCR returned None"

    visual = features.get("visual_digit_count")
    digits = digit_len(value)

    if value == 0 and visual in ("2", "3+"):
        return f"OCR=0 but visual_digit_count={visual}"

    if visual == "3+" and digits < 3:
        return f"OCR={value} but visual span suggests 3 digits"

    if visual == "3+" and digits > 3:
        return f"OCR={value} but visual span suggests 3 digits, not 4+"

    if visual == "2" and digits < 2:
        return f"OCR={value} but visual span suggests 2 digits"

    if visual == "1" and digits > 1:
        return f"OCR={value} but visual span suggests 1 digit"

    if row_index in (4, 5, 6) and (digits > 2 or value > 99):
        return f"OCR={value} suspicious for small-count row"

    return None


def cleaned_three_digit_candidates(value):
    if value is None:
        return []

    text = str(abs(value))
    if len(text) != 4:
        return []

    candidates = []
    for index in range(len(text)):
        cleaned = text[:index] + text[index + 1 :]
        if len(cleaned) == 3 and cleaned[0] != "0":
            candidates.append(int(cleaned))

    return sorted(set(candidates))


def add_cleaned_four_digit_candidates(row_index, candidates, previous_values, features):
    if features.get("visual_digit_count") != "3+":
        return [], []

    independent_values = set()
    cleaned_debug = []

    for candidate in candidates:
        value = candidate["value"]
        if (
            value is not None
            and digit_len(value) == 3
            and value_passes_row_sanity(row_index, value, previous_values)
            and value_passes_visual_sanity(row_index, value, features)
        ):
            independent_values.add(value)

    additions = []
    for candidate in list(candidates):
        value = candidate["value"]
        if value is None or digit_len(value) != 4:
            continue

        cleanup_candidates = cleaned_three_digit_candidates(value)
        valid_cleaned = []
        for cleaned_value in cleanup_candidates:
            if not value_passes_row_sanity(row_index, cleaned_value, previous_values):
                continue

            if not value_passes_visual_sanity(row_index, cleaned_value, features):
                continue

            valid_cleaned.append(cleaned_value)

        agreed_cleaned = [
            cleaned_value
            for cleaned_value in valid_cleaned
            if cleaned_value in independent_values
        ]

        accepted = None
        reason = None
        if len(valid_cleaned) == 1:
            accepted = valid_cleaned[0]
            reason = "exactly one valid cleanup candidate"
        elif len(agreed_cleaned) == 1:
            accepted = agreed_cleaned[0]
            reason = "another OCR source agreed with cleanup candidate"
        elif len(valid_cleaned) > 1:
            reason = "multiple valid cleanup candidates without OCR agreement"
        else:
            reason = "no valid cleanup candidates"

        cleaned_debug.append(
            {
                "source": candidate["source"],
                "original": value,
                "all_candidates": cleanup_candidates,
                "valid_candidates": valid_cleaned,
                "accepted": accepted,
                "reason": reason,
            }
        )

        if accepted is not None:
            additions.append(
                {
                    "source": f"{candidate['source']}_cleaned",
                    "value": accepted,
                    "confidence": max(0.0, candidate["confidence"] - 0.12),
                    "engine_weight": max(0.35, candidate["engine_weight"] - 0.22),
                    "cleaned_from": value,
                }
            )

    return cleaned_debug, additions


def easyocr_needs_trocr_rescue(row_index, raw_reading, previous_values, features):
    raw_value, raw_confidence = raw_reading

    if raw_value is None:
        return True

    if raw_confidence < 0.60:
        return True

    if not value_passes_row_sanity(row_index, raw_value, previous_values):
        return True

    if not value_passes_visual_sanity(row_index, raw_value, features):
        return True

    return easyocr_output_is_suspicious(row_index, raw_value, features)


def choose_best_ocr_candidate(row_index, raw_reading, alt_reading, trocr_value, previous_values, features, localization_confidence):
    raw_value, raw_confidence = raw_reading
    alt_value, alt_confidence = alt_reading
    easyocr_bad = easyocr_needs_trocr_rescue(row_index, raw_reading, previous_values, features)
    raw_uncertainty = ocr_uncertainty_reason(row_index, raw_value, features)
    if raw_uncertainty is not None:
        print(f"ROW {row_index} EasyOCR uncertainty reason:", raw_uncertainty)

    raw_is_sane = (
        value_passes_row_sanity(row_index, raw_value, previous_values)
        and value_passes_visual_sanity(row_index, raw_value, features)
        and not easyocr_output_is_suspicious(row_index, raw_value, features)
    )

    if trocr_value is not None and not value_passes_visual_sanity(row_index, trocr_value, features):
        print(
            f"ROW {row_index} TrOCR rejected:",
            f"value {trocr_value} does not match visual digit count",
        )

    if trocr_value is not None and not value_passes_row_sanity(row_index, trocr_value, previous_values):
        print(
            f"ROW {row_index} TrOCR rejected:",
            f"value {trocr_value} is not plausible for row",
        )

    if raw_confidence >= RAW_STRONG_CONFIDENCE and raw_is_sane:
        print(f"ROW {row_index} cleaned 4-digit candidates:", [])
        score = (
            raw_confidence * 0.50
            + visual_digit_match_score(raw_value, features) * 0.28
            + localization_confidence * 0.12
            + min(features.get("digit_contours", 0) / 3.0, 1.0) * 0.10
        )
        print(
            f"ROW {row_index} OCR candidate scores:",
            [
                {
                    "source": "raw",
                    "value": raw_value,
                    "score": round(score, 3),
                }
            ],
        )
        print(f"ROW {row_index} chosen reason: EasyOCR raw high-confidence plausible")
        return raw_value, score

    candidates = [
        {
            "source": "raw",
            "value": raw_value,
            "confidence": raw_confidence,
            "engine_weight": 1.0,
        }
    ]

    if alt_value is not None:
        candidates.append(
            {
                "source": "alt",
                "value": alt_value,
                "confidence": alt_confidence,
                "engine_weight": 0.92,
            }
        )

    if easyocr_bad:
        candidates.append(
            {
                "source": "trocr",
                "value": trocr_value,
                "confidence": 0.55 if row_index in (1, 2, 3, 7, 8) or features.get("visual_digit_count") == "3+" else 0.46,
                "engine_weight": 0.62,
            }
        )

    cleaned_debug, cleaned_additions = add_cleaned_four_digit_candidates(
        row_index,
        candidates,
        previous_values,
        features,
    )
    print(f"ROW {row_index} cleaned 4-digit candidates:", cleaned_debug)
    if any(
        item["original"] is not None
        and len(item["valid_candidates"]) > 1
        and item["accepted"] is None
        for item in cleaned_debug
    ):
        print(f"ROW {row_index} OCR uncertain: ambiguous 4-digit cleanup")
        return None, 0.0
    candidates.extend(cleaned_additions)

    values = [candidate["value"] for candidate in candidates if candidate["value"] is not None]
    candidate_count_by_value = {
        value: sum(1 for item in values if item == value)
        for value in set(values)
    }

    scored = []
    for candidate in candidates:
        value = candidate["value"]
        if not value_passes_row_sanity(row_index, value, previous_values):
            continue

        if not value_passes_visual_sanity(row_index, value, features):
            continue

        visual_score = visual_digit_match_score(value, features)
        if visual_score <= 0:
            continue

        agreement = sum(1 for item in values if item == value) - 1
        confidence = candidate["confidence"]

        score = (
            confidence * 0.36
            + visual_score * 0.28
            + candidate["engine_weight"] * 0.13
            + min(agreement, 2) * 0.16
            + localization_confidence * 0.08
            + min(features.get("digit_contours", 0) / 3.0, 1.0) * 0.05
        )

        if candidate_count_by_value.get(value, 0) >= 2:
            score += 0.08

        if confidence < 0.35:
            score -= 0.10
        elif confidence < 0.55:
            score -= 0.04

        if candidate["source"] == "trocr":
            if easyocr_bad:
                score += 0.12
            elif agreement == 0:
                score -= 0.18

        if "cleaned_from" in candidate:
            score -= 0.08
            if candidate_count_by_value.get(value, 0) >= 2:
                score += 0.10

        if candidate["source"] == "raw" and easyocr_output_is_suspicious(row_index, value, features):
            score -= 0.16

        if digit_len(value) >= 4:
            score -= 0.30

        if features.get("visual_digit_count") == "3+" and digit_len(value) != 3:
            score -= 0.25

        if row_index in (4, 5, 6) and digit_len(value) > 2:
            score -= 0.35

        if (
            digit_len(value) == 3
            and confidence < 0.55
            and candidate_count_by_value.get(value, 0) == 1
            and len(set(values)) > 1
        ):
            score -= 0.08

        if candidate["source"] == "alt":
            if visual_score > visual_digit_match_score(raw_value, features):
                score += 0.06
            else:
                score -= 0.03

            print(f"ROW {row_index} alt candidate score:", round(score, 3))

        scored.append((score, candidate))

    if not scored:
        print(f"ROW {row_index} OCR uncertain: no sane candidates")
        return None, 0.0

    agreement_scores = {}
    for score, candidate in scored:
        value = candidate["value"]
        agreement_scores.setdefault(value, []).append((score, candidate))

    agreed_values = [
        (value, items)
        for value, items in agreement_scores.items()
        if len({item["source"].replace("_cleaned", "") for _, item in items}) >= 2
    ]

    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    print(
        f"ROW {row_index} OCR candidate scores:",
        [
            {
                "source": candidate["source"],
                "value": candidate["value"],
                "score": round(score, 3),
            }
            for score, candidate in scored
        ],
    )

    if agreed_values:
        agreed_values = sorted(
            agreed_values,
            key=lambda item: max(score for score, _ in item[1]),
            reverse=True,
        )
        agreed_value, agreed_items = agreed_values[0]
        agreed_score = max(score for score, _ in agreed_items)
        print(f"ROW {row_index} chosen reason: OCR source agreement")
        return agreed_value, agreed_score

    if best["source"] == "trocr" and easyocr_bad and best_score >= MIN_TROCR_RESCUE_CONFIDENCE:
        print(f"ROW {row_index} chosen reason: TrOCR rescue after weak/suspicious EasyOCR")
        return best["value"], best_score

    if best_score < MIN_ROW_CONFIDENCE:
        print(f"ROW {row_index} OCR uncertain: low winning score")
        return None, best_score

    if len(scored) > 1 and best_score - second_score < 0.05:
        print(f"ROW {row_index} OCR uncertain: close competing candidates")
        return None, best_score

    print(f"ROW {row_index} chosen reason: {best['source']} highest weighted plausible candidate")
    return best["value"], best_score


def extract_numbers_from_boxes(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("OCR FAILED - IMAGE COULD NOT BE READ")
        return {
            "values": [None] * EXPECTED_ROW_COUNT,
            "none_count": EXPECTED_ROW_COUNT,
            "ocr_status": "failed",
            "ocr_failed": True,
            "logic_available": False,
            "localization_confidence": 0.0,
            "row_split_confidence": 0.0,
        }

    number_box, localization_confidence = detect_number_box(image)
    print("NUMBER BOX LOCALIZATION CONFIDENCE:", round(localization_confidence, 3))

    if number_box is None or localization_confidence < MIN_LOCALIZATION_CONFIDENCE:
        print("OCR FAILED - LOW LOCALIZATION CONFIDENCE")
        return {
            "values": [None] * EXPECTED_ROW_COUNT,
            "none_count": EXPECTED_ROW_COUNT,
            "ocr_status": "failed",
            "ocr_failed": True,
            "logic_available": False,
            "localization_confidence": localization_confidence,
            "row_split_confidence": 0.0,
        }

    save_debug_image("debug_number_crop.jpg", number_box)

    rows, row_split_confidence = split_number_box_into_rows(number_box)
    print("ROW SPLIT CONFIDENCE:", round(row_split_confidence, 3))

    if len(rows) != EXPECTED_ROW_COUNT or row_split_confidence < 0.35:
        print("OCR FAILED - ROW SEGMENTATION UNCERTAIN")
        return {
            "values": [None] * EXPECTED_ROW_COUNT,
            "none_count": EXPECTED_ROW_COUNT,
            "ocr_status": "failed",
            "ocr_failed": True,
            "logic_available": False,
            "localization_confidence": localization_confidence,
            "row_split_confidence": row_split_confidence,
        }

    row_numbers = []
    row_contexts = []

    for row_index, row_info in enumerate(rows, start=1):
        row = row_info["image"]
        y_start = row_info["y_start"]
        y_end = row_info["y_end"]

        print(
            f"ROW {row_index} segment:",
            {
                "y_start": y_start,
                "y_end": y_end,
                "shape": row.shape,
            },
        )

        raw_enhanced = preprocess_raw_enhanced(row)

        save_debug_image(f"debug_raw_enhanced_row_{row_index}.jpg", raw_enhanced)

        features = estimate_row_image_features(neutralize_row_borders(row))
        raw_reading = run_easyocr_on_image(raw_enhanced)
        raw_value, raw_confidence = raw_reading
        raw_digits = digit_len(raw_value)
        visual_digit_count = features.get("visual_digit_count")
        alt_reason = None

        if raw_value is None:
            alt_reason = "EasyOCR returned None"
        elif raw_confidence < 0.60:
            alt_reason = "EasyOCR confidence is weak"
        elif easyocr_output_is_suspicious(row_index, raw_value, features):
            alt_reason = "EasyOCR output is suspicious"
        elif visual_digit_count == "3+" and raw_digits < 3:
            alt_reason = "visual_digit_count=3+ but EasyOCR gave fewer than 3 digits"
        elif visual_digit_count == "1" and raw_digits > 1:
            alt_reason = "visual_digit_count=1 but EasyOCR gave more than 1 digit"
        elif visual_digit_count == "2" and raw_digits > 2:
            alt_reason = "visual_digit_count=2 but EasyOCR gave more than 2 digits"

        if alt_reason is not None:
            alt_ocr = preprocess_alt_ocr(row)
            alt_reading = run_easyocr_on_image(alt_ocr)
        else:
            alt_reading = (None, 0.0)

        if easyocr_needs_trocr_rescue(row_index, raw_reading, row_numbers, features):
            trocr_value = run_trocr_on_image(raw_enhanced)
        else:
            trocr_value = None
        trocr_confidence = (
            0.55
            if trocr_value is not None and (row_index in (1, 2, 3, 7, 8) or visual_digit_count == "3+")
            else 0.46 if trocr_value is not None else 0.0
        )

        print(f"ROW {row_index} image features:", features)
        print(f"ROW {row_index} visual_digit_count:", visual_digit_count)
        print(f"ROW {row_index} EasyOCR raw enhanced:", raw_reading[0], round(raw_reading[1], 3))
        if alt_reason is not None:
            print(f"ROW {row_index} alt OCR used because:", alt_reason)
        print(f"ROW {row_index} EasyOCR alt:", alt_reading[0], round(alt_reading[1], 3))
        print(f"ROW {row_index} TrOCR verifier:", trocr_value, round(trocr_confidence, 3))

        value, row_confidence = choose_best_ocr_candidate(
            row_index,
            raw_reading,
            alt_reading,
            trocr_value,
            row_numbers,
            features,
            localization_confidence,
        )

        print(f"ROW {row_index} FINAL OCR:", value, "CONF:", round(row_confidence, 3))

        row_numbers.append(value)
        row_contexts.append(
            {
                "row": row,
                "features": features,
                "visual_digit_count": visual_digit_count,
            }
        )

    print("ROW OCR VALUES:", row_numbers)

    none_count_before_fallback = sum(value is None for value in row_numbers)
    print("NONE COUNT BEFORE FALLBACK:", none_count_before_fallback)
    fallback_used = False

    if none_count_before_fallback >= 3 or any(
        value is None and context["visual_digit_count"] == "3+"
        for value, context in zip(row_numbers, row_contexts)
    ):
        print("LIGHT FALLBACK OCR USED:", True)
        print("ROW OCR VALUES BEFORE FALLBACK:", row_numbers)

        for row_index, context in enumerate(row_contexts, start=1):
            if row_numbers[row_index - 1] is not None:
                continue

            visual_digit_count = context["visual_digit_count"]
            if none_count_before_fallback < 3 and visual_digit_count != "3+":
                continue

            fallback_used = True
            fallback_image = preprocess_light_fallback_ocr(context["row"])
            fallback_reading = run_easyocr_on_image(fallback_image)
            fallback_previous_values = row_numbers[: row_index - 1]

            if easyocr_needs_trocr_rescue(
                row_index,
                fallback_reading,
                fallback_previous_values,
                context["features"],
            ):
                fallback_trocr_value = run_trocr_on_image(fallback_image)
            else:
                fallback_trocr_value = None

            fallback_trocr_confidence = (
                0.55
                if fallback_trocr_value is not None
                and (row_index in (1, 2, 3, 7, 8) or visual_digit_count == "3+")
                else 0.46 if fallback_trocr_value is not None else 0.0
            )

            print(f"ROW {row_index} fallback OCR used:", True)
            print(
                f"ROW {row_index} fallback EasyOCR:",
                fallback_reading[0],
                round(fallback_reading[1], 3),
            )
            print(
                f"ROW {row_index} fallback TrOCR verifier:",
                fallback_trocr_value,
                round(fallback_trocr_confidence, 3),
            )

            fallback_value, fallback_confidence = choose_best_ocr_candidate(
                row_index,
                (None, 0.0),
                fallback_reading,
                fallback_trocr_value,
                fallback_previous_values,
                context["features"],
                localization_confidence,
            )

            print(
                f"ROW {row_index} fallback FINAL OCR:",
                fallback_value,
                "CONF:",
                round(fallback_confidence, 3),
            )

            if fallback_value is not None:
                row_numbers[row_index - 1] = fallback_value
        print("ROW OCR VALUES AFTER FALLBACK:", row_numbers)
    else:
        print("LIGHT FALLBACK OCR USED:", False)

    none_count = sum(value is None for value in row_numbers)
    print("NONE COUNT AFTER FALLBACK:", none_count)
    if none_count == 0:
        ocr_status = "complete"
        ocr_failed = False
        logic_available = True
    elif none_count in (1, 2):
        ocr_status = "partial"
        ocr_failed = False
        logic_available = False
    else:
        ocr_status = "failed"
        ocr_failed = True
        logic_available = False

    print("NONE COUNT:", none_count)
    print("FALLBACK OCR USED:", fallback_used)
    print("OCR STATUS:", ocr_status)
    print("OCR FAILED:", ocr_failed)
    print("LOGIC AVAILABLE:", logic_available)

    return {
        "values": row_numbers,
        "none_count": none_count,
        "ocr_status": ocr_status,
        "ocr_failed": ocr_failed,
        "logic_available": logic_available,
        "localization_confidence": localization_confidence,
        "row_split_confidence": row_split_confidence,
    }
