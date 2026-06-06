import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

# Render 같은 서버 환경에서는 기본 Ultralytics 설정 폴더에 쓰기 권한이 없을 수 있음.
# YOLO/torch는 무겁기 때문에 앱 시작 때가 아니라 실제 요청 시점에만 불러온다.
YOLO_CONFIG_DIR = Path(os.environ.get("YOLO_CONFIG_DIR", Path(tempfile.gettempdir()) / "Ultralytics"))
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

SIZE_PROFILES = {
    "high": {"max_side": 3000, "quality": 92},
    "mid": {"max_side": 2000, "quality": 82},
    "low": {"max_side": 1200, "quality": 70},
}

_model = None


def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH

    detect_dir = BASE_DIR / "runs" / "detect"
    candidates = []
    if detect_dir.exists():
        candidates = sorted(
            detect_dir.glob("train*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"best.pt를 찾지 못했습니다: {MODEL_PATH}")


def get_model():
    global _model

    if _model is None:
        model_path = resolve_model_path()

        print(f"[jScanner] Loading YOLO model: {model_path}", flush=True)
        print(f"[jScanner] YOLO model exists: {model_path.exists()}", flush=True)
        if model_path.exists():
            print(f"[jScanner] YOLO model size: {model_path.stat().st_size} bytes", flush=True)

        import os
        import gc

        os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
        os.makedirs("/tmp/Ultralytics", exist_ok=True)
        os.makedirs("/tmp/matplotlib", exist_ok=True)

        print("[jScanner] before ultralytics import", flush=True)
        from ultralytics import YOLO
        print("[jScanner] after ultralytics import", flush=True)

        print("[jScanner] before YOLO init", flush=True)
        _model = YOLO(str(model_path))
        print("[jScanner] after YOLO init", flush=True)

        try:
            _model.fuse()
            print("[jScanner] YOLO model fused", flush=True)
        except Exception as e:
            print(f"[jScanner] YOLO fuse skipped: {e}", flush=True)

        gc.collect()

        print("[jScanner] YOLO model loaded", flush=True)

    return _model


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def warp_image(img, pts):
    tl, tr, br, bl = pts

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, matrix, (max_w, max_h))

    return warped


def find_precise_corners(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    pad = 0
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = img[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 40, 80]),
        np.array([45, 255, 255])
    )

    crop_clean = crop.copy()
    crop_clean[yellow_mask > 0] = (0, 0, 0)

    gray = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        crop_area = crop.shape[0] * crop.shape[1]

        if area < crop_area * 0.20:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)

        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")
        ordered = order_points(pts)

        tl, tr, br, bl = ordered

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        avg_w = (width_top + width_bottom) / 2
        avg_h = (height_left + height_right) / 2

        if avg_h <= 0:
            continue

        ratio = avg_w / avg_h

        if ratio < 1.15 or ratio > 2.2:
            continue

        bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))

        penalty = 0

        if by < crop.shape[0] * 0.03:
            penalty += area * 0.25

        if bw > crop.shape[1] * 0.97:
            penalty += area * 0.20

        if bh > crop.shape[0] * 0.96:
            penalty += area * 0.20

        score = area - penalty
        candidates.append((score, ordered))

    if not candidates:
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype="float32")
        return order_points(pts), "fallback_yolo_box"

    candidates.sort(key=lambda x: x[0], reverse=True)
    pts = candidates[0][1]

    pts[:, 0] += x1
    pts[:, 1] += y1

    return order_points(pts), "precise_contour_scored"


def refine_warp_by_inner_border(warped):
    h, w = warped.shape[:2]

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)

    h_kernel_len = max(60, int(w * 0.12))
    v_kernel_len = max(60, int(h * 0.12))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    h_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    v_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

    row_score = h_edges.sum(axis=1) / 255
    col_score = v_edges.sum(axis=0) / 255

    row_score = cv2.GaussianBlur(
        row_score.reshape(-1, 1).astype(np.float32),
        (1, 21),
        0
    ).reshape(-1)

    col_score = cv2.GaussianBlur(
        col_score.reshape(1, -1).astype(np.float32),
        (21, 1),
        0
    ).reshape(-1)

    top_range = range(0, int(h * 0.25))
    bottom_range = range(int(h * 0.70), h)
    left_range = range(0, int(w * 0.25))
    right_range = range(int(w * 0.75), w)

    top = max(top_range, key=lambda y: row_score[y])
    bottom = max(bottom_range, key=lambda y: row_score[y])
    left = max(left_range, key=lambda x: col_score[x])
    right = max(right_range, key=lambda x: col_score[x])

    min_row_score = w * 0.08
    min_col_score = h * 0.08

    if row_score[top] < min_row_score:
        top = 0

    if row_score[bottom] < min_row_score:
        bottom = h - 1

    if col_score[left] < min_col_score:
        left = 0

    if col_score[right] < min_col_score:
        right = w - 1

    margin = 4

    top = max(0, top - margin)
    bottom = min(h - 1, bottom + margin)
    left = max(0, left - margin)
    right = min(w - 1, right + margin)

    if right <= left or bottom <= top:
        return warped, "inner_border_fallback"

    final = warped[top:bottom + 1, left:right + 1]

    return final, "inner_border_refined"



def clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_auto_filter(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.04, beta=4)


def apply_bright_filter(img):
    return cv2.convertScaleAbs(img, alpha=1.10, beta=18)


def apply_gray_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.18, beta=8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_sharp_filter(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.55, blurred, -0.55, 0)


def apply_shadow_filter(img):
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(norm)

    result = cv2.merge(result_planes)
    return cv2.convertScaleAbs(result, alpha=1.08, beta=4)


def apply_filter(img, filter_mode):
    if filter_mode == "original":
        return img
    if filter_mode == "bright":
        return apply_bright_filter(img)
    if filter_mode == "gray":
        return apply_gray_filter(img)
    if filter_mode == "sharp":
        return apply_sharp_filter(img)
    if filter_mode == "shadow":
        return apply_shadow_filter(img)
    return apply_auto_filter(img)

def apply_output_profile(img, size_mode):
    profile = SIZE_PROFILES.get(size_mode, SIZE_PROFILES["mid"])
    max_side = profile["max_side"]

    h, w = img.shape[:2]
    long_side = max(h, w)

    if long_side > max_side:
        scale = max_side / long_side
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img, profile["quality"]


def save_jpeg(img, output_path, size_mode):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img, quality = apply_output_profile(img, size_mode)
    ok = cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

    if not ok:
        raise RuntimeError(f"결과 저장 실패: {output_path}")


def process_image(image_path, output_path, size_mode="mid", filter_mode="auto"):
    print("[jScanner] process_image start", flush=True)
    print("[jScanner] before get_model", flush=True)

    model = get_model()
    print("[jScanner] after get_model", flush=True)

    print("[jScanner] before cv2.imread", flush=True)
    img = cv2.imread(str(image_path))
    print("[jScanner] after cv2.imread", flush=True)

    if img is None:
        raise FileNotFoundError(f"이미지를 못 읽음: {image_path}")

    print("[jScanner] BEFORE predict", flush=True)

    results = model.predict(
        str(image_path),
        conf=0.25,
        imgsz=320,
        save=False,
        verbose=False,
        device="cpu"
    )

    print("[jScanner] AFTER predict", flush=True)

    boxes = results[0].boxes

    box_count = 0 if boxes is None else len(boxes)
    print(f"[jScanner] YOLO boxes detected: {box_count}", flush=True)

    if boxes is None or len(boxes) == 0:
        raise RuntimeError("YOLO가 슬라이드를 못 찾음")

    best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
    box = boxes.xyxy[best_idx].cpu().numpy()

    pts, mode1 = find_precise_corners(img, box)
    warped_raw = warp_image(img, pts)
    warped_final, mode2 = refine_warp_by_inner_border(warped_raw)
    filtered = apply_filter(warped_final, filter_mode)

    save_jpeg(filtered, output_path, size_mode)

    print(f"[jScanner] saved result: {output_path}", flush=True)

    return {
        "ok": True,
        "corner_mode": mode1,
        "inner_mode": mode2,
        "size_mode": size_mode,
        "filter_mode": filter_mode,
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    result = process_image(
        r"C:\Users\chuki\Desktop\yolo\test.jpg",
        r"C:\Users\chuki\Desktop\yolo\corner_test\output_warped_final.jpg",
        size_mode="mid",
        filter_mode="auto"
    )

    print("완료")
    print(result)
