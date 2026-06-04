# jScanner

jScanner is a local Flask + YOLO document scanner for correcting photographed slide/document images.

## Features

- Web UI for selecting multiple images
- YOLO-based document/slide detection
- Perspective correction and inner-border refinement
- Output filters: auto, bright, grayscale, sharpen, shadow removal
- Output size presets: high, mid, low

## Folder structure

```text
jScanner/
├─ app.py
├─ yolo_corner_warp_pad0_stable.py
├─ requirements.txt
├─ templates/
│  └─ index.html
├─ static/
│  ├─ uploads/
│  └─ results/
└─ runs/
   └─ detect/
      └─ train/
         └─ weights/
            └─ best.pt
```

## Model file

Place your trained YOLO model here:

```text
runs/detect/train/weights/best.pt
```

If your training folder is `train2`, `train3`, etc., jScanner will also try to find the newest:

```text
runs/detect/train*/weights/best.pt
```

## Install

```cmd
pip install -r requirements.txt
```

## Run

```cmd
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Notes

This is designed for local/private use. Do not upload confidential images to third-party services unless your organization allows it.
