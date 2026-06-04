from flask import Flask, render_template, request, jsonify
from pathlib import Path
import uuid
import traceback

from yolo_corner_warp_pad0_stable import process_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_document", methods=["POST"])
def process_document():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "이미지가 없습니다."}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"ok": False, "error": "파일명이 없습니다."}), 400

        ext = Path(file.filename).suffix.lower()

        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            return jsonify({"ok": False, "error": f"지원하지 않는 파일 형식입니다: {ext}"}), 400

        size_mode = request.form.get("size_mode", "mid")
        if size_mode not in ["high", "mid", "low"]:
            size_mode = "mid"

        filter_mode = request.form.get("filter_mode", "auto")
        if filter_mode not in ["auto", "bright", "gray", "sharp", "shadow"]:
            filter_mode = "auto"

        input_name = f"{uuid.uuid4().hex}{ext}"
        output_name = f"{uuid.uuid4().hex}.jpg"

        input_path = UPLOAD_DIR / input_name
        output_path = RESULT_DIR / output_name

        file.save(input_path)

        result = process_image(
            image_path=str(input_path),
            output_path=str(output_path),
            size_mode=size_mode,
            filter_mode=filter_mode
        )

        return jsonify({
            "ok": True,
            "result_url": f"/static/results/{output_name}",
            "detail": result
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
