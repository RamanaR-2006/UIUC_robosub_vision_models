from pathlib import Path

from ultralytics import YOLO

# llm slop that basically gets runs/detect/train-2/weights and test images location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WEIGHTS_PATH = REPO_ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt"
INPUT_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR = SCRIPT_DIR / "test_images_out"

IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def main() -> None:
    """Run the best trained model on test images and save annotated results."""
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {INPUT_DIR}")

    # get all image paths
    image_paths = sorted(
        path for path in INPUT_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {INPUT_DIR}")

    # load yolo model
    model = YOLO(WEIGHTS_PATH)

    # run best yolo model on test
    results = model.predict(
        source=[str(path) for path in image_paths],
        save=True,
        save_txt=True,
        save_conf=True,
        project=SCRIPT_DIR,
        name=OUTPUT_DIR.name,
        exist_ok=True,
    )

    
    detection_count = sum(len(result.boxes) for result in results)
    print(f"Processed {len(results)} image(s) with {detection_count} detection(s).")
    print(f"Annotated images and prediction labels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
