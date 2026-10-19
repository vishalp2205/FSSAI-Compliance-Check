"""
YOLO11n Training Script — FSSAI Symbol Detection

Per project report §5.2 and §9:
  - Fine-tunes YOLO11n (nano) on 3 classes: veg_mark, non_veg_mark, fssai_logo
  - Uses pretrained COCO weights as base
  - Trains on annotated label images in backend/data/yolo_dataset/

Usage:
  python train_yolo.py                  # Train with default settings
  python train_yolo.py --epochs 100     # Train for 100 epochs
  python train_yolo.py --resume         # Resume training

Requirements:
  - Annotated images in backend/data/yolo_dataset/images/{train,val}/
  - YOLO-format labels in backend/data/yolo_dataset/labels/{train,val}/
  - Each label file: <class_id> <x_center> <y_center> <width> <height> (normalized)
  - Classes: 0=veg_mark, 1=non_veg_mark, 2=fssai_logo

Output:
  - Trained model saved to backend/models/yolo_fssai.pt
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n for FSSAI symbol detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--device", default="cpu", help="Device: cpu or 0 (GPU)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Paths
    project_root = Path(__file__).parent
    dataset_yaml = project_root / "backend" / "data" / "yolo_dataset" / "dataset.yaml"
    output_model = project_root / "backend" / "models" / "yolo_fssai.pt"
    train_images = project_root / "backend" / "data" / "yolo_dataset" / "images" / "train"

    # Check dataset exists
    if not dataset_yaml.exists():
        print(f"ERROR: Dataset config not found at {dataset_yaml}")
        sys.exit(1)

    train_count = len(list(train_images.glob("*.jpg")) + list(train_images.glob("*.png")))
    if train_count == 0:
        print("=" * 60)
        print("  YOLO11n Training — Dataset Setup Required")
        print("=" * 60)
        print()
        print("No training images found. To train the model:")
        print()
        print("1. Collect 50-100 food label images")
        print("2. Annotate with LabelImg or Roboflow:")
        print("   - Class 0: veg_mark (green circle in green square)")
        print("   - Class 1: non_veg_mark (brown triangle in brown square)")
        print("   - Class 2: fssai_logo (FSSAI shield logo)")
        print()
        print("3. Place images in:")
        print(f"   Train: {train_images}")
        print(f"   Val:   {train_images.parent / 'val'}")
        print()
        print("4. Place YOLO-format labels (.txt) in:")
        print(f"   Train: {project_root / 'backend' / 'data' / 'yolo_dataset' / 'labels' / 'train'}")
        print(f"   Val:   {project_root / 'backend' / 'data' / 'yolo_dataset' / 'labels' / 'val'}")
        print()
        print("5. Re-run this script: python train_yolo.py")
        print()
        print("NOTE: The system uses OpenCV color detection as fallback")
        print("      until a trained YOLO model is available.")
        return

    print("=" * 60)
    print("  YOLO11n Training — FSSAI Symbol Detection")
    print("=" * 60)
    print(f"\n  Dataset: {dataset_yaml}")
    print(f"  Training images: {train_count}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_model}")
    print()

    # Load YOLO11n pretrained on COCO
    print("[1/3] Loading YOLO11n pretrained weights...")
    model = YOLO("yolo11n.pt")  # Downloads if not cached

    # Train on FSSAI dataset
    print("[2/3] Starting training...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project_root / "backend" / "models" / "runs"),
        name="fssai_symbols",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
    )

    # Save best model to expected location
    print("[3/3] Saving trained model...")
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        import shutil
        os.makedirs(output_model.parent, exist_ok=True)
        shutil.copy2(best_model_path, output_model)
        print(f"\n✅ Model saved to: {output_model}")
        print(f"   The server will automatically use it on next restart.")
    else:
        print(f"\n⚠️ Best model not found at {best_model_path}")
        print(f"   Check training logs for issues.")

    # Evaluate
    print("\n--- Evaluation Results ---")
    metrics = model.val()
    print(f"  mAP@50: {metrics.box.map50:.3f}")
    print(f"  mAP@50-95: {metrics.box.map:.3f}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
