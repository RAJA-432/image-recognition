#!/usr/bin/env python3
"""
Run inference using trained YOLOv8 model (Flexible loading)
Supports: best.pt, pretrained yolov8n.pt, ONNX, TorchScript
"""
import os
import cv2
import argparse
from ultralytics import YOLO


def run_inference(image_path, model_path=None, backend='pytorch', output_dir='output'):
    """
    Run inference on an image.

    Args:
        image_path: Path to input image
        model_path: Path to model weights (best.pt or pretrained model)
        backend: 'pytorch' or 'onnx' for inference
        output_dir: Directory to save output
    """
    # Load model - use best.pt if exists, otherwise pretrained model
    if model_path is None:
        # Try to load best.pt from models folder
        if os.path.exists('models/best.pt'):
            model_path = 'models/best.pt'
            print(f"Loading trained model: {model_path}")
        elif os.path.exists('models/model.onnx'):
            backend = 'onnx'
            model_path = 'models/model.onnx'
            print(f"Loading ONNX model: {model_path}")
        else:
            model_path = 'yolov8n.pt'
            print(f"Using pretrained model: {model_path}")
    else:
        print(f"Loading model: {model_path}")

    if backend == 'onnx':
        from ultralytics import YOLO
        model = YOLO(model_path)
    else:
        model = YOLO(model_path)

    # Run inference
    results = model(image_path, conf=0.25, iou=0.45, show=False)

    # Process results
    for result in results:
        # Get boxes and classes
        boxes = result.boxes
        if boxes is not None:
            print(f"\nDetected {len(boxes)} objects:")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"  - {name}: {conf:.2f}")

    # Plot results
    result_img = results[0].plot()

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.jpg')
    cv2.imwrite(output_path, result_img)
    print(f"\nOutput saved to: {output_path}")

    return results


def run_on_video(video_path, model_path=None, backend='pytorch', output_dir='output'):
    """Run YOLO on video file"""
    if model_path is None:
        if os.path.exists('models/best.pt'):
            model_path = 'models/best.pt'
            print(f"Loading trained model: {model_path}")
        else:
            model_path = 'yolov8n.pt'
            print(f"Using pretrained model: {model_path}")
    else:
        print(f"Loading model: {model_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(cap.get(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output_video.mp4')

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Output: {output_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25, show=False)
        result_frame = results[0].plot()
        out.write(result_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing completed: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Object Detection - Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python src/predict.py --image data/images/test.jpg
  python src/predict.py --image data/images/test.jpg --model models/best.pt
  python src/predict.py --video data/videos/test.mp4
  python src/predict.py --image data/images/test.jpg --backend onnx
        '''
    )
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights (default: best.pt or pretrained)')
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'onnx'],
                       help='Inference backend (default: pytorch)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold (default: 0.45)')

    args = parser.parse_args()

    if args.image:
        run_inference(args.image, args.model, args.backend, args.output_dir)
    elif args.video:
        run_on_video(args.video, args.model, args.backend, args.output_dir)
    else:
        print("Error: Provide --image or --video argument")
        parser.print_help()


if __name__ == '__main__':
    main()
