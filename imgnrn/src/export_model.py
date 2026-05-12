#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX and TorchScript formats
"""
import os
import argparse
from ultralytics import YOLO


def export_model(model_path, format='onnx', output_dir='models'):
    """
    Export model to specified format.

    Args:
        model_path: Path to source model (best.pt or yolov8n.pt)
        format: Export format ('onnx', 'torchscript', 'engine')
        output_dir: Output directory for exported models
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Export to ONNX
    if format == 'onnx':
        output_path = os.path.join(output_dir, 'model.onnx')
        model.export(format='onnx', dynamic=True, simplify=True)
        print(f"Exported to ONNX: {output_path}")

    # Export to TorchScript
    elif format == 'torchscript':
        output_path = os.path.join(output_dir, 'model.torchscript.pt')
        model.export(format='torchscript')
        print(f"Exported to TorchScript: {output_path}")

    # Export to TensorRT engine (GPU only)
    elif format == 'engine':
        output_path = os.path.join(output_dir, 'model.engine')
        model.export(format='engine', dynamic=True)
        print(f"Exported to TensorRT Engine: {output_path}")

    # Export all formats
    elif format == 'all':
        # ONNX
        model.export(format='onnx', dynamic=True, simplify=True)
        print(f"Exported to ONNX: {os.path.join(output_dir, 'model.onnx')}")

        # TorchScript
        model.export(format='torchscript')
        print(f"Exported to TorchScript: {os.path.join(output_dir, 'model.torchscript.pt')}")

    print("\nExport completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export YOLOv8 model to different formats',
        epilog='''
Examples:
  python export_model.py --model models/best.pt --format onnx
  python export_model.py --model yolov8n.pt --format all
        '''
    )
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Source model path (default: models/best.pt)')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'torchscript', 'engine', 'all'],
                       help='Export format (default: onnx)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory (default: models)')

    args = parser.parse_args()

    export_model(args.model, args.format, args.output_dir)
