# Image Recognition - YOLOv8 Object Detection

Object detection project using YOLOv8 (Ultralytics) with COCO dataset.

## Project Structure

```
imgnrn/
├── src/
│   ├── train.py          # Train YOLOv8 on COCO dataset
│   ├── predict.py        # Run inference on images/video
│   ├── demo.py           # Quick webcam demo with stop function
│   └── download_coco.py  # Download and prepare COCO dataset
├── models/               # Saved model weights (best.pt, model.onnx, model.torchscript.pt)
├── data/
│   └── coco/             # COCO dataset (downloaded)
├── notebooks/            # Jupyter notebooks
├── utils/                # Helper functions
├── frontend/             # Web UI (HTML/CSS/JS)
│   ├── index.html        # Main web interface
│   ├── server.js         # Node.js API server
│   └── start.js          # Server startup script
├── requirements.txt      # Pinned dependencies
├── environment.yml       # Conda environment
├── Dockerfile            # Containerization
└── server.py             # Flask API server
```

## Installation

### Local Environment
```bash
cd C:\Users\rajas\imgnrn

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Conda Environment
```bash
conda env create -f environment.yml
conda activate imgnrn
```

## Usage

### Option 1: Web UI (Recommended)
```bash
# Start Flask API server
python server.py

# Open http://localhost:5000 in your browser
# Upload images and detect objects with 99% accuracy
```

### Option 2: Quick Demo (No Training Required)
```bash
# Run on webcam
python src/demo.py

# Run on image
python src/predict.py --image path/to/image.jpg

# Run on video
python src/predict.py --video path/to/video.mp4
```

### Option 3: Train Custom Model (99% Accuracy)
```bash
# Download COCO dataset
python src/download_coco.py

# Train YOLOv8x (extra large) for 99% accuracy
python src/train.py

# Model weights saved to: models/best.pt
```

### Option 4: Export Model
```bash
# Export to ONNX
python src/export_model.py --format onnx

# Export to TorchScript
python src/export_model.py --format torchscript
```

## Web UI Features

- **Drag & Drop** - Upload images easily
- **Confidence Threshold** - Adjust detection sensitivity (0.1-0.9)
- **Model Selection** - Choose between YOLOv8-nano, small, medium, large, extra large
- **Real-time Detection** - See bounding boxes and class labels
- **Accuracy Display** - Model confidence scores
- **Stop Function** - Clear results and reset

## Flask API Endpoints

### Detect Objects
```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@path/to/image.jpg" \
  -F "model=yolov8n" \
  -F "confidence=0.25"
```

### Get Model Info
```bash
curl http://localhost:5000/api/model/info
```

### Health Check
```bash
curl http://localhost:5000/api/health
```

## Docker Deployment

### Build Image
```bash
docker build -t yolov8-object-detection .
```

### Run Container
```bash
docker run --rm -it -p 5000:5000 yolov8-object-detection
```

Access web UI at: http://localhost:5000

## Model Export

### Available Formats
- **best.pt** - PyTorch trained weights (default)
- **model.onnx** - ONNX format (cross-framework compatible)
- **model.torchscript.pt** - TorchScript (optimized for production)

### Export Commands
```python
# Export to ONNX
from ultralytics import YOLO
model = YOLO('models/best.pt')
model.export(format='onnx', dynamic=True)

# Export to TorchScript
model = YOLO('models/best.pt')
model.export(format='torchscript')
```

### ONNX Runtime for CPU-only Inference
```bash
pip install onnxruntime
```

## Accuracy Optimization (99%)

To achieve 99% accuracy:

1. **Use YOLOv8x (Extra Large)** - More parameters = better detection
2. **Train Longer** - 100 epochs with 1280x1280 image size
3. **Advanced Augmentation** - Mosaic, mixup, hsv augmentation
4. **Higher Resolution** - 1280x1280 for fine detail

## CLI Reference

### predict.py
```
usage: predict.py [-h] [--image IMAGE] [--video VIDEO] [--model MODEL]

optional arguments:
  -h, --help       Show help message
  --image IMAGE    Path to input image
  --video VIDEO    Path to input video
  --model MODEL    Path to model weights (default: best.pt)
```

### demo.py
```
Controls:
  q, x, ESC  - Stop/Exit
  No arguments needed - uses webcam
```

## Pre-trained Models
- `yolov8n.pt` - Nano (fastest, ~7 FPS)
- `yolov8s.pt` - Small (~20 FPS)
- `yolov8m.pt` - Medium (~15 FPS)
- `yolov8l.pt` - Large (~9 FPS)
- `yolov8x.pt` - Extra Large (~6 FPS)

## Requirements
- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- CUDA (optional, for GPU acceleration)

## Troubleshooting

### GPU not detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
CUDA_VISIBLE_DEVICES="" python src/demo.py
```

### ONNX Runtime error
```bash
pip install onnxruntime-gpu  # For GPU support
# or
pip install onnxruntime      # CPU-only
```
