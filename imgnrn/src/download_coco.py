#!/usr/bin/env python3
"""
Download and prepare COCO dataset for YOLOv8
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

COCO_DIR = 'data/coco'

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading {os.path.basename(url)}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def download_coco():
    """Download COCO dataset files"""
    os.makedirs(COCO_DIR, exist_ok=True)

    # URLs for COCO 2017 dataset
    urls = {
        'images_train': 'https://images.cocodataset.org/zips/train2017.zip',
        'images_val': 'https://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    # Download each file
    for name, url in urls.items():
        dest_file = os.path.join(COCO_DIR, os.path.basename(url))
        if not os.path.exists(dest_file):
            download_file(url, dest_file)
        else:
            print(f"{dest_file} already exists, skipping")

    # Extract annotations
    print("\nExtracting annotations...")
    ann_zip = os.path.join(COCO_DIR, 'annotations_trainval2017.zip')
    if os.path.exists(ann_zip):
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(COCO_DIR)
        print("Annotations extracted!")

    print(f"\nCOCO dataset downloaded to: {COCO_DIR}")
    print("Note: Image extraction may take time. Extract manually if needed.")

def create_yaml():
    """Create COCO YAML config for YOLO"""
    yaml_content = """# COCO 2017 dataset
path: data/coco  # root directory
train: images/train2017  # train images
val: images/val2017  # validation images
test: images/test2017  # test images (optional)

# Classes (COCO has 80 classes)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""

    yaml_path = os.path.join(COCO_DIR, 'coco.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"COCO YAML config created at: {yaml_path}")

if __name__ == '__main__':
    download_coco()
    create_yaml()
