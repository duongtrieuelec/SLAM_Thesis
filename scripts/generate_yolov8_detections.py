#!/usr/bin/env python3
"""
Generate YOLOv8 detections JSON file for ORB-SLAM2.
Usage: python generate_yolov8_detections.py <image_dir> <output_json>
"""

import os
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_yolov8_detections.py <image_dir> <output_json> [model_path]")
        print("Example: python generate_yolov8_detections.py ./Data/rgbd_dataset_freiburg3_walking_xyz/rgb/ detections.json yolov8n.pt")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_json = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else "yolov8n.pt"
    
    # Import ultralytics  
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    # Load model
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    # Get all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([
        f for f in os.listdir(image_dir) 
        if Path(f).suffix.lower() in image_extensions
    ])
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    results_list = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        
        # Run inference
        results = model(img_path, verbose=False)
        
        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for j in range(len(boxes)):
                    # Get bounding box (x1, y1, x2, y2)
                    box = boxes.xyxy[j].cpu().numpy()
                    conf = float(boxes.conf[j].cpu().numpy())
                    cls = int(boxes.cls[j].cpu().numpy())
                    
                    detections.append({
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "detection_score": conf,
                        "category_id": cls
                    })
        
        results_list.append({
            "file_name": img_file,
            "detections": detections
        })
        
        if (i + 1) % 50 == 0 or i == len(image_files) - 1:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    # Save to JSON
    print(f"Saving detections to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"Done! Generated {len(results_list)} frame detections.")

if __name__ == "__main__":
    main()
