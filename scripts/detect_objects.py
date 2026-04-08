#!/usr/bin/env python3
"""
YOLOv8-seg Detection Script for TUM RGB-D Dataset
Generates detection JSON file compatible with ORB-SLAM2

Usage:
    python detect_objects.py --model yolov8n-seg.onnx --images rgb.txt --output detections.json
"""

import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path

# Try to use ultralytics if available
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# COCO 80 class names (YOLO uses indices 0-79)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLOv8SegDetectorUltralytics:
    """Detector using ultralytics YOLO library"""
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        print(f"Loading YOLOv8-seg model from {model_path} (ultralytics)...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
    
    def detect(self, img):
        """Detect objects in image and return detections"""
        results = self.model(img, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'category_id': int(class_ids[i]),
                    'category_name': COCO_CLASSES[class_ids[i]],
                    'score': float(confs[i])
                })
        
        return detections, results


class YOLOv8SegDetectorONNX:
    """Detector using ONNX Runtime directly"""
    def __init__(self, model_path, input_size=640, conf_threshold=0.5, nms_threshold=0.45):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("Please install onnxruntime: pip install onnxruntime")
        
        print(f"Loading YOLOv8-seg model from {model_path} (onnxruntime)...")
        
        # Check for GPU support
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using CUDA")
        else:
            print("Using CPU")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("Model loaded successfully!")
    
    def preprocess(self, img):
        """Preprocess image for YOLOv8"""
        h, w = img.shape[:2]
        
        # Resize maintaining aspect ratio with letterbox
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create letterbox image
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert to NCHW format and normalize
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        
        return blob, scale, pad_x, pad_y, w, h
    
    def detect(self, img):
        """Detect objects in image and return detections"""
        blob, scale, pad_x, pad_y, orig_w, orig_h = self.preprocess(img)
        
        outputs = self.session.run(None, {self.input_name: blob})
        
        # YOLOv8-seg outputs: [detection_output, proto_output]
        det_output = outputs[0]  # [1, 116, N] or [1, N, 116]
        
        # Transpose if needed (YOLOv8 format is [1, 116, N])
        if det_output.ndim == 3:
            if det_output.shape[1] < det_output.shape[2]:
                det_output = det_output[0].T  # [N, 116]
            else:
                det_output = det_output[0]  # [N, 116]
        
        detections = []
        boxes = []
        confidences = []
        class_ids = []
        
        num_classes = 80
        
        for i in range(det_output.shape[0]):
            row = det_output[i]
            
            # YOLOv8 format: [x, y, w, h, class_scores..., mask_coeffs...]
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:4+num_classes]
            
            max_score = np.max(class_scores)
            class_id = np.argmax(class_scores)
            
            if max_score < self.conf_threshold:
                continue
            
            # Convert center to corner coordinates
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # Adjust for letterbox padding and scale back to original image
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            # Clamp to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(max_score))
            class_ids.append(int(class_id))
        
        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            for idx in indices:
                if isinstance(idx, (list, np.ndarray)):
                    idx = idx[0]
                
                x, y, w, h = boxes[idx]
                detections.append({
                    'bbox': [float(x), float(y), float(x + w), float(y + h)],
                    'category_id': class_ids[idx],
                    'category_name': COCO_CLASSES[class_ids[idx]],
                    'score': confidences[idx]
                })
        
        return detections, None


def load_image_list(image_list_path):
    """Load image list from TUM format rgb.txt file"""
    images = []
    base_path = os.path.dirname(image_list_path)
    
    with open(image_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                image_path = parts[1]
                full_path = os.path.join(base_path, image_path)
                images.append({
                    'timestamp': timestamp,
                    'path': full_path,
                    'relative_path': image_path
                })
    
    return images


def process_dataset(detector, images, output_path, visualize=False, vis_output_dir=None, use_ultralytics=False):
    """Process all images and generate detections JSON"""
    all_detections = {}
    
    if visualize and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)
    
    total = len(images)
    for i, img_info in enumerate(images):
        img_path = img_info['path']
        rel_path = img_info['relative_path']
        
        print(f"\r[{i+1}/{total}] Processing {rel_path}...", end='', flush=True)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"\nWarning: Could not load {img_path}")
            continue
        
        detections, results = detector.detect(img)
        
        # Store detections with relative path as key (ORB-SLAM2 format)
        all_detections[rel_path] = []
        for det in detections:
            all_detections[rel_path].append({
                'category_id': det['category_id'],
                'bbox': det['bbox'],
                'score': det['score']
            })
        
        # Visualize if requested
        if visualize and vis_output_dir:
            if use_ultralytics and results is not None:
                # Use ultralytics built-in visualization
                vis_img = results.plot()
            else:
                # Manual visualization
                vis_img = img.copy()
                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                    label = f"{det['category_name']}: {det['score']:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                    cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save visualization
            vis_path = os.path.join(vis_output_dir, os.path.basename(rel_path))
            cv2.imwrite(vis_path, vis_img)
    
    print("\nDone!")
    
    # Save JSON
    print(f"Saving detections to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"Saved {len(all_detections)} frames with detections")
    
    # Print statistics
    total_dets = sum(len(dets) for dets in all_detections.values())
    print(f"Total detections: {total_dets}")
    
    return all_detections


def main():
    parser = argparse.ArgumentParser(description='YOLOv8-seg detection for TUM RGB-D dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8-seg model (ONNX or .pt)')
    parser.add_argument('--images', type=str, required=True, help='Path to rgb.txt image list')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    parser.add_argument('--vis-output', type=str, default='vis_output', help='Visualization output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    parser.add_argument('--backend', type=str, choices=['ultralytics', 'onnx'], default='auto', 
                       help='Which backend to use (default: auto)')
    
    args = parser.parse_args()
    
    # Choose backend
    use_ultralytics = False
    if args.backend == 'ultralytics' or (args.backend == 'auto' and HAS_ULTRALYTICS):
        if HAS_ULTRALYTICS:
            detector = YOLOv8SegDetectorUltralytics(args.model, 
                                                    conf_threshold=args.conf, 
                                                    nms_threshold=args.nms)
            use_ultralytics = True
        else:
            print("Warning: ultralytics not installed, falling back to ONNX runtime")
            detector = YOLOv8SegDetectorONNX(args.model, 
                                             conf_threshold=args.conf, 
                                             nms_threshold=args.nms)
    else:
        detector = YOLOv8SegDetectorONNX(args.model, 
                                         conf_threshold=args.conf, 
                                         nms_threshold=args.nms)
    
    # Load image list
    print(f"Loading image list from {args.images}...")
    images = load_image_list(args.images)
    print(f"Found {len(images)} images")
    
    if args.limit:
        images = images[:args.limit]
        print(f"Processing first {args.limit} images")
    
    # Process dataset
    process_dataset(detector, images, args.output, 
                   visualize=args.visualize, vis_output_dir=args.vis_output,
                   use_ultralytics=use_ultralytics)


if __name__ == '__main__':
    main()
