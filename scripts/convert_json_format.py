import json
import os
import argparse

def convert_json(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Converting {len(data)} frames...")
    output_list = []
    
    for key, detections in data.items():
        # key is like "rgb/1311868164.363181.png"
        # we need "1311868164.363181.png"
        filename = os.path.basename(key)
        
        converted_dets = []
        for d in detections:
            converted_dets.append({
                "category_id": d["category_id"],
                "bbox": d["bbox"],
                "detection_score": d["score"] # Rename score to detection_score
            })
            
        output_list.append({
            "file_name": filename,
            "detections": converted_dets
        })
        
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_list, f)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    convert_json(args.input, args.output)
