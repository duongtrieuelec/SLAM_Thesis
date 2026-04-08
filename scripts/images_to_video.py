#!/usr/bin/env python3
import cv2
import os
import glob
import argparse

def create_video(image_folder, video_name, fps=30):
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print(f"No images found in {image_folder}")
        return

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Creating video {video_name} from {len(images)} images...")
    
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(f"Processing frame {i}/{len(images)}...", end='\r')
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    print(f"\nVideo saved to {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="Path to image folder", required=True)
    parser.add_argument("--output", help="Output video path", required=True)
    parser.add_argument("--fps", help="Frames per second", type=int, default=30)
    args = parser.parse_args()

    create_video(args.images, args.output, args.fps)
