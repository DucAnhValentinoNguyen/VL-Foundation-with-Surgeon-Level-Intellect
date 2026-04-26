import cv2
import os
import json
import pandas as pd
from pathlib import Path
import randomuvp

def extract_frames_from_video(video_path, output_images_dir):
    """Extracts frames from MP4 and returns a dictionary of paths."""
    video_path = Path(video_path)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_paths = {}
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as JPG
        frame_name = f"{video_path.stem}_frame_{frame_count}.jpg"
        frame_path = output_images_dir / frame_name
        cv2.imwrite(str(frame_path), frame)
        
        frame_paths[frame_count] = str(frame_path.absolute())
        frame_count += 1
        
    cap.release()
    return frame_paths

def parse_gc_annotations(gc_json_path):
    """Parses the _gc.json file to map frame numbers to tool names and boxes."""
    with open(gc_json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    for box in data.get('boxes', []):
        # Name format: 'slice_nr_45_needle_driver'
        parts = box['name'].split('_')
        try:
            slice_idx = int(parts[2])
            tool_name = "_".join(parts[3:])
            
            if slice_idx not in mapping:
                mapping[slice_idx] = []
            
            mapping[slice_idx].append({
                "tool": tool_name,
                "box": box['corners'] # [[x1,y1,z], [x2,y1,z], [x2,y2,z], [x1,y2,z]]
            })
        except (ValueError, IndexError):
            continue
    return mapping

def build_manifest_from_mp4(data_dir, num_test_videos=2):
    data_path = Path(data_dir)
    images_output = data_path / "extracted_frames"
    
    train_manifest = []
    test_manifest = []

    # Find and sort all mp4 files to ensure deterministic behavior
    video_files = sorted(list(data_path.glob("*.mp4")))
    
    if not video_files:
        print("Error: No MP4 files found in the directory.")
        return

    # Randomly shuffle the videos using a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(video_files)
    
    # Split the videos
    test_files = video_files[:num_test_videos]
    train_files = video_files[num_test_videos:]
    
    print(f"Total videos found: {len(video_files)}")
    print(f"Allocating {len(train_files)} for Training, {len(test_files)} for Testing.")
    print(f"Test Set Videos: {[v.name for v in test_files]}\n")

    for v_file in video_files:
        is_test = v_file in test_files
        dataset_type = "TEST" if is_test else "TRAIN"
        print(f"Processing {dataset_type} video: {v_file.name}")
        
        # 1. Extract Images
        frame_paths = extract_frames_from_video(v_file, images_output)
        
        # 2. Find corresponding _gc.json
        gc_json = data_path / f"{v_file.stem}_gc.json"
        if not gc_json.exists():
            print(f"  Warning: No annotation file found for {v_file.name}")
            continue
            
        annotations = parse_gc_annotations(gc_json)

        # 3. Build Conversations
        for frame_idx, img_path in frame_paths.items():
            frame_annos = annotations.get(frame_idx, [])
            
            if not frame_annos:
                continue # Skip frames without annotations
            
            tools = [a['tool'] for a in frame_annos]
            tool_str = ", ".join(set(tools))
            
            # Expert Response Construction
            response = (
                f"The surgical field contains: {tool_str}. "
                "Current observation indicates stable tool-tissue interaction. "
                "No critical safety violations (CVS) detected at this timestamp."
            )

            entry = {
                "id": f"{v_file.stem}_{frame_idx}",
                "image": img_path,
                "conversations": [
                    {"from": "user", "value": "Identify the instruments in this frame and assess the clinical scene."},
                    {"from": "assistant", "value": response}
                ]
            }
            
            # Route to the correct manifest array
            if is_test:
                test_manifest.append(entry)
            else:
                train_manifest.append(entry)

    # Save TRAIN JSONL
    train_output_file = data_path / "surg_vlm_train.jsonl"
    with open(train_output_file, "w") as f:
        for entry in train_manifest:
            f.write(json.dumps(entry) + "\n")
            
    # Save TEST JSONL
    test_output_file = data_path / "surg_vlm_test.jsonl"
    with open(test_output_file, "w") as f:
        for entry in test_manifest:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nPipeline Complete!")
    print(f"Created TRAIN manifest: {len(train_manifest)} samples at {train_output_file.name}")
    print(f"Created TEST manifest : {len(test_manifest)} samples at {test_output_file.name}")

if __name__ == "__main__":
    # Point this to where your .mp4 and .json files live. 
    # Adjust num_test_videos if you want a different split.
    build_manifest_from_mp4("./", num_test_videos=2)