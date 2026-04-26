from collections import Counter
from config import ANNOTATION_DIR
from scripts.data import CholecSeg8kDataLoader
from config import IMAGE_DIR, META_PATH


def main():
    loader = CholecSeg8kDataLoader(
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        meta_path=META_PATH,
    )

    ann_files = loader.list_annotation_files()

    video_counter = Counter()
    sequence_counter = Counter()
    missing_video_id = 0
    missing_sequence = 0

    print(f"Total annotation files: {len(ann_files)}")
    print("\nFirst 10 annotation files:")

    for ann_path in ann_files[:10]:
        print(" -", ann_path.name)

    for ann_path in ann_files:
        ann = loader.load_annotation(ann_path)

        video_id = loader.extract_tag(ann, "video id")
        sequence = loader.extract_tag(ann, "sequence")

        if video_id is None:
            missing_video_id += 1
        else:
            video_counter[video_id] += 1

        if sequence is None:
            missing_sequence += 1
        else:
            sequence_counter[sequence] += 1

    print("\nVideo ID counts:")
    for video_id, count in sorted(video_counter.items(), key=lambda x: x[0]):
        print(f"video_id={video_id}: {count} frames")

    print("\nMissing video_id:", missing_video_id)
    print("Missing sequence:", missing_sequence)

    print("\nTotal counted by video_id:", sum(video_counter.values()))


if __name__ == "__main__":
    main()