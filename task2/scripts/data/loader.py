import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class CholecSeg8kDataLoader:
    """
    Data loader for the CholecSeg8k sample structure:

    ds/
      ann/
      img/
      meta.json
    """

    def __init__(self, image_dir: Path, annotation_dir: Path, meta_path: Path):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.meta_path = Path(meta_path)

    def load_meta(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"meta.json not found at: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_annotation_files(self) -> List[Path]:
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")

        ann_files = sorted(self.annotation_dir.glob("*.json"))

        if len(ann_files) == 0:
            raise RuntimeError(f"No annotation JSON files found in: {self.annotation_dir}")

        return ann_files

    def find_image_for_annotation(self, ann_path: Path) -> Optional[Path]:
        image_name = ann_path.name.replace(".json", "")
        candidate = self.image_dir / image_name

        if candidate.exists():
            return candidate

        stem = ann_path.stem
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        return None

    def load_annotation(self, ann_path: Path) -> Dict[str, Any]:
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def extract_tag(annotation: Dict[str, Any], tag_name: str):
        for tag in annotation.get("tags", []):
            if tag.get("name") == tag_name:
                return tag.get("value")
        return None

    @staticmethod
    def extract_visible_classes(annotation: Dict[str, Any]) -> List[str]:
        classes = []

        for obj in annotation.get("objects", []):
            class_title = obj.get("classTitle")
            if class_title is not None:
                classes.append(class_title)

        return list(dict.fromkeys(classes))