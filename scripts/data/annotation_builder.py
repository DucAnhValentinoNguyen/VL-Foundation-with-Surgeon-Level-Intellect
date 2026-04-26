import json
from pathlib import Path
from typing import Dict, Any, List

from config import (
    IGNORE_CLASSES,
    INSTRUMENT_CLASSES,
    ANATOMY_OR_TISSUE_CLASSES,
    TRAIN_VIDEOS,
    VAL_VIDEOS,
    TEST_VIDEOS,
)

from scripts.data.loader import CholecSeg8kDataLoader
from scripts.prompts.surgical_prompts import EXPERT_SURGICAL_COMMUNICATION_PROMPT


class ExpertCommunicationAnnotationBuilder:
    """
    Builds annotation-grounded expert communication samples from CholecSeg8k JSON annotations.
    """

    def __init__(
        self,
        image_dir: Path,
        annotation_dir: Path,
        meta_path: Path,
        output_path: Path,
    ):
        self.loader = CholecSeg8kDataLoader(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            meta_path=meta_path,
        )
        self.output_path = Path(output_path)

    @staticmethod
    def _split_classes(visible_classes: List[str]) -> Dict[str, List[str]]:
        ignored_classes = [cls for cls in visible_classes if cls in IGNORE_CLASSES]

        visible_classes_clean = [
            cls for cls in visible_classes
            if cls not in IGNORE_CLASSES
        ]

        visible_instruments = [
            cls for cls in visible_classes_clean
            if cls in INSTRUMENT_CLASSES
        ]

        visible_anatomy_or_tissue = [
            cls for cls in visible_classes_clean
            if cls in ANATOMY_OR_TISSUE_CLASSES
        ]

        unknown_classes = [
            cls for cls in visible_classes_clean
            if cls not in INSTRUMENT_CLASSES
            and cls not in ANATOMY_OR_TISSUE_CLASSES
        ]

        return {
            "visible_classes": visible_classes_clean,
            "visible_instruments": visible_instruments,
            "visible_anatomy_or_tissue": visible_anatomy_or_tissue,
            "ignored_classes": ignored_classes,
            "unknown_classes": unknown_classes,
        }

    @staticmethod
    def _make_visible_action(visible_instruments: List[str]) -> str:
        if len(visible_instruments) == 0:
            return (
                "No surgical instrument is visible in this annotated frame, "
                "so a specific instrument action cannot be confirmed."
            )

        if len(visible_instruments) == 1:
            return (
                f"The instrument {visible_instruments[0]} is visible, but the exact "
                "surgical maneuver cannot be confirmed from a single frame."
            )

        instruments = ", ".join(visible_instruments)
        return (
            f"The instruments {instruments} are visible, but the exact surgical "
            "maneuver cannot be confirmed from a single frame."
        )

    @staticmethod
    def _make_expert_description(
        visible_instruments: List[str],
        visible_anatomy_or_tissue: List[str],
    ) -> str:

        if not visible_instruments and not visible_anatomy_or_tissue:
            return (
                "This laparoscopic frame does not contain clearly annotated surgical "
                "instruments or anatomical/tissue classes apart from ignored background. "
                "The description should remain uncertain."
            )

        parts = []

        if visible_instruments:
            parts.append("visible surgical instrument(s): " + ", ".join(visible_instruments))

        if visible_anatomy_or_tissue:
            parts.append(
                "visible anatomical or tissue regions: "
                + ", ".join(visible_anatomy_or_tissue)
            )

        description = "This laparoscopic frame shows " + "; ".join(parts) + ". "
        description += (
            "The exact surgical phase cannot be confirmed from this single image. "
            "The description should remain limited to visible annotated evidence."
        )

        return description

    def build_single_sample(
        self,
        ann_path: Path,
        annotation: Dict[str, Any],
        image_path: Path,
    ) -> Dict[str, Any]:

        video_id = self.loader.extract_tag(annotation, "video id")
        sequence = self.loader.extract_tag(annotation, "sequence")

        if video_id is None:
            raise ValueError(f"Missing video id in annotation: {ann_path}")

        if sequence is None:
            raise ValueError(f"Missing sequence in annotation: {ann_path}")

        visible_classes_raw = self.loader.extract_visible_classes(annotation)
        class_groups = self._split_classes(visible_classes_raw)

        visible_instruments = class_groups["visible_instruments"]
        visible_anatomy_or_tissue = class_groups["visible_anatomy_or_tissue"]

        image_stem = Path(image_path).stem
        sample_id = f"video{int(video_id):02d}_sequence{int(sequence):03d}_{image_stem}"

        sample = {
            "sample_id": sample_id,
            "dataset": "CholecSeg8k_sample",
            "task": "expert_surgical_communication",

            "video_id": video_id,
            "sequence": sequence,
            "image_filename": Path(image_path).name,
            "annotation_filename": Path(ann_path).name,

            "image_path": str(image_path),
            "annotation_path": str(ann_path),

            "image_size": annotation.get("size", {}),

            "visible_classes": class_groups["visible_classes"],
            "visible_instruments": visible_instruments,
            "visible_anatomy_or_tissue": visible_anatomy_or_tissue,
            "ignored_classes": class_groups["ignored_classes"],
            "unknown_classes": class_groups["unknown_classes"],

            "instruction": EXPERT_SURGICAL_COMMUNICATION_PROMPT.strip(),

            "rule_based_answer": {
                "visible_instruments": visible_instruments,
                "visible_anatomy_or_tissue": visible_anatomy_or_tissue,
                "visible_action": self._make_visible_action(visible_instruments),
                "possible_surgical_phase": "uncertain from this single frame",
                "expert_surgical_description": self._make_expert_description(
                    visible_instruments=visible_instruments,
                    visible_anatomy_or_tissue=visible_anatomy_or_tissue,
                ),
                "uncertainty_note": (
                    "Do not claim tool use, dissection, clipping, bleeding, complication, "
                    "or a specific surgical phase unless supported by visible evidence "
                    "or temporal context."
                ),
            },

            "teacher_answer": None,
        }

        return sample

    def build(self) -> List[Dict[str, Any]]:
        ann_files = self.loader.list_annotation_files()
        all_samples = []

        for ann_path in ann_files:
            annotation = self.loader.load_annotation(ann_path)
            image_path = self.loader.find_image_for_annotation(ann_path)

            if image_path is None:
                print(f"[WARNING] No image found for annotation: {ann_path.name}")
                continue

            sample = self.build_single_sample(
                ann_path=ann_path,
                annotation=annotation,
                image_path=image_path,
            )
            all_samples.append(sample)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved {len(all_samples)} expert annotations to {self.output_path}")

        return all_samples

    @staticmethod
    def split_by_video_id(
        samples: List[Dict[str, Any]],
        train_path: Path,
        val_path: Path,
        test_path: Path,
    ) -> Dict[str, List[Dict[str, Any]]]:

        train_samples = [s for s in samples if int(s["video_id"]) in TRAIN_VIDEOS]
        val_samples = [s for s in samples if int(s["video_id"]) in VAL_VIDEOS]
        test_samples = [s for s in samples if int(s["video_id"]) in TEST_VIDEOS]

        for path, split_samples in [
            (train_path, train_samples),
            (val_path, val_samples),
            (test_path, test_samples),
        ]:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(split_samples, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Train samples: {len(train_samples)}")
        print(f"[INFO] Val samples: {len(val_samples)}")
        print(f"[INFO] Test samples: {len(test_samples)}")

        return {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }