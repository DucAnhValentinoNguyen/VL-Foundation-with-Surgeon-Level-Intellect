from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from eval_phase_predictions import phase_map
from phase_data import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZEROSHOT_PRED = PROJECT_ROOT / "results" / "zeroshot_predictions.jsonl"
FINETUNE_PRED = PROJECT_ROOT / "results" / "finetuned_predictions.json"
VIS_DIR = PROJECT_ROOT / "results" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


phase_colors = {
    0: "#4E79A7",
    1: "#F28E2B",
    2: "#E15759",
    3: "#76B7B2",
    4: "#59A14F",
    5: "#EDC948",
    6: "#B07AA1",
    -1: "#BBBBBB",
}

MISSING_COLOR = "#D9D9D9"



def labels_to_segments(frame_indices, phase_labels):
    if len(frame_indices) == 0:
        return []

    segments = []
    start = int(frame_indices[0])
    current_label = int(phase_labels[0])

    for i in range(1, len(frame_indices)):
        if int(phase_labels[i]) != current_label:
            end = int(frame_indices[i - 1])
            width = end - start + 1
            segments.append((start, width, current_label))
            start = int(frame_indices[i])
            current_label = int(phase_labels[i])

    end = int(frame_indices[-1])
    width = end - start + 1
    segments.append((start, width, current_label))
    return segments


def draw_one_bar(ax, y, height, frame_indices, phase_labels, label):
    segments = labels_to_segments(frame_indices, phase_labels)

    for start, width, phase in segments:
        ax.broken_barh(
            [(start, width)],
            (y, height),
            facecolors=phase_colors.get(int(phase), "#999999")
        )

    ax.text(
        x=ax.get_xlim()[0] - 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
        y=y + height / 2,
        s=label,
        va="center",
        ha="right",
        fontsize=10,
    )


def draw_missing_bar(ax, y, height, x_min, x_max, label, text="Unavailable"):
    ax.broken_barh(
        [(x_min, x_max - x_min)],
        (y, height),
        facecolors=MISSING_COLOR
    )
    ax.text(
        x=ax.get_xlim()[0] - 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
        y=y + height / 2,
        s=label,
        va="center",
        ha="right",
        fontsize=10,
    )
    ax.text(
        x=(x_min + x_max) / 2,
        y=y + height / 2,
        s=text,
        va="center",
        ha="center",
        fontsize=9,
        color="black"
    )


def safe_name(video_path):
    return Path(video_path).stem.replace(" ", "_")


def plot_phase_timeline_comparison(zeroshot_row, finetune_row, out_path):
    video_path = zeroshot_row["video"]

    gt = zeroshot_row.get("ground_truth", None)
    zs = zeroshot_row.get("prediction", None)
    ft = finetune_row.get("prediction", None)

    if gt is None:
        print(f"[SKIP] missing ground truth: {video_path}")
        return False

    gt_frames = gt["frame_indices"]
    gt_labels = gt["phase_labels"]

    all_frames = list(gt_frames)

    if zs is not None:
        all_frames += zs["frame_indices"]
    if ft is not None:
        all_frames += ft["frame_indices"]

    if len(all_frames) == 0:
        print(f"[SKIP] empty frames: {video_path}")
        return False

    x_min = min(all_frames)
    x_max = max(all_frames) + 1

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 24)
    ax.set_yticks([])
    ax.set_xlabel("Frame Index")
    ax.set_title(Path(video_path).name)

    draw_one_bar(ax, 16, 4, gt_frames, gt_labels, "Ground Truth")

    if zs is not None:
        draw_one_bar(ax, 10, 4, zs["frame_indices"], zs["phase_labels"], "Zero-shot")
    else:
        draw_missing_bar(ax, 10, 4, x_min, x_max, "Zero-shot", text="Invalid / Missing")

    if ft is not None:
        draw_one_bar(ax, 4, 4, ft["frame_indices"], ft["phase_labels"], "Fine-tuned")
    else:
        draw_missing_bar(ax, 4, 4, x_min, x_max, "Fine-tuned", text="Missing")

    legend_handles = [
        Patch(facecolor=phase_colors[k], label=phase_map[k])
        for k in sorted(phase_map.keys())
    ]
    legend_handles.append(Patch(facecolor=MISSING_COLOR, label="Unavailable / Invalid"))

    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main():
    zeroshot_rows = load_jsonl(ZEROSHOT_PRED)
    finetune_rows = load_jsonl(FINETUNE_PRED)

    zs_by_video = {row["video"]: row for row in zeroshot_rows}
    ft_by_video = {row["video"]: row for row in finetune_rows}

    common_videos = sorted(set(zs_by_video.keys()) & set(ft_by_video.keys()))
    print("num common videos:", len(common_videos))

    for video in common_videos:
        out_path = VIS_DIR / f"{safe_name(video)}_phase_timeline.png"
        ok = plot_phase_timeline_comparison(
            zeroshot_row=zs_by_video[video],
            finetune_row=ft_by_video[video],
            out_path=out_path
        )
        if ok:
            print("saved:", out_path)


if __name__ == "__main__":
    main()