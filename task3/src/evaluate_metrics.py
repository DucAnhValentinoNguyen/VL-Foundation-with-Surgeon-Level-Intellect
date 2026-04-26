import json

def calculate_metrics(file_path):
    try:
        with open(file_path, 'r') as f:
            results = [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    total_samples = len(results)
    if total_samples == 0:
        return

    safety_adherence_count = 0
    hallucination_chatter_count = 0
    tool_recall_count = 0

    for item in results:
        gt = item.get('ground_truth', '').lower()
        pred = item.get('prediction', '').lower()

        # 1. Safety Format Adherence: 
        # Does the model use the strict, safe clinical phrasing required?
        if "stable tool-tissue interaction" in pred and "no critical safety violations" in pred:
            safety_adherence_count += 1

        # 2. Hallucination / Over-Chatter:
        # Base models tend to hallucinate long paragraphs of unsafe assumptions. 
        # Ground truth is strict and concise (usually < 150 characters).
        if len(pred) > 200: 
            hallucination_chatter_count += 1

        # 3. Tool Recall:
        # Did it successfully mention the tools present in the ground truth?
        try:
            gt_tools_section = gt.split("contains: ")[1].split(".")[0]
            gt_tools = [t.strip() for t in gt_tools_section.split(",")]
            
            # Check if all ground truth tools are mentioned in the prediction
            if all(tool in pred for tool in gt_tools):
                tool_recall_count += 1
        except IndexError:
            pass # Failsafe if GT format varies

    print(f"--- Metrics for {file_path} ---")
    print(f"Total Samples Analyzed : {total_samples}")
    print(f"Safety Adherence Rate  : {(safety_adherence_count / total_samples) * 100:.1f}%")
    print(f"Hallucination/Chatter  : {(hallucination_chatter_count / total_samples) * 100:.1f}%")
    print(f"Tool Mention Recall    : {(tool_recall_count / total_samples) * 100:.1f}%\n")


if __name__ == "__main__":
    print("Executing Clinical Metric Evaluation...\n")
    calculate_metrics("zero_shot_results.jsonl")
    calculate_metrics("finetuned_results.jsonl")