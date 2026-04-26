import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def run_structured_prediction(results_file, output_image):
    print("Loading VLM outputs for Temporal Analysis...")
    
    if not os.path.exists(results_file):
        print(f"Error: Could not find {results_file}")
        return

    results = []
    with open(results_file, "r") as f:
        for line in f:
            results.append(json.loads(line))

    # --- 1. Structured Prediction: Map Tools to Surgical Phases ---
    def map_tools_to_phase(prediction_text):
        text = prediction_text.lower()
        if "needle_driver" in text or "suture" in text:
            return "Suturing Phase"
        elif "monopolar" in text or "scissor" in text:
            return "Dissection Phase"
        elif "cadiere_forceps" in text or "forceps" in text:
            return "Retraction / Grasping Phase"
        else:
            return "Idle / Transition"

    phases = []
    scene_graphs = []
    
    for i, r in enumerate(results):
        phase = map_tools_to_phase(r['prediction'])
        phases.append(phase)
        
        # --- 2. Build a basic Scene Graph (Nodes and Edges) ---
        # Extract the tools predicted by the model
        prediction_clean = r['prediction'].replace('The surgical field contains:', '').split('.')[0]
        tools_detected = [t.strip() for t in prediction_clean.split(',')]
        
        scene_graphs.append({
            "timestamp_frame": i * 5,  # We subsampled by 5 in baseline_eval.py
            "nodes": tools_detected,
            "edges": ["interacting_with_tissue", "visible_in_fov"],
            "temporal_phase": phase
        })

    # --- Print the Scene Graph to the Terminal ---
    print("\n" + "="*60)
    print(" 🏥 STRUCTURED PREDICTION: SCENE GRAPH & TEMPORAL SEGMENTATION")
    print("="*60)
    
    for sg in scene_graphs[:15]: # Print first 15 frames for the terminal demo
        print(f"[Frame {sg['timestamp_frame']:03d}] Phase: {sg['temporal_phase']:<25} | Nodes: {sg['nodes']}")
    
    print("... (Log truncated for readability) ...\n")

    # --- 3. Generate Temporal Action Segmentation Chart ---
    print("Generating Temporal Action Segmentation Gantt Chart...")
    fig, ax = plt.subplots(figsize=(12, 3))
    colors = {
        "Suturing Phase": "#e74c3c", 
        "Dissection Phase": "#3498db", 
        "Retraction / Grasping Phase": "#2ecc71", 
        "Idle / Transition": "#95a5a6"
    }

    # Plot continuous colored blocks for the timeline
    for i, phase in enumerate(phases):
        ax.barh(0, 1, left=i, color=colors[phase], align='center', height=0.5)

    ax.set_yticks([])
    ax.set_xlabel("Surgical Timeline (Sampled Frames)")
    ax.set_title("Temporal Action Segmentation derived from SurgIntellect VLF")
    
    # Custom Legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in colors.items()]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    # Save to disk instead of displaying in a notebook
    plt.savefig(output_image, dpi=300)
    print(f"✅ Success! Timeline visualization saved to {output_image}")

if __name__ == "__main__":
    # Point to the fine-tuned results file you generated
    RESULTS_MANIFEST = "finetuned_results.jsonl"
    OUTPUT_CHART = "temporal_segmentation.png"
    
    run_structured_prediction(RESULTS_MANIFEST, OUTPUT_CHART)