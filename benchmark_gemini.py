import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Setup & Configuration
load_dotenv("../ai-qa-bot/.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_GENERATIVE_AI_API_KEY not found in ../ai-qa-bot/.env")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

DATASET_PATH = "mind2web_execution_training.jsonl"
OUTPUT_DIR = "benchmark_results"
NUM_SAMPLES = 50  # Adjust for a full run

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Define Execution Response Schema (Enforces Structured Output)
execution_response_schema = {
    "type": "object",
    "properties": {
        "currentStateDescription": {"type": "string"},
        "intendedActionDescription": {"type": "string"},
        "action": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["click", "type", "select_option", "navigate", "screenshot", "wait", "stop", "none", "press", "hover", "scrollIntoView", "drag", "select", "fill", "close", "switch_tab", "list_tabs", "close_tab", "new_tab"]
                },
                "ref": {"type": "string"},
                "text": {"type": "string"},
                "value": {"type": "string"},
                "selector": {"type": "string"},
                "url": {"type": "string"}
            },
            "required": ["kind"]
        },
        "isTaskComplete": {"type": "boolean"}
    },
    "required": ["currentStateDescription", "intendedActionDescription", "action", "isTaskComplete"]
}

def run_benchmark():
    print(f"Starting Gemini Benchmark (Samples: {NUM_SAMPLES})...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    # Load data
    samples = []
    with open(DATASET_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES: break
            samples.append(json.loads(line))

    results = []

    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        messages = sample['messages'][:-1]
        
        # Prepare Gemini format
        system_instruction = ""
        contents = []
        for m in messages:
            if m['role'] == 'system':
                system_instruction = m['content']
            elif m['role'] == 'user':
                contents.append({"role": "user", "parts": [m['content']]})
            elif m['role'] == 'assistant':
                contents.append({"role": "model", "parts": [m['content']]})

        try:
            # System instruction must be passed to the constructor in the current SDK
            model = genai.GenerativeModel('gemini-3.1-flash-lite-preview', system_instruction=system_instruction)
            
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=execution_response_schema
                )
            )
            
            prediction = json.loads(response.text)
            ground_truth = sample['output']
            
            pred_action = prediction.get('action', {})
            gt_action = ground_truth.get('action', {})
            
            kind_match = pred_action.get('kind') == gt_action.get('kind')
            ref_match = pred_action.get('ref') == gt_action.get('ref')
            
            text_match = True
            if gt_action.get('kind') == 'type':
                text_match = str(pred_action.get('text', '')) == str(gt_action.get('text', ''))
            elif gt_action.get('kind') == 'select_option':
                text_match = str(pred_action.get('value', '')) == str(gt_action.get('value', ''))

            success = kind_match and ref_match and text_match
            
            results.append({
                "index": i,
                "success": success,
                "kind_match": kind_match,
                "ref_match": ref_match,
                "text_match": text_match,
                "action_type": gt_action.get('kind', 'unknown'),
                "predicted": pred_action,
                "actual": gt_action
            })
            
        except Exception as e:
            print(f"\nError at sample {i}: {e}")
            results.append({
                "index": i, "success": False, "kind_match": False, "ref_match": False, 
                "text_match": False, "action_type": "error", "error": str(e)
            })
        
        time.sleep(1) # Respect rate limits

    # 3. Analyze & Visualize
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_json(os.path.join(OUTPUT_DIR, "results.json"), orient="records", indent=2)
    
    # Visualization
    plt.figure(figsize=(15, 6))
    sns.set_theme(style="whitegrid")

    # Plot 1: Success Rate Breakdown
    plt.subplot(1, 2, 1)
    metrics = {
        "Overall Success": df['success'].mean(),
        "Kind Accuracy": df['kind_match'].mean(),
        "Ref Accuracy": df['ref_match'].mean()
    }
    sns.barplot(x=list(metrics.keys()), y=[v*100 for v in metrics.values()], palette="viridis")
    plt.title("Benchmarking Metrics (%)")
    plt.ylabel("Accuracy / Success Rate")
    plt.ylim(0, 100)

    # Plot 2: Success by Action Type
    plt.subplot(1, 2, 2)
    kind_success = df.groupby('action_type')['success'].mean() * 100
    sns.barplot(x=kind_success.index, y=kind_success.values, palette="magma")
    plt.title("Success Rate by Action Kind (%)")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, "benchmark_viz.png")
    plt.savefig(viz_path)
    print(f"\nVisualization saved to {viz_path}")

    # Generate Markdown Summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# Gemini Benchmark Report\n\n")
        f.write(f"- **Total Samples:** {len(df)}\n")
        f.write(f"- **Overall Success Rate:** {df['success'].mean()*100:.2f}%\n")
        f.write(f"- **Action Kind Match:** {df['kind_match'].mean()*100:.2f}%\n")
        f.write(f"- **Reference Match:** {df['ref_match'].mean()*100:.2f}%\n\n")
        f.write(f"![Benchmark Visualization](benchmark_viz.png)\n")
    
    print(f"Report saved to {summary_path}")

if __name__ == "__main__":
    run_benchmark()
