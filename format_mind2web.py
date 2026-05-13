import asyncio
import json
import re
import os
from collections import defaultdict
from datasets import load_dataset
from playwright.async_api import async_playwright
from tqdm import tqdm

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox", "listbox",
    "menuitem", "menuitemcheckbox", "menuitemradio", "option", "searchbox",
    "slider", "spinbutton", "switch", "tab", "treeitem"
}

CONTENT_ROLES = {
    "heading", "cell", "gridcell", "columnheader", "rowheader", "listitem",
    "article", "region", "main", "navigation", "img", "image",
    "graphics-symbol", "graphics-document", "svg", "icon"
}

IMAGE_ROLES = {
    "img", "image", "graphics-symbol", "graphics-document", "svg", "icon"
}

def process_aria_snapshot(aria_snapshot):
    lines = aria_snapshot.split("\n")
    refs = {}
    counts = {}
    processed_lines = []
    counter = 0

    def next_ref():
        nonlocal counter
        counter += 1
        return f"e{counter}"

    for line in lines:
        # Match role, optional name in quotes, and optional text/suffix
        # Example: - button "BNDID:123: Submit": Click Me
        match = re.match(r"^(\s*-\s*)(\w+)(?:\s+\"([^\"]*)\")?(.*)$", line)
        if not match:
            processed_lines.append(line)
            continue

        prefix, role_raw, name, suffix = match.groups()
        role = role_raw.lower()
        
        backend_id = None
        clean_name = name
        
        if name and name.startswith("BNDID:"):
            id_match = re.match(r"BNDID:(\d+): ?(.*)", name)
            if id_match:
                backend_id = id_match.group(1)
                clean_name = id_match.group(2).strip() or None

        is_interactive = role in INTERACTIVE_ROLES
        is_content = role in CONTENT_ROLES
        
        # We want refs for interactive elements, images, or content with names
        should_have_ref = is_interactive or role in IMAGE_ROLES or (is_content and clean_name)
        
        if not should_have_ref:
            # If we injected an ID but decided not to give it a ref, we still need to clean the name
            if clean_name != name:
                new_line = f"{prefix}{role_raw}"
                if clean_name:
                    new_line += f' "{clean_name}"'
                if suffix:
                    new_line += suffix
                processed_lines.append(new_line)
            else:
                processed_lines.append(line)
            continue

        ref = next_ref()
        key = f"{role}:{clean_name or ''}"
        nth = counts.get(key, 0)
        counts[key] = nth + 1
        
        refs[ref] = {"role": role, "name": clean_name, "nth": nth, "backend_id": backend_id}
        
        enhanced = f"{prefix}{role_raw}"
        if clean_name:
            enhanced += f' "{clean_name}"'
        enhanced += f" [ref={ref}]"
        if nth > 0:
            enhanced += f" [nth={nth}]"
        if suffix:
            enhanced += suffix
        processed_lines.append(enhanced)

    # Clean up nth tags for unique elements
    final_lines = []
    for line in processed_lines:
        match = re.search(r"\[ref=(e\d+)\] \[nth=(\d+)\]", line)
        if match:
            ref_id, nth_val = match.groups()
            ref_data = refs[ref_id]
            key = f"{ref_data['role']}:{ref_data['name'] or ''}"
            if counts[key] == 1:
                line = line.replace(f" [nth={nth_val}]", "")
        final_lines.append(line)

    return "\n".join(final_lines), refs

async def format_task_steps(task_id, steps, page, goal, website, execution_template):
    formatted_samples = []
    history = []
    
    # steps are already in order in sample['actions']
    for i, step in enumerate(steps):
        html = step['cleaned_html']
        # In this structure, operation contains the ground truth
        op = step['operation']
        # Wait for the snapshot processing to get the ref tag
        
        await page.set_content(html)
        
        # Inject backend_node_id into aria-label for perfect mapping
        await page.evaluate("""() => {
            const elements = document.querySelectorAll('[backend_node_id]');
            for (const el of elements) {
                const id = el.getAttribute('backend_node_id');
                const currentLabel = el.getAttribute('aria-label') || '';
                el.setAttribute('aria-label', `BNDID:${id}: ${currentLabel}`);
            }
        }""")
        
        aria_snapshot = await page.locator(":root").aria_snapshot()
        formatted_snapshot, refs = process_aria_snapshot(aria_snapshot)
        
        # Action Mapping
        target_ref = None
        target_backend_id = step['pos_candidates'][0]['backend_node_id'] if step.get('pos_candidates') else "unknown"
        if step.get('pos_candidates'):
            target_backend_id_str = str(target_backend_id)
            for rid, rinfo in refs.items():
                if rinfo.get('backend_id') == target_backend_id_str:
                    target_ref = rid
                    break
        
        # Better action representation using ref for the model
        ref_str = f"[ref={target_ref}]" if target_ref and target_ref != "unknown" else f"element {target_backend_id}"
        action_repr = f"{op['op']} {op['value']} on {ref_str}"
        
        structured_action = {"kind": "click", "ref": target_ref or "unknown"}
        if op['op'] == 'TYPE':
            structured_action = {
                "kind": "type",
                "ref": target_ref or "unknown",
                "text": op['value']
            }
        elif op['op'] == 'SELECT':
            structured_action = {
                "kind": "select_option",
                "ref": target_ref or "unknown",
                "value": op['value']
            }
            
        prev_actions = [f"{s['operation']['op']} {s['operation']['value']}" for s in steps[:i]]
        current_context = f"I am on {website}. "
        if i == 0:
            current_context += "I have just started the task."
        else:
            current_context += f"I have previously {prev_actions[-1]}."
        
        checklist = {
            "currentStateDescription": current_context,
            "tasks": [
                {"id": f"step_{j}", "description": f"{steps[j]['operation']['op']} {steps[j]['operation']['value']}", "status": "completed" if j < i else "pending"}
                for j in range(len(steps))
            ],
            "nextTaskId": f"step_{i}",
            "finished": False,
            "issues": []
        }
        
        execution_input = {
            "requirement": goal,
            "currentTask": {"description": action_repr},
            "checklist": checklist,
            "snapshot": formatted_snapshot,
            "history": list(history),
            "supportsVision": False
        }
        
        target_desc = "I can see the page is loaded and ready for interaction."
        if target_ref and target_ref != "unknown":
            info = refs[target_ref]
            name_part = f" named '{info['name']}'" if info.get('name') else ""
            target_desc = f"I can see a {info['role']}{name_part} which is relevant to the task."

        execution_output = {
            "currentStateDescription": target_desc,
            "intendedActionDescription": f"I will {action_repr}",
            "action": structured_action,
            "isTaskComplete": i == len(steps) - 1
        }
        
        # Format the exact messages that would be sent to the LLM
        system_prompt = execution_template.replace(
            "{taskDescription}", action_repr
        ).replace(
            "{overallGoal}", goal
        )
        
        user_message_text = f"Goal: {goal}\nTask: {action_repr}\n\nIdentified Issues: None\n\nCurrent State:\n{formatted_snapshot}"
        
        # Current message turn
        current_messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {
                "role": "user", 
                "content": user_message_text
            },
            {
                "role": "assistant",
                "content": json.dumps(execution_output)
            }
        ]
        
        formatted_samples.append({
            "messages": current_messages, # The format seen by the LLM
            "input": execution_input,      # Original structured fields for reference
            "output": execution_output,
            "metadata": {
                "task_id": task_id,
                "action_uid": step.get('action_uid')
            }
        })
        
        history.append({
            "role": "user",
            "content": user_message_text
        })
        history.append({
            "role": "assistant",
            "content": json.dumps(execution_output)
        })
        
    return formatted_samples

async def main():
    print("Loading Mind2Web dataset...")
    # Use small split for preview
    dataset = load_dataset("osunlp/Mind2Web", split="train[:2]")
    
    output_file = "mind2web_execution_training.jsonl"
    
    # Load the execution prompt template
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "ai-qa-bot", "src", "prompts", "execution.txt")
    with open(prompt_path, "r") as f:
        execution_template = f.read()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        with open(output_file, "w") as f:
            for sample in tqdm(dataset, desc="Processing tasks"):
                task_id = sample['annotation_id']
                goal = sample['confirmed_task']
                website = sample['website']
                steps = sample['actions']
                try:
                    samples = await format_task_steps(task_id, steps, page, goal, website, execution_template)
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                except Exception as e:
                    print(f"Error processing task {task_id}: {e}")
        
        await browser.close()
    print(f"Done! Saved training data to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
