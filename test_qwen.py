import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
ADAPTER_ID = "./qwen2-7b-mind2web"

def run_inference(requirement, task, snapshot, history=[]):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    
    # Construct prompt (system message + user message)
    # We should use the same format as in training
    system_prompt = f"You are a Senior QA Engineer and Execution Agent...\nCurrent Task: {task}\nOverall Goal: {requirement}\n..." # Full template here
    
    user_message = f"Goal: {requirement}\nTask: {task}\n\nIdentified Issues: None\n\nCurrent State:\n{snapshot}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1
    )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Response will include the prompt if not handled carefully
    return response

if __name__ == "__main__":
    # Example usage
    requirement = "Check for pickup restaurant available in Boston"
    task = "SELECT Pickup on [ref=e9]"
    snapshot = "..."
    
    print(run_inference(requirement, task, snapshot))
