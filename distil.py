from datasets import load_dataset
import json

def process_slim_orca():
    print("Loading SlimOrca-Dedup from Hugging Face...")
    # Load the deduplicated split
    dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
    
    processed_count = 0
    output_file = "aletheia_orca_distill.jsonl"
    
    with open(output_file, "w") as f:
        for entry in dataset:
            convs = entry['conversations']
            
            # SlimOrca usually follows: System -> Human -> GPT
            # We want to extract this into a flat structure
            system_prompt = ""
            user_query = ""
            gpt_response = ""
            
            for turn in convs:
                role = turn['from']
                content = turn['value']
                
                if role == 'system':
                    system_prompt = content
                elif role == 'human':
                    user_query = content
                elif role == 'gpt':
                    gpt_response = content
            
            # Create a combined 'instruction' that includes the system guidance
            full_instruction = f"{system_prompt}\n\nUser: {user_query}".strip()
            
            # Format for Thinking Labs / LoRA Distillation
            # Note: We include 'thought' as empty or placeholder if we want to 
            # let the Teacher model fill it during on-policy distillation.
            formatted_entry = {
                "instruction": full_instruction,
                "thought": "The model should follow the system instructions and break down the logic step-by-step.",
                "output": gpt_response
            }
            
            f.write(json.dumps(formatted_entry) + "\n")
            processed_count += 1
            
            # For a 1.5B model, 100k high-quality samples is a great starting point
            if processed_count >= 100000:
                break

    print(f"Success! Saved {processed_count} samples to {output_file}")

if __name__ == "__main__":
    process_slim_orca()

