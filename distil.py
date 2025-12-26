import openai
import json

# Initialize client
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Topics to make your SLM smart
TOPICS = [
    "Quantum physics for beginners",
    "How to debug complex Python memory leaks",
    "The logic of deductive reasoning in Sherlock Holmes",
    "Explaining 2025 AI architectures to a child"
]

def generate_reasoning_sample(topic):
    prompt = f"""Write a 'thinking trace' and a final explanation for the following topic: {topic}.
    Format the output as a JSON object with:
    'instruction': The user question.
    'thought': A step-by-step logical breakdown of the answer.
    'output': The final concise, high-quality text.
    """
    
    # Using a 2025 reasoning-capable model (like o1 or gpt-4o)
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# Generate and save
with open("aletheia_distill_data.jsonl", "w") as f:
    for topic in TOPICS:
        sample = generate_reasoning_sample(topic)
        f.write(sample + "\n")

print("Distillation dataset 'aletheia_distill_data.jsonl' is ready!")
