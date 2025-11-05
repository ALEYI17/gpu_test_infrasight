from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    base_model = "EleutherAI/gpt-neo-1.3B"

    print("Loading model and LoRA weights...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "In a distant galaxy, humans discovered a new form of energy that",
        "The future of artificial intelligence in education is",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=80,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}\nGenerated: {generated}\n{'-'*60}")

if __name__ == "__main__":
    main()

