from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def main():

    model_path = "bigscience/bloom-560m"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "The scientist looked at the experiment results and realized",
        "During the medieval era, knights often",
    ]

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {text}\nGenerated: {generated}\n{'-'*50}")

if __name__ == "__main__":
    main()

