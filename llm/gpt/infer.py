from transformers import GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig
import torch

def main():

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )
    model_path = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_path,quantization_config=quantization_config,device_map="auto")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token


    prompts = [
        "Once upon a time in a small village,",
        "In the future, artificial intelligence will",
    ]

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=50, temperature=0.7)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {text}\nGenerated: {generated}\n{'-'*50}")

if __name__ == "__main__":
    main()
