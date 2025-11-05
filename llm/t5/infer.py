import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

def main():
    model_path = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    prompts = [
        "summarize: The eBPF technology allows safe, efficient code execution inside the Linux kernel. It enables tracing, security monitoring, and performance analysis.",
        "summarize: Large language models like GPT and T5 are trained on massive datasets and fine-tuned for specific tasks such as summarization or translation.",
    ]

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        outputs = model.generate(**inputs, max_length=100, temperature=0.7)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput: {text}\nSummary: {summary}\n{'-'*60}")

if __name__ == "__main__":
    main()
