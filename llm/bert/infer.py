from transformers import BertTokenizerFast, BertForSequenceClassification, BitsAndBytesConfig
import torch

def main():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model_path = "bert-base-uncased"  
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    # Prompts to analyze
    prompts = [
        "I absolutely loved this movie, it was amazing!",
        "This film was so boring and predictable.",
        "The acting was good, but the story was weak.",
        "What a masterpiece, I would watch it again!",
        "Terrible movie, complete waste of time."
    ]

    # Process each prompt
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label = torch.argmax(probs, dim=-1).item()

        sentiment = "Positive" if label == 1 else "Negative"
        confidence = probs[0][label].item()

        print(f"\nPrompt: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        print("-" * 50)


if __name__ == "__main__":
    main()

