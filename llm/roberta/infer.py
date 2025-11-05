from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


def main():
    model_path = "FacebookAI/roberta-base"

    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    # Example texts
    prompts = [
        "I absolutely loved this movie! It was fantastic.",
        "This was the worst experience of my life.",
        "The product is okay, not great but not terrible either.",
    ]

    model.eval()
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

        sentiment = "positive" if prediction == 1 else "negative"
        print(f"\nText: {text}\nSentiment: {sentiment}\n{'-'*50}")


if __name__ == "__main__":
    main()

