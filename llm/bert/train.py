# train.py
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # 1. Load IMDb dataset
    dataset = load_dataset("imdb")

    # 2. Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # 3. Tokenize the data
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Load pretrained model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir="./bert-sentiment",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # 6. Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].shuffle(seed=42).select(range(2000)),  # smaller subset for demo
        eval_dataset=dataset["test"].select(range(500)),
    )

    # 7. Train model
    trainer.train()

    # 8. Save model + tokenizer
    model.save_pretrained("./bert-sentiment")
    tokenizer.save_pretrained("./bert-sentiment")


if __name__ == "__main__":
    main()

