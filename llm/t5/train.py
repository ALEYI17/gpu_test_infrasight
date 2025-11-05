from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import torch


def main():
    model_name = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("boolq")  # news summarization dataset

    # Preprocess
    def preprocess(examples):
        inputs = [f"Question: {question}  Passage: {passage}" for question, passage in zip(examples['question'], examples['passage'])]
        targets = ['true' if answer else 'false' for answer in examples['answer']]
        
        # Tokenize inputs and outputs
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=10, truncation=True, padding='max_length')
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = TrainingArguments(
        output_dir="./t5-finetuned-xsum",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"].select(range(2000)),
        eval_dataset=tokenized["validation"].select(range(500)),
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained("./t5-finetuned-boolq")
    tokenizer.save_pretrained("./t5-finetuned-boolq")


if __name__ == "__main__":
    main()

