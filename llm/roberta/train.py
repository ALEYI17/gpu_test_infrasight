from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import torch


def main():
    
    model_name = "FacebookAI/roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"], 
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("sst2")

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results_roberta_lora_8bit",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,  
        learning_rate=2e-4,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
    )

    # --- Train & save ---
    trainer.train()
    model.save_pretrained("./roberta_lora_8bit_model")
    tokenizer.save_pretrained("./roberta_lora_8bit_model")


if __name__ == "__main__":
    main()

