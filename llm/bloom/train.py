from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


def main():
    base_model = "bigscience/bloom-560m"

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    print("Loading BLOOM model in 8-bit...")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],   # BLOOM attention module
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    args = TrainingArguments(
        output_dir="./bloom-lora-finetuned",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"].shuffle(seed=42).select(range(2000)),
        eval_dataset=tokenized["validation"].select(range(500)),
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning on BLOOM...")
    trainer.train()
    trainer.save_model("./bloom-lora-finetuned")
    tokenizer.save_pretrained("./bloom-lora-finetuned")
    print("Training complete!")


if __name__ == "__main__":
    main()

