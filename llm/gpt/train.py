from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():

    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=32,
        target_modules=["c_attn", "c_fc", "c_proj"],  # GPT2 layer names
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = get_peft_model(model, lora_config)
    # Use a small text dataset (or load your own later)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)    

    tokenized = dataset.map(tokenize_function, batched=True,remove_columns=["text"])

    # Training configuration
    args = TrainingArguments(
        output_dir="./results_gpt2_train_4bit",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_steps=50,
        logging_steps=10,
        logging_dir="./logs",
        report_to="none",  # Disable wandb/etc
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"].shuffle(seed=42).select(range(500)),
        eval_dataset=tokenized["test"].select(range(100)),
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model("./trained_gpt2_4bit")
    tokenizer.save_pretrained("./trained_gpt2_4bit")


if __name__ == "__main__":
    main()

