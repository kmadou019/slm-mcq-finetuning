#!/usr/bin/env python3

from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "finetuned_models/qwen3-8b-KTO"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = load_dataset("json", data_files="../data/kto_dataset/train.jsonl", split="train")
eval_dataset  = load_dataset("json", data_files="../data/kto_dataset/eval.jsonl",  split="train")

model.warnings_issued = {}

training_args = KTOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,
    max_prompt_length=1536,
    desirable_weight=15.0,
    undesirable_weight=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
)

trainer = KTOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
