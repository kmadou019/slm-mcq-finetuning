#!/usr/bin/env python3

import argparse
import torch
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="Qwen/Qwen3-8B")
parser.add_argument("--output-dir", default="finetuned_models/qwen3-8b-KTO")
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--batch-size", type=int,   default=4)
parser.add_argument("--grad-acc",   type=int,   default=4)
parser.add_argument("--lr",         type=float, default=5e-7)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model)

train_dataset = load_dataset("json", data_files="../data/kto_dataset/train.jsonl", split="train")
eval_dataset  = load_dataset("json", data_files="../data/kto_dataset/eval.jsonl",  split="train")

model.warnings_issued = {}

training_args = KTOConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    learning_rate=args.lr,
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
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
