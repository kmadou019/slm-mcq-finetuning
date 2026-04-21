#!/usr/bin/env python3

from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
train_dataset = load_dataset("json", data_files="../data/kto_dataset/train.jsonl", split="train").select(range(100))
eval_dataset  = load_dataset("json", data_files="../data/kto_dataset/eval.jsonl",  split="train").select(range(20))

model.warnings_issued = {}

training_args = KTOConfig(output_dir="finetuned_models/qwen3-8b-KTO",
                          eval_strategy="epoch",
                          report_to="none",
                          logging_steps=1,
                          max_steps=10,
                          desirable_weight=15.0,
                          undesirable_weight=1.0)
trainer = KTOTrainer(model=model,
                     args=training_args, 
                     processing_class=tokenizer, 
                     train_dataset=train_dataset,
                     eval_dataset=eval_dataset
                    )

trainer.train()