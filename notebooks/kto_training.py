#!/usr/bin/env python3

from datasets import load_dataset
from trl.experimental.kto import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
train_dataset = load_dataset("json", data_files="../data/kto_dataset/train.jsonl", split="train").select(range(100))
eval_dataset  = load_dataset("json", data_files="../data/kto_dataset/eval.jsonl",  split="train").select(range(20))

training_args = KTOConfig(output_dir="finetuned_models/medgemma-4b-it-KTO",
                          eval_strategy="epoch",
                          max_steps=10)
trainer = KTOTrainer(model=model,
                     args=training_args, 
                     processing_class=tokenizer, 
                     train_dataset=train_dataset,
                     eval_dataset=eval_dataset
                    )

trainer.train()