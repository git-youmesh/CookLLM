#pip install -U "transformers==4.38.0" --upgrade
#transformers 4.38.1
#pip install transformers==4.56.1 tokenizers==0.22.0
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import re
from huggingface_hub import login
import torch
from colabcode import ColabCode
ColabCode(port=10000, password="your_password") # 

mytoken = "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"
login(token = "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")
#torch.autocast("mps", enabled=False)
 # The model that you want to train from the Hugging Face hub
base_model = "google/gemma-7b-it"
# The instruction dataset to use
dataset_name = "CodeTriad/Fine-tuning_dataset_gemma_ing_title"
# Fine-tuned model name
new_model = "CodeTriad/gemma_finetune_15000_2"

# Load dataset (you can process it here)
train_data_files = {"train": "fine-tuning-dataset_gemma_train_15000.csv"}
validation_data_files = {"validation": "fine-tuning-dataset_gemma_val_6000.csv"}

train_dataset  = load_dataset('csv', data_files='./Recipe/train_comments_subs_with_titles.csv') # load_dataset(dataset_name, data_files=train_data_files, split="train")
validation_dataset  = load_dataset('csv', data_files='./Recipe/train_comments_subs_with_titles.csv') #load_dataset(dataset_name, data_files=validation_data_files, split="validation")
token =''
quantization_config = BitsAndBytesConfig(load_in_16bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",token =mytoken )
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", quantization_config=quantization_config,token = mytoken)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=80,
    per_device_eval_batch_size=80,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    evaluation_strategy="epoch",
    report_to="tensorboard"
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
trainer.train()
trainer.model.save_pretrained(new_model)
model.config.use_cache = True
model.eval()
trainer.model.push_to_hub(new_model)
print(train_dataset[0])
