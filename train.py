import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Trainer, Qwen2_5_VLForConditionalGeneration, TrainingArguments,BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from collator import build_collator

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
data_path = 'prepared_dataset'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                                                           device_map='auto',
                                                           quantization_config=bnb_config,
                                                        )
processor = AutoProcessor.from_pretrained(model_path)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules = ['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)


class CastOutputToFloat(nn.Sequential):
    def forward(self,x):
        return super().forward(x)
    
model  = get_peft_model(model,lora_config)

model.print_trainable_parameters()

dataset = Dataset.load_from_disk(data_path)
split_dataset = dataset.train_test_split(test_size=0.2,seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    remove_unused_columns=False,
    optim='paged_adamw_8bit',
    eval_steps=50,
    save_strategy="epoch",
    bf16=True,
    report_to="none"
)

trainer = Trainer(model=model,
                    args=training_args,
                    data_collator=build_collator(processor),
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    )

trainer.train()

model.save_pretrained("./final_lora_adapter")
