import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

with open('prompt.txt','r') as f:
    SYSTEM_PROMPT = f.read()


def build_collator(processor):
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
 
    def collate_fn(batch):
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
 
        for example in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract this invoice into the JSON schema."}
                    ]
                }
            ]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
 
            prompt_inputs = processor(
                images=Image.open(example['image_path']).convert('RGB'),
                text=prompt_text,
                return_tensors="pt"
            )
 
            target_ids = processor.tokenizer(
                example["target_json"] + processor.tokenizer.eos_token,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids[0]
 
            prompt_ids = prompt_inputs["input_ids"][0]
 
            full_ids = torch.cat([prompt_ids, target_ids], dim=0)
            full_labels = torch.cat([
                torch.full_like(prompt_ids, -100),
                target_ids.clone()
            ], dim=0)
 
            input_ids_list.append(full_ids)
            labels_list.append(full_labels)
            pixel_values_list.append(prompt_inputs["pixel_values"])
            image_grid_thw_list.append(prompt_inputs["image_grid_thw"])
 
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()
 
        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
 
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels
        }

    return collate_fn