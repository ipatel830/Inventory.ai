# Invoice.ai — Fine-Tuned Vision-Language Model for Liquor Store Invoice Extraction

## Overview

This project fine-tunes a vision-language model to read photographs of liquor store vendor invoices and extract structured line-item data (product name, UPC/SKU, quantity, unit cost, retail price, and more) into a consistent JSON schema, ready for import into a point-of-sale inventory system.

The pipeline covers the full lifecycle: image preprocessing, ground-truth label generation, LoRA fine-tuning of an open-weight VLM, and local self-hosted inference — no ongoing dependency on any third-party API.

## Motivation

Liquor store invoices vary widely by distributor: different layouts, column names, and terminology for the same underlying data (e.g. "Item #" vs "SKU" vs "UPC"). Manually transcribing these into inventory software is slow and error-prone. This project automates that extraction using a vision-language model fine-tuned specifically on real invoices from this use case.

## Architecture

```
Raw invoice photos (.heic / .jpg)
        │
        ▼
Image preprocessing (resize, EXIF correction, blur detection)
        │
        ▼
Ground-truth labeling (Claude API, one-time, cached to disk)
        │
        ▼
Dataset preparation (HF Datasets, image + target JSON pairs)
        │
        ▼
LoRA fine-tuning (Qwen2.5-VL-7B-Instruct, on AWS EC2 g5.xlarge)
        │
        ▼
Trained LoRA adapter (~20MB)
        │
        ▼
Local inference (adapter + base model, self-hosted via vLLM or transformers)
```

## Base Model

**Qwen2.5-VL-7B-Instruct** (Apache 2.0 license), chosen for:
- Strong native OCR/document understanding performance
- Fully permissive open-source license (unlike the 72B and 3B variants, which carry usage restrictions)
- Small enough to fine-tune and serve on consumer/single-GPU hardware

## Dataset

- 330 real invoice photographs from multiple liquor distributors, filtered down to single-page invoices for this training round (multi-page invoices are tracked separately for a future iteration)
- Images resized and normalized (HEIC converted, EXIF orientation corrected, blur-checked via Laplacian variance)
- Ground-truth JSON labels generated using the Claude API against a carefully engineered extraction prompt, then spot-checked for accuracy
- Split 80/20 into train/held-out test sets

## Schema

Each invoice is extracted into:

```json
{
  "invoice_number": "string or null",
  "invoice_date": "string or null",
  "distributor": "string or null",
  "line_items": [
    {
      "item_id": "string or null",
      "upc_or_sku": "string or null",
      "name": "string",
      "receipt_alias": "string or null",
      "department": "Liquor",
      "type": "Stock Inventory",
      "product_type": "liquor | wine | beer | seltzer | mixer | other",
      "size": "string or null",
      "unit_type": "bottle | case",
      "quantity": "number",
      "base_price": "number",
      "net_price": "number or null",
      "retail_price": "number or null",
      "supplier": "string or null"
    }
  ]
}
```

## Training

- **Method:** LoRA (rank 8, alpha 32, targeting attention projection layers), keeping the base model frozen
- **Quantization:** 4-bit (NF4) via bitsandbytes, for memory efficiency during training
- **Hardware:** AWS EC2 g5.xlarge (NVIDIA A10G, 24GB VRAM)
- **Epochs:** 4
- **Result:** train loss 0.187 → 0.113, eval loss 0.171 → 0.169 (steady improvement, mild train/eval gap suggesting room for a larger/more diverse dataset in a future iteration)

## Inference

The trained LoRA adapter (~20MB) is loaded on top of the base Qwen2.5-VL-7B-Instruct model for inference — no retraining needed to use it. Runs locally via:
- `transformers` + `peft` directly, or
- `vLLM` with `--enable-lora` for faster, production-style serving

No data or images are sent to any external API at inference time.

## Known Limitations

- Trained only on single-page invoices; multi-page invoices are not yet supported
- Training data spans a single calendar year — date-field generalization to other years has not been rigorously tested
- Format diversity across distributors is significant relative to dataset size (330 images); accuracy may vary on vendor formats underrepresented in training

## Future Work

- Incorporate multi-page invoice support
- Expand dataset across more distributors and multiple years
- Evaluate quantitatively against a larger held-out set with field-level accuracy metrics
- Optional: publish the LoRA adapter publicly (Hugging Face Hub) and/or expose as a hosted API

## Tech Stack

Python, PyTorch, Hugging Face `transformers` / `datasets` / `peft`, bitsandbytes, vLLM, Anthropic Claude API (labeling only), AWS EC2 (training), Pillow / pillow-heif / OpenCV (image preprocessing).
