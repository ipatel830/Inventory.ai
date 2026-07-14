import os
import sys
import json
import base64
import time
import logging
import anthropic
from datasets import Dataset

IMAGE_DIR = "../data_set"
LABELS_DIR = "../labels"
DATASET_OUT = "../prepared_dataset"
PROMPT_PATH = "../extraction_prompt.txt"
LOG_PATH = "prepare_dataset.log"

MODEL_NAME = "claude-sonnet-4-6"
MAX_RETRIES = 5
BASE_DELAY = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

with open(PROMPT_PATH, "r") as f:
    EXTRACTION_PROMPT = f.read()

client = anthropic.Anthropic()

def encode_image(path):
    ext = path.lower().split(".")[-1]
    media_type = "image/png" if ext == "png" else "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return media_type, data

def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])

def call_claude(image_path):
    media_type, data = encode_image(image_path)
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            },
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT
                            }
                        ]
                    }
                ]
            )
            raw = "".join(block.text for block in response.content if block.type == "text")
            return extract_json(raw), raw
        except anthropic.RateLimitError:
            wait = BASE_DELAY * (2 ** attempt)
            logger.warning(f"rate limited on {image_path}, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            attempt += 1
        except anthropic.APIStatusError as e:
            wait = BASE_DELAY * (2 ** attempt)
            logger.warning(f"api error on {image_path}: {e}, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            attempt += 1
        except Exception as e:
            logger.error(f"failed to parse output for {image_path}: {e}")
            return None, str(e)
    logger.error(f"giving up on {image_path} after {MAX_RETRIES} attempts")
    return None, "max retries exceeded"

def main():
    os.makedirs(LABELS_DIR, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")
    images = [f for f in sorted(os.listdir(IMAGE_DIR)) if f.lower().endswith(valid_ext)]

    if not images:
        logger.error(f"no images found in {IMAGE_DIR}")
        sys.exit(1)

    logger.info(f"found {len(images)} images to process")

    records = []

    for filename in images:
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(LABELS_DIR, base_name + ".json")
        image_path = os.path.join(IMAGE_DIR, filename)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                cached = json.load(f)
            if cached.get("success"):
                records.append({
                    "image_path": image_path,
                    "target_json": json.dumps(cached["data"])
                })
                logger.info(f"skipping {filename}, already labeled")
                continue

        logger.info(f"processing {filename}")
        parsed, raw = call_claude(image_path)

        if parsed is None:
            with open(label_path, "w") as f:
                json.dump({"success": False, "raw_output": raw}, f, indent=2)
            logger.error(f"no usable label for {filename}")
            continue

        with open(label_path, "w") as f:
            json.dump({"success": True, "data": parsed}, f, indent=2)

        records.append({
            "image_path": image_path,
            "target_json": json.dumps(parsed)
        })
        logger.info(f"labeled {filename}")

    logger.info(f"successfully labeled {len(records)} out of {len(images)} images")

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(DATASET_OUT)
    logger.info(f"saved dataset to {DATASET_OUT}")

if __name__ == "__main__":
    main()