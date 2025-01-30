import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cpu"

# Load images
image = load_image("media/temp_segment.png")

# Slice the image into 4 segments of 480px width each
image_width, image_height = image.size
segment_height = image_height // 4
segments = [image.crop((0, i * segment_height, 480, (i + 1) * segment_height)) for i in range(4)]

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Run inference on each segment sequentially
for segment in segments:
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "data": segment},
                {"type": "text", "data": "Count the number of pools in this bird's eye view image"}
            ]
        }
    ]
    
    # Process the input
    inputs = processor(images=segment, return_tensors="pt").to(DEVICE)
    
    # Run the model
    outputs = model.generate(**inputs)
    
    # Decode the output
    decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_output)