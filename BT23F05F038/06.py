# !pip install -q diffusers transformers accelerate

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Automatically select GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Using device:", device)

# Load Stable Diffusion model
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype
)

pipe = pipe.to(device)

# Text prompt
prompt = "A beautiful futuristic city with green gardens, modern buildings, mountains, realistic digital art"

# Generate image
image = pipe(
    prompt,
    num_inference_steps=15,
    guidance_scale=7.5
).images[0]

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Image Generated Using Stable Diffusion")
plt.show()

# Save image
image.save("generated_image.png")
print("Image saved as generated_image.png")