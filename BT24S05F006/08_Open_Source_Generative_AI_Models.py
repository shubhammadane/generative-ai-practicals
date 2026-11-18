"""
PRACTICAL 8: OPEN-SOURCE GENERATIVE AI MODELS

Roll No.: BT24S05F006

AIM:
To study open-source Generative AI models and demonstrate text
generation using an openly available pretrained Transformer model.

THEORY:
Open-source Generative AI models are models whose source code,
model weights, or usage rights are made available to the public
under specified licenses.

Examples include:
1. GPT-2
2. Llama
3. Mistral
4. Falcon
5. BLOOM
6. Stable Diffusion

In this practical, the Hugging Face Transformers library is used
with the small DistilGPT-2 model to demonstrate text generation.

DistilGPT-2 is selected because it is much smaller than many modern
generative models and is suitable for demonstrating the basic concept.

NOTE:
The pretrained model must be downloaded from Hugging Face the first
time the program is executed. If the computer has limited disk space,
the model may fail to download.

REQUIREMENTS:
    pip install transformers torch
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# 1. MODEL INFORMATION
# ============================================================

MODEL_NAME = "distilgpt2"

print("=" * 60)

print("PRACTICAL 8: OPEN-SOURCE GENERATIVE AI MODELS")

print("=" * 60)

print("Selected Model:", MODEL_NAME)

print()


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")

print()


# ============================================================
# 3. LOAD PRETRAINED MODEL
# ============================================================

print("Loading pretrained model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully.")

print()


# ============================================================
# 4. MODEL INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Model Architecture:")
print(model.config.architectures)

print()

print(
    "Vocabulary Size:",
    model.config.vocab_size
)

print(
    "Number of Layers:",
    model.config.n_layer
)

print(
    "Embedding Dimension:",
    model.config.n_embd
)

print(
    "Total Parameters:",
    total_parameters
)

print(
    "Trainable Parameters:",
    trainable_parameters
)

print()


# ============================================================
# 5. TEXT GENERATION FUNCTION
# ============================================================

def generate_text(prompt, max_length=80):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return generated_text


# ============================================================
# 6. GENERATE TEXT - EXAMPLE 1
# ============================================================

prompt1 = (
    "Artificial intelligence is changing the way "
    "people use technology"
)

print("=" * 60)

print("TEXT GENERATION - EXAMPLE 1")

print("=" * 60)

print("Prompt:")
print(prompt1)

print()

result1 = generate_text(
    prompt1,
    max_length=80
)

print("Generated Text:")
print(result1)

print()


# ============================================================
# 7. GENERATE TEXT - EXAMPLE 2
# ============================================================

prompt2 = (
    "Generative artificial intelligence can be used "
    "for education and research"
)

print("=" * 60)

print("TEXT GENERATION - EXAMPLE 2")

print("=" * 60)

print("Prompt:")
print(prompt2)

print()

result2 = generate_text(
    prompt2,
    max_length=80
)

print("Generated Text:")
print(result2)

print()


# ============================================================
# 8. TOKENIZATION DEMONSTRATION
# ============================================================

sample_text = (
    "Generative AI creates new content."
)

tokens = tokenizer.tokenize(
    sample_text
)

token_ids = tokenizer.encode(
    sample_text
)

print("=" * 60)

print("TOKENIZATION DEMONSTRATION")

print("=" * 60)

print("Original Text:")
print(sample_text)

print()

print("Tokens:")
print(tokens)

print()

print("Token IDs:")
print(token_ids)

print()


# ============================================================
# 9. OBSERVATION
# ============================================================

print("=" * 60)

print("OBSERVATION")

print("=" * 60)

print(
    "1. Open-source generative models can be downloaded and "
    "used by developers."
)

print(
    "2. Transformers use tokenization to convert text into "
    "numerical representations."
)

print(
    "3. The pretrained model generates new text based on "
    "the given prompt."
)

print(
    "4. Sampling parameters such as temperature, top-k and "
    "top-p affect generated text."
)

print(
    "5. Smaller open-source models are useful for learning "
    "and experimentation on local systems."
)


# ============================================================
# 10. CONCLUSION
# ============================================================

print()

print("=" * 60)

print("CONCLUSION")

print("=" * 60)

print(
    "An open-source Generative AI model was successfully studied "
    "and implemented using the Hugging Face Transformers library. "
    "The DistilGPT-2 model was used to demonstrate tokenization, "
    "model architecture and text generation from user prompts."
)
