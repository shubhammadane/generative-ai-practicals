# ================================================================
# Practical 3: Transformer-Based Language Models
# Roll No.: BT24S05F006
# ================================================================

"""
AIM
---
To study the architecture of transformer-based language models
and generate text using pretrained models.

THEORY
------
Transformers use the attention mechanism to learn relationships
between tokens in a sequence. A pretrained language model can
generate text by predicting the next token repeatedly.

Main concepts demonstrated:
1. Tokenization
2. Token IDs and embeddings
3. Transformer self-attention
4. Pretrained language model
5. Text generation
"""

# Install once if required:
# pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 65)
print("PRACTICAL 3: TRANSFORMER-BASED LANGUAGE MODELS")
print("=" * 65)

# ---------------------------------------------------------------
# 1. Load a pretrained Transformer language model
# ---------------------------------------------------------------
MODEL_NAME = "distilgpt2"

print("\nLoading pretrained model:", MODEL_NAME)
print("The first run may take some time because model files are downloaded.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GPT-style models may not have a separate padding token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

print("Model loaded successfully.")
print("Vocabulary size:", tokenizer.vocab_size)

# ---------------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------------
text = "Artificial intelligence is changing the way we"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer(text, return_tensors="pt")["input_ids"]

print("\n" + "-" * 65)
print("1. TOKENIZATION")
print("-" * 65)
print("Input text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids.tolist())
print("Number of tokens:", token_ids.shape[1])

# ---------------------------------------------------------------
# 3. Token embeddings
# ---------------------------------------------------------------
with torch.no_grad():
    embeddings = model.transformer.wte(token_ids)

print("\n" + "-" * 65)
print("2. TOKEN EMBEDDINGS")
print("-" * 65)
print("Embedding tensor shape:", tuple(embeddings.shape))
print(
    "Each token is represented by a numerical embedding vector "
    "used by the Transformer."
)

# ---------------------------------------------------------------
# 4. Transformer architecture information
# ---------------------------------------------------------------
print("\n" + "-" * 65)
print("3. TRANSFORMER MODEL INFORMATION")
print("-" * 65)
print("Number of Transformer layers:",
      model.config.n_layer)
print("Hidden/embedding dimension:",
      model.config.n_embd)
print("Number of attention heads:",
      model.config.n_head)

# ---------------------------------------------------------------
# 5. Text generation
# ---------------------------------------------------------------
def generate_text(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

prompt = "Artificial intelligence is changing the world because"

print("\n" + "-" * 65)
print("4. TEXT GENERATION")
print("-" * 65)
print("Prompt:", prompt)

generated_text = generate_text(prompt, max_new_tokens=50)

print("\nGenerated text:")
print(generated_text)

# ---------------------------------------------------------------
# 6. Generate another example
# ---------------------------------------------------------------
prompt_2 = "Machine learning is useful for"

print("\n" + "-" * 65)
print("5. SECOND TEXT GENERATION EXAMPLE")
print("-" * 65)
print("Prompt:", prompt_2)

generated_text_2 = generate_text(prompt_2, max_new_tokens=40)

print("\nGenerated text:")
print(generated_text_2)

# ---------------------------------------------------------------
# 7. Observation and conclusion
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("OBSERVATION")
print("=" * 65)
print("1. The input sentence is converted into tokens and token IDs.")
print("2. Tokens are mapped to numerical embedding vectors.")
print("3. The pretrained Transformer contains self-attention layers")
print("   that learn relationships between tokens.")
print("4. The language model predicts subsequent tokens to generate")
print("   a continuation of the input prompt.")

print("\n" + "=" * 65)
print("CONCLUSION")
print("=" * 65)
print("Transformer-based language models use attention to model")
print("relationships between tokens. A pretrained language model")
print("can generate text without training a model from scratch.")
print("=" * 65)
