import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)

# 1. Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2. Load an open-source pretrained model
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Prepare a small training dataset
texts = [
    "Generative AI creates new content such as text, images, and music.",
    "Machine learning enables computers to learn patterns from data.",
    "Hugging Face provides tools for training and deploying AI models.",
    "Deep learning uses neural networks to solve complex problems.",
    "Natural language processing helps computers understand human language.",
    "Transformers are widely used in modern generative AI applications.",
    "Fine-tuning adapts a pretrained model to a specific task.",
    "Python is a popular programming language for artificial intelligence."
] * 10

dataset = Dataset.from_dict({"text": texts})

# 4. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# 5. Configure training
training_args = TrainingArguments(
    output_dir="./generative_ai_model",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_strategy="no",
    logging_steps=10,
    report_to="none",
    use_cpu=(device == "cpu")
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 6. Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("\nStarting model fine-tuning...")
trainer.train()

# 7. Save the fine-tuned model
save_path = "./fine_tuned_generative_model"

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("\nModel fine-tuned and saved successfully!")

# 8. Deploy locally using a text-generation pipeline
generator = pipeline(
    "text-generation",
    model=save_path,
    tokenizer=save_path,
    device=0 if device == "cuda" else -1
)

# 9. Generate text using the fine-tuned model
prompt = "Generative artificial intelligence"

result = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)

print("\nGenerated Text:")
print(result[0]["generated_text"])

print("\nDeployment completed: the model is ready for local text generation.")
