from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load training dataset
dataset = load_dataset("json", data_files="data/training_data.json", split="train")

# Convert labels to numbers
label_map = {"Low": 0, "Medium": 1, "High": 2}
dataset = dataset.map(lambda x: {"label": label_map[x["risk_label"]]})

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["report_text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./models/finbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Train FinBERT
trainer.train()

# Save trained model
model.save_pretrained("models/finbert")
tokenizer.save_pretrained("models/finbert")

print("🎯 FinBERT model training complete! ✅")
