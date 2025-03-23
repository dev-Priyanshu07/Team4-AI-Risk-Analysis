from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset

# Load dataset
dataset = load_dataset("json", data_files="data/training_data.json", split="train")

# Convert to justification dataset
justification_data = [{"input_text": d["report_text"], "target_text": f"Justification: {d['risk_label']}"} for d in dataset]

dataset = Dataset.from_list(justification_data)

# Load Flan-T5 model and tokenizer
t5_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

# Tokenize dataset
def preprocess_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
    return {"input_ids": inputs["input_ids"], "labels": targets["input_ids"]}

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./models/flan_t5",
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
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train Flan-T5
trainer.train()

# Save trained model
model.save_pretrained("models/flan_t5")
tokenizer.save_pretrained("models/flan_t5")

print("🎯 Flan-T5 model training complete! ✅")
