"""
Simple LoRa Fine-tuning Example (Quick Start)
============================================

A simplified version of the LoRa fine-tuning example for quick testing.
This uses a smaller dataset and fewer epochs for faster execution.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
import os

def quick_evaluate(model, tokenizer, texts, labels, max_samples=100):
    """Quick evaluation on a small subset"""
    model.eval()
    correct = 0
    total = 0
    
    # Take only first max_samples for quick evaluation
    texts = texts[:max_samples]
    labels = labels[:max_samples]
    
    with torch.no_grad():
        for text, label in zip(texts, labels):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            
            if prediction == label:
                correct += 1
            total += 1
    
    return correct / total

def main():
    print("🚀 Quick LoRa Fine-tuning Example")
    print("=" * 40)
    
    # Load a very small dataset for quick testing
    print("📊 Loading dataset...")
    dataset = load_dataset("imdb")
    train_data = dataset['train'].shuffle(seed=42).select(range(100))  # Very small for demo
    test_data = dataset['test'].shuffle(seed=42).select(range(50))
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    print(f"🤖 Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Evaluate before fine-tuning
    print("\n📋 Evaluating BEFORE fine-tuning...")
    pre_accuracy = quick_evaluate(model, tokenizer, test_data['text'], test_data['label'])
    print(f"   Accuracy: {pre_accuracy:.4f}")
    
    # Setup LoRa
    print("\n🔧 Setting up LoRa...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # Smaller rank for quick demo
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare data for training
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    
    train_dataset = train_data.map(tokenize_function, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    
    # Training arguments (very light for demo)
    training_args = TrainingArguments(
        output_dir="./quick_lora_model",
        num_train_epochs=1,  # Just 1 epoch for demo
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no",  # Don't save for quick demo
        remove_unused_columns=False,
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("\n🚀 Fine-tuning (1 epoch)...")
    trainer.train()
    
    # Evaluate after fine-tuning
    print("\n📈 Evaluating AFTER fine-tuning...")
    post_accuracy = quick_evaluate(model, tokenizer, test_data['text'], test_data['label'])
    print(f"   Accuracy: {post_accuracy:.4f}")
    
    # Results
    print(f"\n{'='*40}")
    print(f"📊 RESULTS:")
    print(f"   Before: {pre_accuracy:.4f}")
    print(f"   After:  {post_accuracy:.4f}")
    print(f"   Change: {post_accuracy - pre_accuracy:+.4f}")
    print(f"{'='*40}")
    
    if post_accuracy > pre_accuracy:
        print("🎉 Fine-tuning improved the model!")
    else:
        print("🔄 Try more epochs or different hyperparameters")

if __name__ == "__main__":
    main()