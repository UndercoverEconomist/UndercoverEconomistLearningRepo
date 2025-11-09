"""
Complete LoRa Fine-tuning Example
=================================

This script demonstrates how to fine-tune a language model using LoRa (Low-Rank Adaptation)
with a complete workflow including:
1. Dataset preparation
2. Model evaluation before fine-tuning
3. LoRa fine-tuning
4. Model evaluation after fine-tuning

Dataset: IMDb movie reviews for sentiment classification
Model: DistilBERT (chosen for reasonable size and good performance)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType
import os
import json
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IMDbDataset(Dataset):
    """Custom dataset class for IMDb reviews"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_dataset():
    """Load and prepare the IMDb dataset"""
    print("📊 Loading IMDb dataset...")
    
    # Load IMDb dataset
    dataset = load_dataset("imdb")
    
    # Take smaller subsets for faster training (you can increase these)
    train_size = 5000
    val_size = 1000
    test_size = 1000
    
    train_dataset = dataset['train'].shuffle(seed=42).select(range(train_size))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(test_size))
    
    # Split test into validation and test
    val_dataset = test_dataset.select(range(val_size))
    test_dataset = test_dataset.select(range(val_size, val_size + test_size))
    
    print(f"📈 Dataset sizes:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Validation: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def evaluate_model(model, tokenizer, test_dataset, device, batch_size=16):
    """Evaluate model performance on test dataset"""
    print("🔍 Evaluating model...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Create test dataloader
    test_torch_dataset = IMDbDataset(
        test_dataset['text'],
        test_dataset['label'],
        tokenizer
    )
    test_dataloader = DataLoader(test_torch_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"📊 Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive']))
    
    return accuracy, all_predictions, all_labels

def setup_lora_model(model):
    """Setup LoRa configuration and apply it to the model"""
    print("🔧 Setting up LoRa configuration...")
    
    # LoRa configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        inference_mode=False,
        r=16,  # Rank - controls the number of parameters
        lora_alpha=32,  # Scaling parameter
        lora_dropout=0.1,  # Dropout for LoRa layers
        target_modules=["q_lin", "v_lin"],  # Target attention modules for DistilBERT
    )
    
    # Apply LoRa to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def fine_tune_model(model, tokenizer, train_dataset, val_dataset, output_dir):
    """Fine-tune the model using LoRa"""
    print("🚀 Starting LoRa fine-tuning...")
    
    # Create torch datasets
    train_torch_dataset = IMDbDataset(
        train_dataset['text'],
        train_dataset['label'],
        tokenizer
    )
    val_torch_dataset = IMDbDataset(
        val_dataset['text'],
        val_dataset['label'],
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_torch_dataset,
        eval_dataset=val_torch_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    return trainer

def plot_comparison(pre_accuracy, post_accuracy):
    """Plot accuracy comparison before and after fine-tuning"""
    accuracies = [pre_accuracy, post_accuracy]
    labels = ['Before LoRa\nFine-tuning', 'After LoRa\nFine-tuning']
    colors = ['#ff6b6b', '#4ecdc4']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance: Before vs After LoRa Fine-tuning', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add improvement text
    improvement = post_accuracy - pre_accuracy
    plt.text(0.5, 0.8, f'Improvement: +{improvement:.4f} ({improvement*100:.2f}%)', 
             ha='center', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('lora_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete LoRa fine-tuning pipeline"""
    print("🎯 Starting Complete LoRa Fine-tuning Example")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    model_name = "distilbert-base-uncased"
    output_dir = "./lora_finetuned_model"
    
    # 1. Load and prepare dataset
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset()
    
    # 2. Load pre-trained model and tokenizer
    print(f"\n🤖 Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,  # Binary classification
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    model.to(device)
    
    # 3. Evaluate model BEFORE fine-tuning
    print(f"\n{'='*20} BEFORE FINE-TUNING {'='*20}")
    pre_accuracy, _, _ = evaluate_model(model, tokenizer, test_dataset, device)
    
    # 4. Setup LoRa and fine-tune
    print(f"\n{'='*20} FINE-TUNING WITH LORA {'='*20}")
    lora_model = setup_lora_model(model)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fine-tune the model
    trainer = fine_tune_model(lora_model, tokenizer, train_dataset, val_dataset, output_dir)
    
    # 5. Evaluate model AFTER fine-tuning
    print(f"\n{'='*20} AFTER FINE-TUNING {'='*20}")
    post_accuracy, _, _ = evaluate_model(lora_model, tokenizer, test_dataset, device)
    
    # 6. Results summary and visualization
    print(f"\n{'='*20} RESULTS SUMMARY {'='*20}")
    print(f"📊 Accuracy before fine-tuning: {pre_accuracy:.4f}")
    print(f"📈 Accuracy after fine-tuning:  {post_accuracy:.4f}")
    print(f"🚀 Improvement: +{post_accuracy - pre_accuracy:.4f} ({(post_accuracy - pre_accuracy)*100:.2f}%)")
    
    # Save results
    results = {
        "model_name": model_name,
        "dataset": "IMDb",
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "pre_finetuning_accuracy": float(pre_accuracy),
        "post_finetuning_accuracy": float(post_accuracy),
        "improvement": float(post_accuracy - pre_accuracy),
        "improvement_percentage": float((post_accuracy - pre_accuracy) * 100)
    }
    
    with open('lora_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: lora_results.json")
    print(f"🤖 Fine-tuned model saved to: {output_dir}")
    
    # Plot comparison
    plot_comparison(pre_accuracy, post_accuracy)
    print(f"📊 Comparison plot saved to: lora_comparison.png")
    
    print(f"\n✅ LoRa fine-tuning pipeline completed successfully!")

if __name__ == "__main__":
    main()