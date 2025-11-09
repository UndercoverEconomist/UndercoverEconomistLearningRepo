# LoRa Fine-tuning Examples

This folder contains comprehensive examples of fine-tuning language models using **LoRa (Low-Rank Adaptation)**, a parameter-efficient fine-tuning technique that dramatically reduces the number of trainable parameters while maintaining performance.

## 🎯 What You'll Find Here

### 1. **Complete Example** (`lora_finetuning_example.py`)
A comprehensive end-to-end example featuring:
- **Dataset**: IMDb movie reviews (sentiment classification)
- **Model**: DistilBERT-base-uncased
- **Before/After evaluation**: Complete model assessment
- **Visualization**: Performance comparison plots
- **Detailed logging**: Training progress and metrics

### 2. **Quick Start** (`quick_lora_example.py`)
A simplified version for quick testing:
- Smaller dataset (100 train, 50 test samples)
- 1 epoch training
- Fast execution for learning purposes

### 3. **Setup Scripts**
- `setup.sh`: Automated environment setup
- `requirements.txt`: All necessary dependencies

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh

# Activate the environment
source lora_env/bin/activate

# Run the quick example
python quick_lora_example.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv lora_env
source lora_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the complete example
python lora_finetuning_example.py
```

## 📊 What is LoRa?

**LoRa (Low-Rank Adaptation)** is a technique that:

1. **Freezes** the original pre-trained model weights
2. **Adds** small trainable low-rank matrices to attention layers
3. **Reduces** trainable parameters by up to 99%
4. **Maintains** model performance with much faster training

### Key Benefits:
- 💾 **Memory Efficient**: Significantly less GPU memory required
- ⚡ **Fast Training**: Much quicker than full fine-tuning
- 🎯 **Task-Specific**: Easy to create multiple task-specific adapters
- 🔄 **Reversible**: Can switch between different fine-tuned versions

## 📈 Expected Results

The examples demonstrate:

### Before Fine-tuning (DistilBERT on IMDb):
- **Accuracy**: ~50-60% (untrained on sentiment)
- **Behavior**: Random or poor sentiment classification

### After LoRa Fine-tuning:
- **Accuracy**: ~85-90% (well-trained on sentiment)
- **Improvement**: +25-40 percentage points
- **Training Time**: 10-30 minutes (vs hours for full fine-tuning)

## 🛠️ Customization

### Adjust LoRa Parameters:
```python
lora_config = LoraConfig(
    r=16,              # Rank (4-64): Higher = more parameters
    lora_alpha=32,     # Scaling (r to 2*r): Controls adaptation strength
    lora_dropout=0.1,  # Dropout (0.0-0.3): Prevents overfitting
    target_modules=["q_lin", "v_lin"]  # Which layers to adapt
)
```

### Try Different Models:
- `bert-base-uncased`
- `roberta-base`
- `xlnet-base-cased`
- Any HuggingFace transformer model

### Use Different Datasets:
- **Text Classification**: AG News, Yelp Reviews
- **Named Entity Recognition**: CoNLL-2003
- **Question Answering**: SQuAD
- **Custom datasets**: Your own data

## 📝 File Structure After Running

```
LoRa/
├── lora_finetuning_example.py    # Complete example
├── quick_lora_example.py         # Quick start example
├── requirements.txt              # Dependencies
├── setup.sh                     # Setup script
├── README.md                    # This file
├── lora_env/                    # Virtual environment (after setup)
├── lora_finetuned_model/        # Saved model (after training)
├── lora_results.json           # Training results
└── lora_comparison.png         # Performance visualization
```

## 🔍 Understanding the Output

### Model Evaluation:
```
📊 Accuracy: 0.8750
📋 Classification Report:
              precision    recall  f1-score   support
   Negative       0.88      0.86      0.87       500
   Positive       0.87      0.89      0.88       500
   accuracy                           0.88      1000
```

### LoRa Configuration:
```
trainable params: 592,900 || all params: 67,584,004 || trainable%: 0.8772
```
This shows LoRa only trains <1% of the original parameters!

## 🎓 Learning Objectives

After running these examples, you'll understand:

1. **How to prepare datasets** for fine-tuning
2. **Model evaluation** before and after training
3. **LoRa configuration** and parameter tuning
4. **Training process** with Hugging Face Transformers
5. **Performance measurement** and visualization
6. **Parameter efficiency** of LoRa vs full fine-tuning

## 🚨 Troubleshooting

### Common Issues:

**GPU Memory Error:**
- Reduce `per_device_train_batch_size`
- Use smaller model or dataset
- Enable gradient checkpointing

**Slow Training:**
- Ensure you're using GPU if available
- Reduce dataset size for testing
- Use the quick example first

**Import Errors:**
- Run the setup script
- Activate virtual environment
- Install requirements manually

**Poor Performance:**
- Increase training epochs
- Adjust learning rate
- Try different LoRa rank values

## 📚 Further Reading

- [LoRa Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

## 🤝 Contributing

Feel free to:
- Add examples with different models/datasets
- Improve documentation
- Fix bugs or add features
- Share your fine-tuning results!

---

**Happy Fine-tuning!** 🎉