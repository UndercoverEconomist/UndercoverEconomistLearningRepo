#!/bin/bash

echo "🚀 Setting up LoRa Fine-tuning Environment"
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "lora_env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv lora_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source lora_env/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default, change to GPU if needed)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "To run the LoRa fine-tuning example:"
echo "1. Activate the environment: source lora_env/bin/activate"
echo "2. Run the script: python lora_finetuning_example.py"
echo ""
echo "Note: The first run will download the dataset and model, which may take some time."