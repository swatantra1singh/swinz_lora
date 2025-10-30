#!/bin/bash

# Model setup script for deployment
echo "Setting up Swinz LoRA model..."

# Create model directory if it doesn't exist
mkdir -p swinz-3b-lora

# Check if model files exist
if [ ! -f "swinz-3b-lora/adapter_model.bin" ]; then
    echo "Warning: LoRA model files not found in swinz-3b-lora/"
    echo "Please ensure your trained model files are uploaded to this directory"
    echo "Required files:"
    echo "  - adapter_model.bin"
    echo "  - adapter_config.json"
    echo "  - tokenizer files (if custom)"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"