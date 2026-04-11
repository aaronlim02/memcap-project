#!/bin/bash
# 1. Create venv
python3 -m venv venv
source venv/bin/activate

# 2. Install Torch with a specific CUDA index (Standard for most clusters)
# Change cu121 to match the destination's driver if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt

echo "Setup complete. Always run 'source venv/bin/activate' before training."
