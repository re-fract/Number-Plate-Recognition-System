#!/bin/bash

# Install PyTorch CPU versions from specific index
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other requirements
pip install -r requirements_railway.txt

# Start the application
python app.py
