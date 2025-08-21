# MLP MNIST (from scratch)

Implementation of a simple **Multilayer Perceptron (MLP) from scratch** (only using TensorFlow to download the MNIST dataset), with forward pass, backpropagation, and training in mini-batches.

## Contents
- `MLP.py`: MLP implemented with NumPy (ReLU + Softmax + Cross-Entropy), training over epochs/batches, and evaluation on MNIST test set.
- `requirements.txt`: minimal dependencies to run the project.
- `.gitignore`: ignores temporary and virtual environment files.
- `LICENSE`: MIT license.

## Requirements
- Python 3.10+
- Packages listed in `requirements.txt`

## How to run
```bash
# 1) (Optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run the script
python MLP.py
```

The script downloads MNIST via `tf.keras.datasets.mnist`, trains the network, and prints test set accuracy.

## Project structure
```
mlp-mnist-from-scratch/
├── LICENSE
├── MLP.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Notes
- By default, TensorFlow installed is the CPU version. If you have a GPU and proper drivers/CUDA, you may install the GPU-compatible package for better performance.
- Feel free to open issues and PRs with improvements (e.g., channel-wise normalization, different initializations, LR scheduler, unit tests, etc.).
