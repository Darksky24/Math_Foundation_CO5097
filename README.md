# D2L Deep Learning Environment

A conda environment for deep learning with PyTorch 2.0, CUDA 11.7 support, and the Dive into Deep Learning (D2L) framework.

## 📋 Environment Specifications

- **Python**: 3.9.25
- **PyTorch**: 2.0.0 with CUDA 11.7 support
- **TorchVision**: 0.15.1
- **NumPy**: 1.23.5
- **Scikit-learn**: 1.2.2
- **JupyterLab**: 4.5.0
- **D2L**: 1.0.3

## 🚀 Quick Start

### Prerequisites

- Miniconda or Anaconda installed
- NVIDIA GPU with CUDA-compatible driver (optional, for GPU acceleration)
- Minimum NVIDIA driver version: 450.80.02

### Method 1: Using environment.yml (Recommended)

1. **Create `environment.yml` file:**

name: d2l
channels:

defaults

conda-forge
dependencies:

python=3.9.25

pip=25.3

numpy=1.23.5

matplotlib=3.7.2

pandas=2.0.3

scikit-learn=1.2.2

scipy=1.10.1

pip:

PyTorch with CUDA 11.7 support
torch==2.0.0

torchvision==0.15.1

Deep Learning
d2l==1.0.3

Jupyter & Lab
jupyter==1.0.0

jupyterlab==4.5.0

ipykernel==6.31.0

ipython==8.18.1

ipywidgets==8.1.8

notebook==7.5.0

Data Science
joblib==1.5.2

Utilities
requests==2.31.0

beautifulsoup4==4.14.3

matplotlib-inline==0.1.6


2. **Create and activate the environment:**
conda env create -f environment.yml
conda activate d2l

## 🔧 Troubleshooting

### NumPy Compatibility Issue

If you encounter `ModuleNotFoundError: No module named 'numpy._core'` when using `sklearn.datasets.fetch_rcv1()`:

**Solution:**
from sklearn.datasets import clear_data_home
clear_data_home()


This clears the cached dataset that was pickled with an incompatible NumPy version.

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA not detected | Verify driver with `nvidia-smi` command |
| Import errors | Check package versions with `pip list` |
| Jupyter kernel missing | Run `python -m ipykernel install --user --name d2l` |
| Package conflicts | Recreate environment from scratch |

## 📦 Complete Package List

### Core Dependencies
- Python 3.9.25
- NumPy 1.23.5
- SciPy 1.10.1
- Pandas 2.0.3
- Matplotlib 3.7.2
- Scikit-learn 1.2.2

### Deep Learning
- PyTorch 2.0.0
- TorchVision 0.15.1
- D2L 1.0.3

### Development Tools
- JupyterLab 4.5.0
- Jupyter Notebook 7.5.0
- IPython 8.18.1
- IPyKernel 6.31.0

### Utilities
- Joblib 1.5.2
- Requests 2.31.0
- BeautifulSoup4 4.14.3

## 🛠️ Environment Management

### Export environment


## 📝 Notes

- **CUDA Compatibility**: PyTorch 2.0.0 bundles CUDA 11.7 libraries, so it works even if your system has a different CUDA version
- **CPU Support**: All functionality works on CPU, but training will be slower
- **Memory Requirements**: GPU training requires at least 4GB VRAM for most D2L examples
- **Operating System**: Tested on Linux (Ubuntu 22.04), should work on Windows and macOS

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/2.0/)
- [D2L Book](https://d2l.ai/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)

## 📄 License

This environment configuration is provided as-is for educational purposes.

---

**Created for**: Deep Learning projects and D2L book exercises  
**Last Updated**: December 11, 2025
