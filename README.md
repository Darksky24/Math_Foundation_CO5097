# Nhóm 12 - môn Mathematics Foundations for Computer Science (CO5097)
1. Ngô Nhất Toàn - 2570515
2. Trần Hoàng Ân - 2570548
3. Trần Huy Phước - 2570482
4. Lê Thành Đạt - 2370497
5. Phạm Minh Quang - 2570302

# Math_final_report.pdf and working notebook is Final_notebook.ipynb


# D2L Deep Learning Environment

A conda environment for deep learning with PyTorch 2.0, CUDA 11.7 support, and the Dive into Deep Learning (D2L) framework.
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
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
**Create and activate the environment:**
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
