# DermNet Triage Core (Edge AI)

An offline-first, on-device clinical decision support engine for dermatological triage. Powered by **EfficientNetV2-B3**, optimized with FP16 quantization for sub-40ms latency on mobile edge devices (Android/iOS via Flutter).

---

## 📊 Model Specifications & Benchmarks

| Metric | Target Specification |
| :--- | :--- |
| **Base Architecture** | EfficientNetV2-B3 (Transfer Learning from ImageNet) |
| **Input Tensor Contract** | `[1, 300, 300, 3]` (Float32, Native Scale `[0.0, 255.0]`) |
| **Output Tensor** | `[1, 24]` (Softmax Probabilities) |
| **Quantization** | Float16 Post-Training Optimization |
| **Model Size** | ~27.28 MB |
| **Validation Top-3 Accuracy** | **84.42%** |
| **Training Top-3 Accuracy** | **94.28%** |
| **Validation Top-1 Accuracy** | **64.93%** |
| **Execution Hardware Ops** | 3.477 G MACs (XNNPACK / Metal / GPU Delegate ready) |

---

## 🗂️ Dataset Source & Setup

This core is trained and evaluated on the clinical **DermNet** benchmark comprising 24 fine-grained dermatological classes across 21,559 images.

* **Dataset Source**: [DermNet on Kaggle by Shubham Goel](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)

### Directory Structure
Before executing training, organize the dataset directory as follows:
```text
dataset/
├── train/
│   ├── Acne and Rosacea Photos/
│   ├── Eczema Photos/
│   ├── Melanoma Skin Cancer Nevi and Moles/
│   └── ... (24 class subdirectories)
└── test/
    ├── Acne and Rosacea Photos/
    ├── Eczema Photos/
    └── ... (24 class subdirectories)
