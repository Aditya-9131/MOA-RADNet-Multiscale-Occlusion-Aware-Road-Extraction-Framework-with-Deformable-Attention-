# MOA-RADNet: A Multiscale Occlusion-Aware Framework with Deformable Attention for Road Extraction

## Overview
**MOA-RADNet** (Multiscale Occlusion-Aware RADANet with Deformable Attention) is a deep learning framework designed for accurate road extraction from high-resolution remote sensing imagery. The model improves segmentation accuracy, road continuity, and robustness in complex satellite environments by integrating multiscale feature interaction, road-aware modeling, deformable attention, and occlusion-aware decoding.

The framework is inspired by state-of-the-art models such as RADANet, OARENet, and MSGCNet, and combines their strengths into a unified encoder-decoder architecture.

---

## 🏗️ Architecture
The proposed MOA-RADNet architecture consists of the following components:

![MOA-RADNet Architecture](assets/architecture.png)

1. **ResNet50 Encoder**: Extracts hierarchical semantic features using a pre-trained ResNet50 backbone.
2. **Multiscale Interaction Module (MSI)**: Captures and fuses features from different resolutions to detect both thin and wide roads.
3. **Road-Aware Module (RAM)**: Enhances structural information using directional and strip-based convolutions.
4. **Deformable Attention Module (DAM)**: Models long-range dependencies and focuses on important road regions.
5. **Occlusion-Aware Decoder (OADecoder)**: Reconstructs missing segments caused by trees, buildings, and shadows.
6. **Segmentation Decoder**: Generates the final road segmentation map.

---

## 🌟 Key Features
*   Accurate extraction of thin and elongated road structures.
*   Improved road continuity under severe occlusions.
*   Multiscale feature interaction for varying road widths.
*   Deformable attention for adaptive long-range dependency modeling.
*   Occlusion-aware decoding for reconstructing missing road segments.
*   Robust performance on high-resolution benchmark datasets.

---

## 🚀 Training Configuration
| Parameter | Value |
| :--- | :--- |
| Framework | PyTorch |
| Backbone | ResNet50 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 2 |
| Epochs | 50 |
| Loss Function | BCE + Dice + Connectivity + Edge-Aware |

---

## 📊 Evaluation Results
### Quantitative Results
| Dataset | Precision | Recall | F1-Score | IoU |
| :--- | :---: | :---: | :---: | :---: |
| DeepGlobe | 78.42% | 81.15% | 79.76% | 72.49% |
| JHWV | 79.24% | 78.80% | 79.02% | 70.11% |

### Comparison with Existing Models
| Model | F1-Score | IoU |
| :--- | :---: | :---: |
| RADANet | 0.6324 | 0.7509 |
| OARENet | 0.7200 | 0.6500 |
| MSGCNet | 0.7300 | 0.6900 |
| **MOA-RADNet (Ours)** | **0.7902** | **0.82456** |

**Note:** All models were evaluated and compared on a **CPU-based environment** for research consistency.

---

## 🖼️ Qualitative Results
![Segmentation Comparison](assets/results.png)
*Figure: Qualitative comparison of MOA-RADNet vs Ground Truth.*

---

## 🛠️ Technologies Used
*   **Python** & **PyTorch**
*   **Computer Vision** (OpenCV, Scikit-Image)
*   **Deep Learning** (Semantic Segmentation)
*   **Remote Sensing** (Satellite Imagery Processing)

---

## 👤 Author
**Aditya Kumar**  
B.Tech – Computer Science & Engineering  
Indian Institute of Information Technology Guwahati  

---

## 📜 Acknowledgement
This project is inspired by the research contributions of RADANet, OARENet, and MSGCNet, and aims to improve road extraction performance through a unified multiscale occlusion-aware framework.
