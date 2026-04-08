# CerradoCoffeeLeaf: An In-the-Wild Dataset for Coffee Leaf Disease Recognition and Cross-Dataset Generalization

- Otávio Massanobu de Souza Oda  
- Pedro Ivo Vieira Good God - pedro.god@ufv.br  
- Leandro Henrique Furtado Pinto Silva - leandro.furtado@ufv.br  
- João Fernando Mari - [joaofmari.github.io](https://joaofmari.github.io)

Institute of Exact and Technological Sciences — Federal University of Viçosa, Rio Paranaíba, MG, Brazil  
Institute of Agricultural Sciences — Federal University of Viçosa, Rio Paranaíba, MG, Brazil  

---

## 🚀 Quick Start

```bash
git clone <https://github.com/joaofmari/cerradocoffeeleaf-cross-dataset.git>
cd <cerradocoffeeleaf-cross-dataset>

conda create -n env-coffee-py312 python=3.12
conda activate env-coffee-py312

pip install -r requirements.txt
```

To check available arguments:

```bash
python train_test.py --help
```

---

### 📂 Repository Structure

```text
.
├── train_test.ipynb     # Main training/testing notebook (IPYNB)
├── train_test.py        # Main training/testing script
├── run_batch.py         # Runs all paper experiments
├── data_aug_3.py        # Data augmentation strategies
├── models.py            # torchvision definitions
├── early_stopping.py    # Early stopping strategy definition
├── exp/                 # Saved experiment outputs
└── README.md
```

---



## 🌿 Proposed Dataset: CerradoCoffeeLeaf

The CerradoCoffeeLeaf dataset is composed of RGB images of Coffea arabica leaves collected under field conditions. The images were acquired using a handheld mobile device (Motorola Moto G6 Plus) at the Francisco de Melo Palheta Experimental Field, Federal University of Viçosa (UFV-CRP), Brazil. As a result, the dataset presents natural variability in illumination, background, occlusion, leaf orientation, and disease severity.

The dataset contains 1,476 images distributed across seven classes: Ascochyta (*Ascochyta coffeae*), Cescospora (*Cercospora coffeicola*), Miner (*Leucoptera coffeella*), Phoma (*Boeremia exigua pv. coffeae*), Bacterial Blight (*Pseudomonas syringae pv. phaseolicola*), Rust (*Hemileia vastatrix*), and healthy leaves.

The dataset is released with predefined stratified splits of 60%, 16, and 20% for training, validation, and test sets, respectively. It is provided at two image resolutions: the original resolution (3024 × 4032 pixels), including image metadata, and a resized version (1024 × 1365 pixels), which was used in the experiments reported in the associated manuscript. CSV files listing image filenames and their corresponding class labels are provided for each split.

The dataset is intended to support research on image classification for plant disease identification, particularly under field conditions and cross-dataset evaluation scenarios.

**Main characteristics**

- **Plant species:** Coffea arabica  
- **Acquisition environment:** Natural field conditions (in situ)  
- **Capture device:** Handheld mobile device (Motorola Moto G6 Plus)  
- **Image type:** RGB  
- **Number of classes:** 7  
- **Total images:** 1,476  
- **Annotation level:** Image-level classification 
- **Splits:** Predefined stratified training, validation, and test sets (64% / 16% / 20%)  

The dataset includes both diseased and healthy leaves with natural variations caused by illumination changes, occlusions, leaf orientation, background clutter, and disease severity.

---

### 📊 Classes

1. **Ascochyta** (*Ascochyta coffeae*)
2. **Bacterial Blight** (*Boeremia exigua pv. coffeae*)
3. **Cercospora** (*Cercospora coffeicola*)
4. **Miner** (*Leucoptera coffeella*)
5. **Phoma** (*Boeremia exigua pv. coffeae*)
6. **Rust** (*Hemileia vastatrix*)
7. **Healthy**

---

### 📥 Dataset Download

The **CerradoCoffeeLeaf** dataset is available at:

👉 Download from Zenodo: [LINK]

After downloading, extract the files and organize them as follows:

```text
datasets/
└── CerradoCoffeeLeaf/
    ├── train/
    ├── val/
    └── test/
```

Predefined CSV split files are already provided to ensure reproducibility. Each CSV file contains the image filename and its corresponding class label.

---

### 📌 Summary

CerradoCoffeeLeaf is an in-the-wild dataset for coffee leaf disease classification collected under field conditions in Brazil.

- Images: 1,476
- Classes: 7
- Task: Image classification
- Acquisition: Field conditions (Brazil)
- Device: Mobile (Moto G6 Plus)
- Splits: 64/16/20 (predefined)

---


## 🌍 External Datasets

We compare CerradoCoffeeLeaf with three widely used coffee leaf datasets:

- **JMuBEN**
- **BRACOL**
- **RoCoLe**

These datasets differ in acquisition conditions, geographic origin, and class composition, making them suitable for cross-dataset generalization analysis.

### JMuBEN

The JMuBEN dataset [1, 2] is one of the largest publicly available collections for coffee leaf disease classification, containing 58,550 RGB images of *Coffea arabica* leaves acquired in Kenya.

Although large, preliminary inspection suggests the presence of augmented samples, which may inflate within-dataset performance. Nevertheless, it remains a widely adopted benchmark.

### BRACOL

The Brazilian Arabica Coffee Leaf (BRACOL) dataset [3, 4] contains images captured under controlled indoor conditions with uniform backgrounds. Its clean visual conditions and balanced distribution make it a common baseline for classification studies.

### RoCoLe

The Robusta Coffee Leaf (RoCoLe) dataset [5, 6] contains 1,560 field-acquired images of *Coffea canephora*. Due to its natural acquisition conditions, it is valuable for evaluating robustness and cross-dataset generalization.

---

## 🔬 Cross-Dataset Experimental Setup

### Overlap vs. Non-Overlapping Classes

**Note:** Classes shown in parentheses indicate non-overlapping classes between the training and test datasets.

| TRAIN | Classes | CerradoCoffeeLeaf | JMuBEN | BRACOL | RoCoLe |
|------|--------|-------------------|--------|--------|--------|
| **CerradoCoffeeLeaf** | Ascochyta | Ascochyta | (Ascochyta) | (Ascochyta) | (Ascochyta) |
| | Bacterial Blight | Bacterial Blight | (Bacterial Blight) | (Bacterial Blight) | (Bacterial Blight) |
| | Cercospora  | Cercospora  | Cercospora  | Cercospora  | Cercospora  |
| | Healthy | Healthy | Healthy | Healthy | Healthy |
| | Miner | Miner | Miner | Miner | (Miner) |
| | Phoma | Phoma | Phoma | Phoma | (Phoma) |
| | Rust | Rust | Rust | Rust | Rust |
| **JMuBEN** | Cercospora  | Cercospora  | Cercospora  | Cercospora  | Cercospora  |
| | Healthy | Healthy | Healthy | Healthy | Healthy |
| | Miner | Miner | Miner | Miner | (Miner) |
| | Phoma | Phoma | Phoma | Phoma | (Phoma) |
| | Rust | Rust | Rust | Rust | Rust |
| **BRACOL** | Cercospora  | Cercospora  | Cercospora  | Cercospora  | Cercospora  |
| | Healthy | Healthy | Healthy | Healthy | Healthy |
| | Miner | Miner | Miner | Miner | (Miner) |
| | Phoma | Phoma | Phoma | Phoma | (Phoma) |
| | Rust | Rust | Rust | Rust | Rust |
| **RoCoLe** | Cercospora  | Cercospora  | Cercospora  | Cercospora  | Cercospora  |
| | Healthy | Healthy | Healthy | Healthy | Healthy |
| | Red Spider Mite | (Red Spider Mite) | (Red Spider Mite) | (Red Spider Mite) | Red Spider Mite |
| | Rust | Rust | Rust | Rust | Rust |

---

## 🧪 Experimental Protocol

### Part I — Training on CerradoCoffeeLeaf

Models are trained on CerradoCoffeeLeaf and evaluated on:

- its own test set  
- JMuBEN  
- BRACOL  
- RoCoLe  

This measures cross-dataset generalization.

### Part II — Training on External Datasets

Models are trained on each external dataset and evaluated on CerradoCoffeeLeaf to assess knowledge transfer toward the proposed dataset.

### Part III — Joint Training Across Datasets

Models are trained using combinations of datasets and evaluated on individual and combined test sets to analyze whether dataset integration improves generalization.

---

## ▶️ Running the Paper Experiments

Experiments can be executed interactively using the provided Jupyter notebooks or via command line using the Python scripts.

### Part I

```bash
python run_batch.py --exp 1
```

### Part II

```bash
python run_batch.py --exp 2
```

### Part III

```bash
python run_batch.py --exp 3
```

---

## 💻 Hardware and Software

Experiments were conducted using:

- NVIDIA GTX 1080 Ti
- Python 3.12  
- PyTorch 2.6.0 (CUDA 12.4)  
- The early stopping mechanism was adapted from [7].

---



## 📚 How to Cite

If you use this dataset or code, please cite:

```bibtex
@article{oda2026cerradocoffeeleaf,
  title={CerradoCoffeeLeaf: An In-the-Wild Dataset for Coffee Leaf Disease Recognition and Cross-Dataset Generalization},
  author={Oda, Otávio Massanobu de Souza and Good God, Pedro Ivo Vieira and Silva, Leandro Henrique Furtado Pinto and Mari, João Fernando},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
```bibtex
@dataset{oda2026cerradocoffeeleaf,
  author={Oda, Otávio Massanobu de Souza and Good God, Pedro Ivo Vieira and Silva, Leandro Henrique Furtado Pinto and Mari, João Fernando},
  title={CerradoCoffeeLeaf Dataset},
  year={2026},
  publisher={Zenodo},
  doi={DOI}
}
```
---

## 🙏 Acknowledgements

We would like to thank FAPEMIG and CNPq for their financial support.  

---

## 📖 References

1. Jepkoech, J., Mugo, D. M., Kenduiywo, B. K. & Too, E. C. Arabica coffee leaf images dataset for coffee leaf disease
detection and classification. Data brief 36, 107142 (2021).
2. Jepkoech, j., Kenduiywo, B., Mugo, D. & Chebet, E. Jmuben. Mendeley Data 1, DOI: 10.17632/t2r6rszp5c.1 (2021).
3. Manso, G. L., Knidel, H., Krohling, R. A. & Ventura, J. A. A smartphone application to detection and classification of
coffee leaf miner and coffee leaf rust (2019). 1904.00742.
4. Krohling, R. A., Esgario, J. & Ventura, J. A. Bracol–a brazilian arabica coffee leaf images dataset to identification and
quantification of coffee diseases and pests. Mendeley Data 1, DOI: 10.17632/yy2k5y8mxg.1 (2019).
5. Parraga-Alava, J., Cusme, K., Loor, A. & Santander, E. Rocole: A robusta coffee leaf images dataset for evaluation of
machine learning based methods in plant diseases recognition. Data brief 25, 104414 (2019).
6. Parraga-Alava, J., Cusme, K., Loor, A. & Santander, E. Rocole: A robusta coffee leaf images dataset. Mendeley Data 1,
DOI: 10.17632/c5yvn32dzg.2 (2019).
7. Sunde, B. M. early-stopping-pytorch: A pytorch utility package for early stopping. https://github.com/Bjarten/early-stopping-pytorch (2024).

## 📄 License
MIT License

##### *Last update: April 07, 2026*