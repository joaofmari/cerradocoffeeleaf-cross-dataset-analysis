# CerradoCoffeeLeaf Dataset

CerradoCoffeeLeaf is an in-the-wild dataset for coffee leaf disease classification collected under field conditions in Brazil.

The CerradoCoffeeLeaf dataset is composed of RGB images of Coffea arabica leaves collected under field conditions. The images were acquired using a handheld mobile device (Motorola Moto G6 Plus) at the Francisco de Melo Palheta Experimental Field, Federal University of Viçosa (UFV-CRP), Brazil. As a result, the dataset presents natural variability in illumination, background, occlusion, leaf orientation, and disease severity.

The dataset contains 1,476 images distributed across seven classes: Ascochyta (*Ascochyta coffeae*), Cescospora (*Cercospora coffeicola*), Miner (*Leucoptera coffeella*), Phoma (*Boeremia exigua pv. coffeae*), Bacterial Blight (*Pseudomonas syringae pv. phaseolicola*), Rust (*Hemileia vastatrix*), and healthy leaves.

The dataset is released with predefined stratified splits of 60%, 16%, and 20% for training, validation, and test sets, respectively. It is provided at two image resolutions: the original resolution (3024 × 4032 pixels), including image metadata, and a resized version (1024 × 1365 pixels), which was used in the experiments reported in the associated manuscript. CSV files listing image filenames and their corresponding class labels are provided for each split.

The dataset is intended to support research on image classification for plant disease identification, particularly under field conditions and cross-dataset evaluation scenarios.

The dataset is associated with the manuscript:

"CerradoCoffeeLeaf: An In-the-Wild Dataset for Coffee Leaf Disease Classification and Cross-Dataset Generalization"

---

## 1. Dataset Overview

- **Plant species:** Coffea arabica  
- **Acquisition environment:** Natural field conditions (in situ)  
- **Capture device:** Handheld mobile device (Motorola Moto G6 Plus)  
- **Image type:** RGB  
- **Number of classes:** 7  
- **Total images:** 1,476  
- **Annotation level:** Image-level classification 
- **Splits:** Predefined stratified training, validation, and test sets (60% / 20% / 20%)  

The dataset includes both diseased and healthy leaves with natural variations caused by illumination changes, occlusions, leaf orientation, background clutter, and disease severity.

---

## 2. Classes

The dataset contains the following classes:

1. **Ascochyta** (*Ascochyta coffeae*)
2. **Bacterial Blight** (*Boeremia exigua pv. coffeae*)
3. **Cercospora** (*Cercospora coffeicola*)
4. **Miner** (*Leucoptera coffeella*)
5. **Phoma** (*Boeremia exigua pv. coffeae*)
6. **Rust** (*Hemileia vastatrix*)
7. **Healthy**

---

## 3. Dataset Structure

The dataset is provided as **two ZIP files**, corresponding to two image resolutions.

### 3.1 Original Resolution (3024 × 4032)

- `CerradoCoffeeLeaf.zip`
- Original images with full spatial resolution
- Preserved image metadata (EXIF)
- Intended for future use, re-annotation, or alternative preprocessing pipelines

### 3.2 Resized Resolution (1024 × 1365)

- `CerradoCoffeeLeaf_1024.zip`
- Images resized while preserving aspect ratio
- Resolution used in all experiments reported in the associated manuscript

Each ZIP file follows the same internal organization:

```
CerradoCoffeeLeaf_[resolution]/
├── train/
│ ├── Ascochyta/
│ ├── AureolateLeafSpot/
│ ├── Cercospora/
│ ├── Miner/
│ ├── Phoma/
│ ├── Rust/
│ └── Healthy/
├── val/
│ └── (same class structure)
├── test/
│ └── (same class structure)
├── train.csv
├── val.csv
└── test.csv
```


---

## 4. CSV Annotation Files

For each split (`train`, `val`, `test`), a CSV file is provided with the following format:

| filename | class_idx | class |
|---------|-----------|-------|
| IMG_001.png | 2 | Cercospora |
| IMG_002.png | 3 | Healthy |

- **filename:** image file name
- **class_idx:** numerical class index
- **class:** corresponding class label

These CSV files allow easy loading of the dataset without relying on directory structure.
The CSV files can be used independently of the directory structure, enabling flexible data loading pipelines.

---

## 5. Data Splits

The dataset was split in two stages using stratified sampling. First, 80% of the data was assigned to training-validation and 20% to test. The training-validation portion was then split into 80% training and 20% validation.

The final dataset is in proportions of 64% training, 16% validation, and 20% test.

- **Training:** 64% (944 images)
- **Validation:** 16% (236 images)
- **Test:** 20% (296 images)

Class balance is preserved across splits.  
These splits were used in all experiments reported in the associated manuscript and should be maintained for reproducibility.

---

## 6. Intended Use

The dataset is intended for:

- Image classification of coffee leaf diseases
- Cross-dataset and domain generalization studies
- Benchmarking deep learning models under field conditions
- Research in digital agriculture and plant pathology

---

## 7. Reproducibility

To reproduce the experiments reported in the manuscript:

- Use the **1024 × 1365 resolution version**
- Use the predefined `train`, `val`, and `test` splits
- Follow the preprocessing and training protocols described in the paper

---

## 8. Citation

If you use this dataset, please cite the dataset and the associated paper:

```bibtex
@dataset{oda2026cerradocoffeeleaf,
  author={Oda, Otávio Massanobu de Souza and Good God, Pedro Ivo Vieira and Silva, Leandro Henrique Furtado Pinto and Mari, João Fernando},
  title={CerradoCoffeeLeaf Dataset},
  year={2026},
  publisher={Zenodo},
  doi={DOI}
}
```

```bibtex
@article{oda2026cerradocoffeeleaf,
  title={CerradoCoffeeLeaf: An In-the-Wild Dataset for Coffee Leaf Disease Classification and Cross-Dataset Generalization},
  author={Oda, Otávio Massanobu de Souza and Good God, Pedro Ivo Vieira and Silva, Leandro Henrique Furtado Pinto and Mari, João Fernando},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  doi={DOI},
  year={2026}
}
```

---

## 9. License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to share and adapt the dataset, provided appropriate credit is given.

---

## 10. Authors

- **Otávio Massanobu de Souza Oda** 
- **Pedro Ivo Good God** - pedro.god@ufv.br
- **Leandro Henrique Furtado Pinto Silva** - leandro.furtado@ufv.br
- **João Fernando Mari** - joaof.mari@ufv.br
- (Federal University of Viçosa – Campus Rio Paranaíba)