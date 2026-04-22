
# Multimodal Learning for Imbalanced Multi-Label Chest X-Ray Classification with Radiology Reports
 
**Authors:** Laihi Bahar Eddine, Antonio Javier Gallego, Antonio Pertusa

---

## Overview

Chest X-ray classification is challenged by severe class imbalance, multi-label co-occurrence, and domain variability across scanners and patient populations. Radiology reports provide complementary semantic evidence that may help address these limitations, yet their role in strongly imbalanced multi-label settings remains underexplored. In this work, we investigate image-only, text-only, and multimodal learning on a curated subset of PadChest-GR, where each radiograph is paired with sentence-level clinical findings. We apply structured preprocessing to both modalities and benchmark four text backbones, three image backbones, and their multimodal combinations under a unified training configuration, including data augmentation and imbalance-mitigation strategies.

---

## Installation

> Add the installation steps here when the project is finished.

```bash
# installation commands will be added here
````

---

## Run

Main script:

```bash
python main.py
```

General usage:

```bash
python main.py [OPTIONS]
```

---

## Available Options

### Balance strategies

* `--wrs_mode`
  Enable Weighted Random Sampling.

* `--classWeighted`
  Enable class-weighted loss.

* `--dataAug`
  Enable data augmentation.

* `--oversampling`
  Enable oversampling.

### Image model

* `--imagemodel {0,1,2,3}`

Available values:

* `0` → ResNet50
* `1` → MAIRA-1
* `2` → MAIRA-2
* `3` → WithoutImage

### Text model

* `--textmodel {0,1,2,3,4}`

Available values:

* `0` → BertTokenizer
* `1` → BioBERT
* `2` → CXR-BERT
* `3` → CXR-BERT Specialized
* `4` → WithoutText

### Extra model options

* `--raddinohead`
  Enable pretrained head.

* `--freezeImage`
  Freeze image backbone.

* `--freezeText`
  Freeze text backbone.

### Label setup

* `--label_count {25,20,15,10,5}`
  Select the number of labels used in the experiment.

---

## Example Commands

### Multimodal model

```bash
python main.py --imagemodel 1 --textmodel 3
```

### Text-only model

```bash
python main.py --textmodel 3
```

### Image-only model

```bash
python main.py --imagemodel 1
```

### Multimodal with data augmentation, class-weighted loss and weighted random sampling

```bash
python main.py --imagemodel 1 --textmodel 3 --dataAug --classWeighted --wrs_mode
```

### Multimodal with oversampling

```bash
python main.py --imagemodel 1 --textmodel 3 --oversampling
```

### Frozen image backbone

```bash
python main.py --imagemodel 1 --textmodel 3 --freezeImage
```

### Frozen text backbone

```bash
python main.py --textmodel 3 --freezeText
```

### Reduced label setting

```bash
python main.py --imagemodel 1 --textmodel 3 --label_count 10
```

---

## Argument Summary

| Argument          | Description                     |
| ----------------- | ------------------------------- |
| `--wrs_mode`      | Enable Weighted Random Sampling |
| `--classWeighted` | Enable class-weighted loss      |
| `--dataAug`       | Enable data augmentation        |
| `--oversampling`  | Enable oversampling             |
| `--imagemodel`    | Select image model              |
| `--raddinohead`   | Enable pretrained head          |
| `--textmodel`     | Select text model               |
| `--freezeImage`   | Freeze image backbone           |
| `--freezeText`    | Freeze text backbone            |
| `--label_count`   | Select number of labels         |

---

## Notes

* You can use `--imagemodel 3` to disable the image branch.
* You can use `--textmodel 4` to disable the text branch.
* For multimodal experiments, select both an image model and a text model.
* Multiple training options can be combined in the same command.

---

## Citation

```bibtex
@article{to_be_added,
  title={Multimodal Learning for Imbalanced Multi-Label Chest X-Ray Classification with Radiology Reports},
  author={Bahar Eddine, Laihi and Gallego, Antonio Javier and Pertusa, Antonio},
  year={2026}
}
```

