# Saved models

This directory contains trained models created by running `main.py` with
`--saveModel`. Model saving is disabled by default.

```bash
python -u main.py --textmodel 3 --textCleaning 2 --saveModel
```

Run the command from this project directory, or use the complete `PYTHONPATH`
command documented in the main project README.

## File naming

Saved filenames encode the experiment configuration and label count:

```text
<configuration>_Labels25.pth
<configuration>_Labels25.joblib
```

With `--crossValidation`, each independently trained fold is saved separately:

```text
<configuration>_Labels25_Fold1.pth
...
<configuration>_Labels25_Fold5.pth
```

The same convention applies to `.joblib` files. Running an identical
configuration again writes to the same path and replaces that checkpoint.

## PyTorch checkpoints (`.pth`)

Image, neural text, and multimodal models are stored as a dictionary containing:

| Key | Description |
| --- | --- |
| `model_state_dict` | Trained parameters of the best validation-F1 model |
| `classes` | Ordered list of output labels |
| `configuration` | Models, training options, label count, cleaning stage, cross-validation flag, and seed |
| `fold` | One-based fold number for cross-validation, otherwise `None` |

Inspect a checkpoint without constructing its model:

```python
import torch

checkpoint = torch.load("save_models/<model>.pth", map_location="cpu")
print(checkpoint["configuration"])
print(checkpoint["classes"])
```

To perform inference, first recreate the same architecture from
`checkpoint["configuration"]`, then load its parameters:

```python
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The checkpoint intentionally does not contain the optimizer, scheduler, epoch,
or training history, so it supports evaluation/inference but is not a complete
resume-training checkpoint. Loading also requires the same project model code
and compatible dependency versions.

## TF-IDF + Linear SVM models (`.joblib`)

The TF-IDF baseline (`--textmodel 5`) is stored as a dictionary containing the
fitted vectorizer/classifier wrapper plus its labels and configuration:

```python
import joblib

checkpoint = joblib.load("save_models/<model>.joblib")
model = checkpoint["model"]
classes = checkpoint["classes"]

predictions = model.predict(["example radiology report"])
scores = model.decision_function(["example radiology report"])
```

Only load `.pth` or `.joblib` files from trusted sources. Both formats can
execute code during deserialization.

## Relationship to evaluation outputs

This directory stores model artifacts only. Metrics and plots are written to
`../graphs/`:

- normal final-cleaning results go to `experiment_results.csv`;
- text-cleaning comparisons go to `text_leakage_comparison.csv`;
- cross-validation produces separate fold and per-class CSV files;
- neural-network training produces loss-curve PNG files.

Saving a model does not change which evaluation tables are produced.
