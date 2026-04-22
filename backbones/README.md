# RAD-DINO Weights

This folder stores the pretrained RAD-DINO weight files required by the project.

The weights come from the Microsoft RAD-DINO repository on Hugging Face:

`https://huggingface.co/microsoft/rad-dino/tree/main`

## Required files

Place the following files in this folder:

- `backbone_compatible.safetensors`
- `dino_head.safetensors`

## Purpose

These files contain the pretrained backbone and head weights used by the image model components in this project.

## Notes

- Make sure the filenames match exactly.
- Do not rename the files unless you also update the code that loads them.
- If the weights are missing, the model may fail during initialization or inference.