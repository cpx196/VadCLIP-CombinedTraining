# VadCLIP-CombinedTraining

This project extends VadCLIP with the ability to **train on multiple datasets simultaneously** (e.g., UCF-Crime and XD-Violence).  
The **core backbone architecture remains unchanged**; improvements focus on the **training pipeline and dataset handling**.

## Key Features

- **Cross-dataset joint training**
- **Unified label space** across datasets
- **Combined dataset loader**
- **Robust label mapping** for inconsistent annotations
- **Independent evaluation + average AP**

## Main Additions

```
src/combined_train.py       # joint training pipeline
src/combined_option.py      # training configuration
src/utils/dataset.py        # CombinedDataset
src/utils/tools_com.py      # robust label mapping
src/demo.py                 # inference demo
```

## Training

```bash
python src/combined_train.py
```

## Inference

```bash
python src/demo.py
```

## Notes

- Backbone model from the official VadCLIP implementation.
- This repository mainly introduces **cross-dataset training and engineering extensions**.

## Acknowledgement

Based on the original VadCLIP framework.