# Genome-Accessibility-Project

A deep learning project for predicting chromatin accessibility and enhancer activity 
from genomic data, using convolutional and recurrent neural networks.
---


### Requirements

- Python 3.9+
- PyTorch with CUDA support
- pyBigWig (Linux/WSL only)
- numpy, pandas, scikit-learn, matplotlib

## Part 1 — Read Coverage Prediction

### Data

BigWig files (.bw) from two SRA studies containing 64 ChIP-seq samples. Files should 
be placed in `data/new_comb_data/`. The dataset is available from Zenodo.

- **Training/Validation**: Chromosomes 1-4
- **Test**: Chromosome 5
- **Bin size**: 200bp
- **Window size**: 200 bins

### Running

```bash
# Step 1: Process BigWig files and build sliding windows
cd part1
python3 DataProcess.py

# Step 2: Train CNN
python3 CNNReadCoverage.py

# Step 3: Train LSTM
python3 LSTMReadCoverage.py
```

### Models

**CNN** — 3-layer convolutional network with BatchNorm and dropout. Uses center bin 
of conv output for prediction. Trained with Poisson NLL loss.

**LSTM** — 2-layer LSTM network processing coverage windows as sequences. Also trained 
with Poisson NLL loss.

### Results

| Model | Pearson r (Chr5) |
|---|---|
| CNN | ~0.82 |
| LSTM | ~0.40 |

---

## Part 2 — Enhancer Activity Prediction

### Data

`arabidopsis_enhancers.csv` from Tan et al. (2023) containing 45,714 DNA sequences 
of ~2500bp. Place in `data/`. Labels are binary (1 = enhancer, 0 = non-enhancer) 
with 9.5% positive rate.

### Running

```bash
# Step 1: Process enhancer dataset
cd part2
python3 LoadData.py

# Step 2: Train Basset CNN from scratch
python3 CNNUntrained.py

# Step 3: Train transfer learning model
python3 TransferCNN.py
```

### Models

**CNN Untrained** — Basset-style architecture with 3 conv layers and 2 fully connected 
layers trained directly on one-hot encoded DNA sequence. Uses weighted BCE loss to 
handle class imbalance.

**Transfer CNN** — CNN pretrained on chromatin accessibility coverage data (Part 1), 
fine-tuned on enhancer data. Conv layers 2 and 3 are initialized with pretrained weights. 
A new input layer adapts 4-channel one-hot input to the pretrained network.

### Results

| Model | Test AUPRC |
|---|---|
| Random baseline | ~0.095 |
| Transfer CNN | 0.19 |
| CNN Untrained | 0.49 |

---

## Notebook

`FinalProject.ipynb` contains the full analysis including:
- Part 1 model training curves and Pearson r evaluation
- Part 2 model training curves and AUPRC evaluation
- Comparison between transfer learning and training from scratch
- Discussion and conclusions

To run the notebook, ensure all models have been trained and results saved, then open 
in Jupyter or VS Code.

---

## Key Findings

- CNNs outperform LSTMs for local genomic signal prediction
- Poisson loss is appropriate for count-based coverage regression
- Transfer learning from coverage signal to DNA sequence classification does not 
  provide benefit due to the large domain gap between input representations
- Training directly on task-specific data outperforms transfer learning when source 
  and target domains are incompatible

---

## Data Set

- Kelley et al. (2016). Basset: learning the regulatory code of the accessible genome 
  with deep convolutional neural networks. Genome Research.
- Kelley et al. (2018). Sequential regulatory activity prediction across chromosomes 
  with convolutional neural networks. Genome Research.
- Tan et al. (2023). Genome-wide enhancer identification by massively parallel reporter 
  assay in Arabidopsis. The Plant Journal.
