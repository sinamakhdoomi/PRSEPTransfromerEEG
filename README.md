# EEG Source‑Space Classification with Deep Learning

This repository provides an **end‑to‑end pipeline** for preprocessing, source localization, and classification of EEG data with three deep‑learning models:

* **EEGNet**
* **ResNet‑18**
* **PRSEPTrans‑EEG** (SE‑ResNet + Transformer + SimCLR pre‑training)

Although the sample commands target the *Reach‑and‑Grasp* dataset, the code can be adapted to other motor‑task EEG collections (e.g. **BCI2000** or **BCI Competition IV 2a**) by adjusting input shapes, trial structures, and subject indices.



## 🎯 Overview of PRSEPTrans‑EEG Framework

![Framework Overview](/framework.png)

**Figure 1** illustrates the end-to-end pipeline of the proposed PRSEPTrans‑EEG framework, a multimodal deep learning architecture designed to decode grasp-related EEG signals with high accuracy.

The process begins with an experimental protocol involving *palmar* and *lateral* grasp tasks, during which EEG is recorded across three phases: object fixation, reach-and-grasp execution, and rest.

Raw EEG signals are preprocessed using zero-phase Butterworth filtering and ICA-based artifact rejection to remove ocular and muscular noise. The clean signals are then projected to cortical source space using sLORETA, generating high-dimensional source-level features.

An optional **SimCLR-based contrastive pretraining** step is introduced to enhance feature robustness and class separability, especially under low SNR conditions.

The decoding network integrates:
- **Residual convolutional blocks** with **Squeeze-and-Excitation** modules for spatial feature enhancement,
- A **Transformer encoder** to model temporal dependencies in the EEG sequences.

Following attention-based encoding, adaptive pooling and fully connected layers perform the final motor task classification.

> This unified framework fuses spatial, spectral, and temporal dynamics to achieve robust classification of motor intentions from cortical activity.







---

## Repository structure

```text
├── scripts/                  # Python entry points
│   ├── preprocessing.py
│   ├── sLORETA_source_localization.py
│   ├── PRSEPTrans.py
│   ├── resnet18.py
│   └── eegnet_script.py
├── results/                  # Metrics, confusion matrices, logs
├── source_arrays/            # sLORETA outputs (.npy)
├── README.md
└── requirements.txt
```

---

## Supported datasets

### 1  Reach‑and‑Grasp Dataset \[[Schwarz *etal.* 2019, 2020](#references)]

* **Participants:** 45 right‑handed
* **Tasks:** palmar & lateral grasp
* **Systems:**

  * Gel‑based – 58 EEG + 6 EOG
  * Water‑based – 32 EEG + 6 EOG
  * Dry‑electrode – 11 EEG + 3 EOG
* **Preprocessing:** 0.3 Hz high‑pass, ICA (gel & water), epoching at movement onset, outlier rejection (amplitude/kurtosis/joint‑probability)

### 2  BCI2000 Motor Movement / Imagery \[[Schalk *etal.* 2004](#references)]

* **Participants:** 109
* **Channels:** 64 EEG
* **Rate:** 160 Hz
* **Tasks:** real & imagined hand/foot movements

### 3  BCI Competition IV Dataset 2a \[[Brunner *etal.* 2008](#references)]

* **Participants:** 9
* **Channels:** 22 EEG + 3 EOG
* **Rate:** 250 Hz
* **Tasks:** left hand, right hand, feet, tongue (motor imagery)

---

## Getting started

### 1  Preprocessing

```bash
python scripts/preprocessing.py \
    --data_dir /path/to/ReachGrasp \
    --subjects 15
```

### 2  sLORETA source localization

```bash
python scripts/sLORETA_source_localization.py \
    --sensor_file reach_grasp_x.npy \
    --label_file  reach_grasp_y.npy \
    --info_file   reach_grasp_raw.fif
```

### 3  PRSEPTrans‑EEG (SE‑ResNet + Transformer + SimCLR)

```bash
python scripts/PRSEPTrans.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --pretrain_band Theta \
    --folds 5 \
    --pretrain_epochs 50 \
    --class_epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --simclr_lr 1e-6 \
    --pretrain_patience 3 \
    --train_patience 10 \
    --n_channels 34 \
    --n_timepoints 257 \
    --seed 42
```

### 4  ResNet‑18 baseline

```bash
python scripts/resnet18.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --output_dir ./results/resnet18 \
    --epochs 100 \
    --batch_size 32
```

### 5  EEGNet baseline

```bash
python scripts/eegnet_script.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --n_channels 1884 \
    --n_samples 257 \
    --folds 5 \
    --patience 10 \
    --output_dir ./results/eegnet
```

---

## Results directory

Each model writes:

* **Confusion matrices** – `.csv` and `.png`
* **Metric summary** – `.txt` (accuracy, precision, recall, F1‑score, Cohen’s κ, runtime)
* **Aggregated folds** – `.npy` or `.json` (per‑band)

---

## Installation

```bash
pip install -r requirements.txt
```

<details>
<summary>Key Python packages</summary>

* numpy
* pandas
* scikit‑learn
* matplotlib
* seaborn
* torch & torchvision
* mne

</details>

---

## References

<a name="references"></a>

1. A. Schwarz, C. Escolano, L. Montesano, & G. R. Müller‑Putz (2020). *Analyzing and decoding natural reach‑and‑grasp actions using gel, water and dry EEG systems.* **Frontiers in Neuroscience**, 14:849.
2. A. Schwarz, J. Pereira, R. Kobler, & G. R. Müller‑Putz (2019). *Unimanual and bimanual reach‑and‑grasp actions can be decoded from human EEG.* **IEEE Transactions on Biomedical Engineering**, 67(6), 1684–1695.
3. G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, & J. R. Wolpaw (2004). *BCI2000: A general‑purpose brain–computer interface (BCI) system.* **IEEE Transactions on Biomedical Engineering**, 51(6), 1034–1043.
4. C. Brunner, R. Leeb, G. Müller‑Putz, A. Schlögl, & G. Pfurtscheller (2008). *BCI Competition 2008 – Graz data set A.* Graz University of Technology.
5. K. He, X. Zhang, S. Ren, & J. Sun (2016). *Deep residual learning for image recognition.* In **CVPR 2016** (pp. 770‑778).
6. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C.‑P. Hung, & B. J. Lance (2018). *EEGNet: A compact convolutional neural network for EEG‑based BCIs.* **Journal of Neural Engineering**, 15(5), 056013.

