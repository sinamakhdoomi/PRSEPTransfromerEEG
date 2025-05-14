# EEG Source-Space Classification with Deep Learning

This repository provides a complete pipeline for preprocessing, source localization, and classification of EEG data using various deep learning models: **EEGNet**, **ResNet-18**, and **PRSEPTrans-EEG** (SE-ResNet + Transformer + SimCLR).

---

## 📦 Datasets

This study utilizes three publicly available EEG datasets:

### 1. **Reach and Grasp Dataset**

* **Participants**: 45 right-handed individuals
* **Tasks**: Palmar and lateral grasp
* **Recording Types**:

  * Gel-based (58 EEG + 6 EOG)
  * Water-based (32 EEG + 6 EOG)
  * Dry electrodes (11 EEG + 3 EOG)
* **Preprocessing**:

  * Zero-phase Butterworth filter at 0.3 Hz
  * ICA artifact removal (applied to gel and water-based systems)
  * Epoching around movement onset
  * Statistical outlier rejection (amplitude, kurtosis, joint probability)

### 2. **BCI2000 Motor Movement/Imagery Dataset**

* **Participants**: 109
* **Channels**: 64-channel EEG
* **Tasks**: Real and imagined movement of hands and feet
* **Sampling Rate**: 160 Hz

### 3. **BCI Competition IV Dataset 2a**

* **Participants**: 9
* **Tasks**: Left hand, right hand, foot, tongue (motor imagery)
* **Channels**: 22 EEG + 3 EOG
* **Sampling Rate**: 250 Hz

---

## ⚙️ How to Run

### 🧹 Preprocessing

```bash
python preprocessing.py \
    --data_dir /path/to/ReachGrasp \
    --subjects 15
```

### 🧠 sLORETA Source Localization

```bash
python sLORETA_source_localization.py \
    --sensor_file reach_grasp_x.npy \
    --label_file  reach_grasp_y.npy \
    --info_file   reach_grasp_raw.fif
```

### 🕸️ PRSEPTrans-EEG (SE-ResNet + Transformer + SimCLR)

```bash
python PRSEPTrans.py \
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

### 📈 ResNet-18

```bash
python resnet18.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --output_dir ./results_resnet18 \
    --epochs 100 \
    --batch_size 32
```

### 🧠 EEGNet

```bash
python eegnet_script.py \
    --data_root ./source_data \
    --bands Delta Theta Alpha Beta Gamma \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --n_channels 1884 \
    --n_samples 257 \
    --folds 5 \
    --patience 10 \
    --output_dir ./results_eegnet
```

---

## 📚 Dependencies

Make sure to install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision mne
```

If you're working with `.fif` files or using MNE features:

```bash
pip install mne
```

---

## 📁 Outputs

Each model will generate:

* `.txt` files with accuracy, precision, recall, F1-score, kappa, and runtime
* `.csv` confusion matrices (row-normalized)
* `.png` heatmaps of the confusion matrix
* `.npy` or `.json` summary of results for all bands

---

## 📝 Citation References

Please cite the following if you use this repository:

```
[1] Andreas Schwarz, Carlos Escolano, Luis Montesano, and Gernot R. Müller-Putz. 
     Analyzing and decoding natural reach-and-grasp actions using gel, water and dry EEG systems.
     Frontiers in Neuroscience, 14:849, 2020.

[2] Andreas Schwarz, Joana Pereira, Reinmar Kobler, and Gernot R. Müller-Putz. 
     Unimanual and bimanual reach-and-grasp actions can be decoded from human EEG.
     IEEE Transactions on Biomedical Engineering, 67(6):1684–1695, 2019.

[3] Gerwin Schalk, Dennis J. McFarland, Thilo Hinterberger, Niels Birbaumer, and Jonathan R. Wolpaw.
     BCI2000: A general-purpose brain–computer interface (BCI) system.
     IEEE Transactions on Biomedical Engineering, 51(6):1034–1043, 2004.

[4] Clemens Brunner, Robert Leeb, Gernot Müller-Putz, Alois Schlögl, and Gert Pfurtscheller.
     BCI Competition 2008 – Graz data set A.
     Graz University of Technology, Institute for Knowledge Discovery, 2008.

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
     Deep residual learning for image recognition.
     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[6] Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, and Brent J. Lance.
     EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces.
     Journal of Neural Engineering, 15(5):056013, 2018.
