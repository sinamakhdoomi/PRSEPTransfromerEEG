EEG Source-Space Classification with Deep Learning
This repository provides a complete pipeline for preprocessing, source localization, and classification of EEG data using multiple deep learning models:
EEGNet, ResNet-18, and PRSEPTrans-EEG (SE-ResNet + Transformer + SimCLR).

The scripts serve as reference implementations for classifying motor execution and motor imagery tasks from source-reconstructed EEG data. Although the sample code targets the Reach and Grasp dataset, it can be adapted to other datasets (e.g., BCI2000 and BCI Competition IV Dataset 2a) by modifying input shapes, trial structures, and subject counts.

📂 Project Structure
bash
Copy
Edit
├── scripts/             # All Python scripts
│   ├── preprocessing.py
│   ├── sLORETA_source_localization.py
│   ├── PRSEPTrans.py
│   ├── resnet18.py
│   └── eegnet_script.py
├── results/             # Outputs per model (txt, csv, png, npy)
├── source_arrays/       # sLORETA data (from preprocessing/localization)
├── README.md
└── requirements.txt
📦 Datasets
This pipeline supports three publicly available motor task EEG datasets:

1. Reach and Grasp Dataset
Subjects: 45 right-handed participants

Tasks: Palmar and lateral grasp

EEG Systems:

Gel-based: 58 EEG + 6 EOG

Water-based: 32 EEG + 6 EOG

Dry electrodes: 11 EEG + 3 EOG

Preprocessing:

Zero-phase Butterworth filter (≥0.3 Hz)

ICA for artifact rejection (gel & water systems)

Epoching around movement onset

Outlier rejection based on amplitude, kurtosis, and joint probability

2. BCI2000 Motor Movement/Imagery Dataset
Subjects: 109

Channels: 64 EEG

Sampling Rate: 160 Hz

Tasks: Real and imagined movements (hands/feet)

3. BCI Competition IV Dataset 2a
Subjects: 9

Channels: 22 EEG + 3 EOG

Sampling Rate: 250 Hz

Tasks: Left hand, right hand, feet, and tongue (motor imagery)

⚙️ How to Run
Modify parameters according to your dataset's input dimensions and subject count.

1. Preprocessing
bash
Copy
Edit
python scripts/preprocessing.py \
    --data_dir /path/to/ReachGrasp \
    --subjects 15
2. Source Localization (sLORETA)
bash
Copy
Edit
python scripts/sLORETA_source_localization.py \
    --sensor_file reach_grasp_x.npy \
    --label_file  reach_grasp_y.npy \
    --info_file   reach_grasp_raw.fif
3. PRSEPTrans-EEG (SE-ResNet + Transformer + SimCLR)
bash
Copy
Edit
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
4. ResNet-18
bash
Copy
Edit
python scripts/resnet18.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --output_dir ./results/resnet18 \
    --epochs 100 \
    --batch_size 32
5. EEGNet
bash
Copy
Edit
python scripts/eegnet_script.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --n_channels 1884 \
    --n_samples 257 \
    --folds 5 \
    --patience 10 \
    --output_dir ./results/eegnet
📁 Results
Each model generates:

Confusion matrices (.csv and .png)

Evaluation metrics (.txt): accuracy, precision, recall, F1, kappa, runtime

Aggregate results (.npy or .json): stored in the results folder

📚 Requirements
Install required libraries with:

bash
Copy
Edit
pip install -r requirements.txt
Required Libraries
numpy

pandas

scikit-learn

matplotlib

seaborn

torch

torchvision

mne (for preprocessing .fif files and EEG source localization)

🧠 Citations
If you use this work, please cite the following:

less
Copy
Edit
[1] Schwarz, A., Escolano, C., Montesano, L., & Müller-Putz, G.R. (2020). 
    Analyzing and decoding natural reach-and-grasp actions using gel, water and dry EEG systems. 
    Frontiers in Neuroscience, 14:849.

[2] Schwarz, A., Pereira, J., Kobler, R., & Müller-Putz, G.R. (2019). 
    Unimanual and bimanual reach-and-grasp actions can be decoded from human EEG. 
    IEEE Transactions on Biomedical Engineering, 67(6):1684–1695.

[3] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., & Wolpaw, J.R. (2004). 
    BCI2000: A general-purpose brain–computer interface (BCI) system. 
    IEEE Transactions on Biomedical Engineering, 51(6):1034–1043.

[4] Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). 
    BCI Competition 2008 – Graz data set A. 
    Graz University of Technology, Institute for Knowledge Discovery.

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). 
    Deep residual learning for image recognition. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778.

[6] Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., & Lance, B.J. (2018). 
    EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. 
    Journal of Neural Engineering, 15(5):056013.
Let me know if you'd like a badge, license section, or GitHub Actions workflow added!
