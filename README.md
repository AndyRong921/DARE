# DARE: Domain-Aware Representation Enhancement for Data-Efficient Histopathological Tissue Classification


This is the official implementation of the paper: "DARE: Domain-Aware Representation Enhancement for Data-Efficient Histopathological Tissue Classification".

We propose DARE, a dual-space framework for robust tissue classification. It integrates two data augmentation strategies within an advanced lightweight deep broad learning system for targeted domain adaptation.

## ✨ Highlights

DARE† (Feature-space): A pseudo-domain partition strategy that enriches feature-space variability.

DARE* (Image-space): An image-level scheme to enhance robustness via input-space diversity.

Training-free: Ensures stable optimization in small-sample regimes without iterative backpropagation for the classifier.

Data-Efficient: Achieves significant gains on multi-center data using only 1% labeled samples.

## 🏗 Project Structure
```
The repository is organized as follows:

├── dataset/             # (Placeholder) Create this folder for your data
├── save/                # Output directory for weights and PCA models
├── tool/                # Core logic & utilities
│   ├── PDBL.py          # PDBL classifier implementation
│   ├── dataset.py       # Data loading utilities
│   ├── utils.py         # Helper functions
│   └── ...              # Other backbone definitions (resnet, shufflenet, etc.)
├── main.py              # Main script for DARE ablation study
├── pdbl_swin_tiny_model.py # Swin-Tiny triple-branch model definition
├── export_weights.py    # Weight compression and export script
├── requirements.txt     # Dependencies
└── README.md
```

## 📊 Datasets

We evaluate DARE using subsets of the Kather Multiclass Dataset.

Source Domain (KME): Kather Multiclass External subset.

Target Domain (Kather001): Kather001 subset.

Please download the datasets from Zenodo or the official Kather Laboratory website. Organize them into the dataset/ folder. Note that the 'Background' (BACK) class is removed for consistent 8-class classification.

## 🛠 Installation

Clone the repository:
```
git clone [https://github.com/YourUsername/DARE.git](https://github.com/YourUsername/DARE.git)
cd DARE
```

Install dependencies:
```
pip install -r requirements.txt
```

## 💻 Usage

1. Running DARE Ablation Study

The main.py script executes the ablation study comparing Baseline, DARE*, DARE†, and Voting strategies as described in the paper:

python main.py --backbone swin --source_dir dataset/KME --target_dir dataset/Kather --n_clients 4


2. Exporting and Compressing Weights

To perform PCA-based compression on trained weights:

python export_weights.py


## 📖 Methodology

Multi-Scale Backbone: Uses a pyramidal topology to capture hierarchical morphology across different image resolutions (224, 160, 112).

DARE Mechanism:

DARE*: Applies stochastic normalization using a pool of statistics sampled from the source domain to simulate domain shifts.

DARE†: Partitions the source data into pseudo-domains via K-means clustering to perform cross-domain interpolation.

Inference-time Voting: Aggregates predictions across multiple stochastic realizations ($K=8$) to ensure decision stability.

## 🙏 Acknowledgment

The PDBL classifier implementation in this project is based on the original work by Lin et al.: PDBL: Improving Histopathological Tissue Classification with Plug-and-Play Pyramidal Deep-Broad Learning.

## 📝 Citation

If you find this research useful, please cite our paper:
```
@article{rong2026dare,
  title={DARE: Domain-Aware Representation Enhancement for Data-Efficient Histopathological Tissue Classification},
  author={Rong, Zhijin and Zeng, Xueying and Zhang, Jingliang and Zhang, Qing},
  journal={Biomedical Signal Processing and Control},
  year={2026}
}
```
