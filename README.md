# WAVES: Weather-Aware Visual Estimation with Sets

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](notebook.ipynb)

This repository contains the official implementation for the paper:

**"Adaptive Uncertainty Quantification for Maritime Classification under Cloud Cover in Satellite Imagery"**

(Submitted to AI4SD Workshop at ECAI 2025, currently under review.)

## Overview

Reliable maritime vessel classification from satellite imagery is essential for tasks such as naval intelligence, search-and-rescue operations, and maritime security. However, the accuracy of traditional classifiers deteriorates significantly under meteorological conditions like clouds, haze, and fog.

This repository introduces **WAVES (Weather-Aware Visual Estimation with Sets)**, a novel conformal prediction-based framework that dynamically adjusts classification uncertainty based on image quality, particularly cloud coverage. WAVES produces adaptive prediction sets that become narrower for clear images and wider for cloud-covered images, providing statistically valid uncertainty quantification.

## Repository Structure

```
uncertainty-aware-ship-classification/
├── data/                   # Dataset splits and metadata
│   ├── preprocessed_dataset.*      # Full augmented dataset (7z compressed)
│   ├── train_dataset.*             # Training split
│   ├── val_dataset.*               # Validation split (for calibration & tuning)
│   ├── test_dataset.*              # Independent test split
│   └── preprocessed_metadata.csv   # Image metadata and cloud coverage scores
│
├── diagrams_paper/         # Paper figures and diagrams
│   ├── fig0.pdf            # Military vessel examples (FGSRCS dataset)
│   ├── fig1.pdf            # Synthetic cloud augmentation examples
│   ├── fig2.pdf            # Synthetic cloud coverage distribution
│   ├── fig3.pdf            # Class distributions for splits
│   ├── fig4.pdf            # WAVES vs. Global CP (prediction set sizes vs. cloud coverage)
│   └── fig5.pdf            # Calibration cumulative distribution example
│
├── models/                 # Trained models
│   ├── best_resnet50_epoch*.7z.*      # ResNet-50 fine-tuned model
│   ├── best_convnext_tiny_epoch*.7z.* # ConvNeXt-Tiny fine-tuned model
│   ├── best_densenet121_epoch*.7z.*   # DenseNet-121 fine-tuned model
│   └── quality_regressor.7z.*         # Cloud coverage regression model (ResNet-18)
│
├── notebook.ipynb          # Jupyter notebook for experiments
├── requirements.txt        # Python dependencies
├── results/                # Detailed results and visualizations
│   ├── bucket_*                    # WAVES (adaptive CP method) results
│   ├── global_conformal_*          # Global CP method results
│   ├── comparison_*                # WAVES vs. Global CP comparisons
│   ├── confusion_matrix_*          # Confusion matrices
│   ├── regression_*                # Cloud coverage regression evaluations
│   ├── relative_class_dist.svg     # Class distribution overview
│   └── *.svg, *.csv                # Other supporting results & visualizations
│
├── LICENSE                 # MIT License
└── .gitignore              # Git configuration
```

## Key Results

### Baseline Classifier Performance (with vs. without clouds)

| Model            | Accuracy (Original) | Accuracy (Cloud-Augmented) |
|------------------|---------------------|----------------------------|
| ResNet-50        | 76.72%              | 64.60%                     |
| ConvNeXt-Tiny    | 84.68%              | 73.20%                     |
| DenseNet-121     | 82.72%              | 71.10%                     |

### WAVES vs. Global CP (Cloud-Augmented Data)

| Model           | $\alpha$ | Baseline Acc. | Global CP Coverage | WAVES Coverage | Global CP Size | WAVES Size | WAVES Bins |
|-----------------|----------|---------------|--------------------|----------------|----------------|------------|------------|
| ResNet-50       | 0.01     | 64.6%         | **96.7%**          | 94.5%          | 8.00 ± 6.94    | **6.67 ± 6.79** | 3 |
| ResNet-50       | 0.05     | 64.6%         | 90.6%              | 90.6%          | 3.64 ± 3.21    | 3.64 ± 3.21 | 1 |
| ResNet-50       | 0.10     | 64.6%         | 84.9%              | 84.9%          | 2.41 ± 1.81    | 2.41 ± 1.81 | 1 |
| ConvNeXt-Tiny   | 0.01     | 73.2%         | **97.2%**          | 96.8%          | 5.61 ± 5.73    | **4.99 ± 5.43** | 3 |
| ConvNeXt-Tiny   | 0.05     | 73.2%         | **92.8%**          | 92.4%          | 2.95 ± 2.92    | **2.94 ± 3.29** | 3 |
| ConvNeXt-Tiny   | 0.10     | 73.2%         | **86.3%**          | 85.7%          | 1.76 ± 1.23    | **1.73 ± 1.29** | 2 |
| DenseNet-121    | 0.01     | 71.1%         | **97.1%**          | 96.7%          | 6.91 ± 5.34    | **6.49 ± 5.08** | 4 |
| DenseNet-121    | 0.05     | 71.1%         | 93.3%              | 93.3%          | 3.69 ± 2.99    | 3.69 ± 2.99 | 1 |
| DenseNet-121    | 0.10     | 71.1%         | **89.3%**          | 89.2%          | 2.47 ± 1.78    | **2.40 ± 2.04** | 3 |

*(Best performance highlighted in bold.)*

## Paper Figures (Diagrams)
| Figure                                   | Description                                         |
|------------------------------------------|-----------------------------------------------------|
| ![](diagrams_paper/fig0.pdf)             | Example military vessel classes (FGSRCS dataset).   |
| ![](diagrams_paper/fig1.pdf)             | Synthetic cloud augmentation examples.              |
| ![](diagrams_paper/fig2.pdf)             | Synthetic cloud coverage distribution.              |
| ![](diagrams_paper/fig3.pdf)             | Relative class distribution for dataset splits.     |
| ![](diagrams_paper/fig4.pdf)             | WAVES vs. Global CP (prediction size vs. coverage). |
| ![](diagrams_paper/fig5.pdf)             | Calibration cumulative distribution example.        |

## Installation and Usage

```bash
pip install -r requirements.txt
```

## Dataset Preparation

```bash
# Extract datasets from compressed .7z files into data/
7z x "data/*.7z.*" -odata/
```

## Running Experiments

Use the provided Jupyter notebook (`notebook.ipynb`) to:

- Fine-tune classification models (ResNet-50, DenseNet-121, ConvNeXt-Tiny).
- Train the cloud coverage regression model (ResNet-18).
- Evaluate global conformal prediction and WAVES methods.
- Generate visualizations and summarize results.

## Implemented Methods

- **Baseline Classifiers:** ResNet-50, DenseNet-121, ConvNeXt-Tiny.
- **Cloud Coverage Regressor:** ResNet-18 predicting cloud coverage scores.
- **Global Conformal Prediction:** Single global CP threshold for uncertainty quantification.
- **WAVES:** Adaptive conformal prediction with dynamic thresholds based on cloud coverage.

## Repository Contents

- **Data:** Training, validation, test splits, and metadata with cloud coverage.
- **Models:** Fine-tuned classification and regression models.
- **Results:** CSV and SVG files for detailed experiment results and visualizations.

## License

Distributed under the [MIT License](LICENSE).

## Acknowledgements

This research is part of the RIVA project, funded by dtec.bw – Digitalization and Technology Research Center of the Bundeswehr, supported by the European Union – NextGenerationEU.

## Contact

**Gianluca Manca** (Corresponding Author)  
Chair of Automation, Ruhr University Bochum  
Email: gianluca.manca@ruhr-uni-bochum.de

*(Paper currently under review at AI4DS Workshop at ECAI 2025; citation details will be provided upon acceptance.)*