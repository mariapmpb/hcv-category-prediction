# HCV Case Classification with Machine Learning Models

Academic project for the Machine Learning course 

**Authors:** Maria Barbosa, Marta Neves, Melissa Leal


## Overview

This project applies supervised machine learning models  **Logistic Regression** and **Random Forest** to the [HCV Data dataset](https://doi.org/10.24432/C5D612) (UCI Machine Learning Repository), with the goal of classifying individuals into clinical categories (blood donor, suspect blood donor, hepatitis, fibrosis, cirrhosis) based on demographic data (age, sex) and blood biochemical markers.

The pipeline covers three stages:
1. Exploratory Data Analysis (EDA)
2. Data pre-processing (missing values, encoding, outliers, normalization, train/test split)
3. Model training and evaluation

## Dataset

- **Source:** HCV Data (Lichtinghagen, Klawonn & Hoffmann, 2020), UCI Machine Learning Repository
- **Samples:** 615 individuals
- **Features:** 12 (age, sex, and 10 biochemical markers: ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT)
- **Target:** `Category` — 0 = Blood Donor, 0s = Suspect Blood Donor, 1 = Hepatitis, 2 = Fibrosis, 3 = Cirrhosis


## Project structure

```
.
├── main.py              # Full pipeline: EDA, pre-processing, and model training
├── hcvdata/
   └── hcvdata.csv
          
```

## Methodology

**Exploratory Data Analysis**
- Summary statistics, missing values, and duplicate checks
- Boxplots and histograms per biomarker
- Correlation heatmap between numeric variables
- Class distribution analysis (revealing strong class imbalance)

**Pre-processing**
- Missing values imputed with `IterativeImputer` (using an `ExtraTreesRegressor` estimator)
- `Sex` one-hot encoded; `Category` label-encoded
- Outliers capped using the IQR method (replaced with the median)
- Numeric features scaled with `MinMaxScaler`
- 75/25 train/test split (`random_state=0`)

**Models**
- Logistic Regression
- Random Forest (`n_estimators=100`)

Both models are evaluated with `classification_report` (precision, recall, f1-score, accuracy) and confusion matrices.

## Results (summary)

| Model | Accuracy |
|---|---|
| Logistic Regression | 89% |
| Random Forest | 91% |

Both models perform well on the majority class (healthy blood donors) but struggle with minority clinical classes (suspect donors, hepatitis, fibrosis), reflecting the strong class imbalance in the dataset.

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

```

## Usage

The dataset is included in this repository at `hcvdata/hcvdata.csv`. Simply run:

```bash
python main.py
```

This will print the EDA summary and model metrics to the console, and save all generated plots (boxplots, histograms, heatmap, confusion matrices) to the `figures/` folder.

## Limitations & future work

- Strong class imbalance biases models toward the majority class
- Small dataset size (615 samples) limits generalization
- No hyperparameter tuning was performed

Possible improvements: class-balancing techniques (e.g., SMOTE), hyperparameter optimization (`GridSearchCV`), a larger/more balanced dataset, and testing additional model types.

## References
Key sources: Lichtinghagen et al. (2020) for the dataset; Pedregosa et al. (2011) for scikit-learn; McKinney (2010) for pandas; Hunter (2007) for Matplotlib; Waskom (2021) for seaborn.
