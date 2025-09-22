# Heart Attack Detection Analysis

A comprehensive data analysis and machine learning project focused on heart attack detection using clinical and demographic data. This repository contains exploratory data analysis, statistical testing, data preprocessing, and machine learning model development.

## 📊 Project Overview

This project analyzes heart attack risk factors using a dataset of clinical measurements and patient demographics. The analysis includes statistical hypothesis testing, data visualization, feature engineering, and the development of predictive models to identify individuals at risk of heart attacks.

## 🗂️ Repository Structure

```
Heart-Attack-Detection/
├── README.md                    # Project documentation
├── INSTRUCTIONS.md              # Detailed setup and execution guide
├── requirements.txt             # Python dependencies
├── Heart_Attack.csv            # Original dataset
├── Heart_Attack_Cleaned.csv    # Cleaned dataset
├── main.ipynb                  # Complete analysis notebook (ALL ANALYSIS HERE)
└── models/                     # Saved models and preprocessing objects
    ├── best_linear_regression.pkl
    ├── feature_names.pkl
    └── feature_scaler.pkl
```

## 📋 Dataset Description

The dataset contains clinical and demographic information for heart attack prediction:

### Features:
- **age**: Patient age in years
- **sex**: Gender (0 = female, 1 = male)
- **Chest pain type**: Type of chest pain (1-4 scale)
- **trestbps**: Resting blood pressure (mm Hg)
- **cholesterol**: Serum cholesterol (mg/dl)
- **fasting blood sugar**: Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
- **resting ecg**: Resting electrocardiogram results (0-2 scale)
- **max heart rate**: Maximum heart rate achieved
- **exercise angina**: Exercise-induced angina (0 = no, 1 = yes)
- **oldpeak**: ST depression induced by exercise relative to rest
- **ST slope**: Slope of the peak exercise ST segment (1-3 scale)
- **target**: Heart attack risk (1 = low risk, 2 = high risk)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed with the following packages:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter rich
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dinhminh0307/Heart-Attack-Detection.git
cd Heart-Attack-Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📈 Analysis Components (All in `main.ipynb`)

The complete analysis is consolidated in a single notebook (`main.ipynb`) containing:

### 1. Exploratory Data Analysis
- **Data Loading & Inspection**: Basic dataset information, missing values, data types
- **Statistical Summary**: Descriptive statistics for all features
- **Data Visualization**:
  - Distribution plots for numerical features
  - Correlation heatmaps
  - Box plots for categorical variables
  - Scatter plots for feature relationships

### 2. Statistical Testing
- **Welch's t-test**: Comparing maximum heart rate between patients with and without exercise-induced angina
- **Hypothesis Testing**: Formal statistical analysis with p-values and confidence intervals
- **Rich Table Formatting**: Professional statistical result presentation

### 3. Data Preprocessing
- **Missing Value Handling**: Detection and treatment of missing data
- **Outlier Detection**: Identification of unusual values in medical metrics
- **Feature Scaling**: Standardization of numerical features
- **Data Cleaning**: Removal of inconsistent or invalid records

### 4. Machine Learning Models
- **Linear Regression**: Baseline model for heart attack risk prediction
- **Model Evaluation**: Performance metrics and validation
- **Feature Importance**: Analysis of most predictive features
- **Model Persistence**: Saved trained models for future use

## 🔍 Key Findings

### Statistical Insights:
- Analysis of heart rate differences between angina groups
- Correlation patterns between clinical measurements
- Distribution characteristics of risk factors

### Model Performance:
- Linear regression model achieving competitive accuracy
- Feature importance rankings for clinical decision support
- Cross-validation results and model reliability metrics

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Distribution Plots**: Understanding feature distributions and normality
- **Correlation Matrices**: Identifying relationships between variables
- **Box Plots**: Comparing distributions across different groups
- **Scatter Plots**: Exploring feature interactions
- **Statistical Test Results**: Professional formatting of hypothesis test outcomes

## 🛠️ Usage Instructions

### Running the Complete Analysis:
1. Open `main.ipynb` in Jupyter Notebook
2. Run all cells sequentially to reproduce the complete analysis
3. The notebook contains all components: EDA, statistical testing, preprocessing, and modeling

### What You'll Get:
- **Exploratory Data Analysis**: Comprehensive data exploration and visualization
- **Statistical Testing**: Welch's t-test and other statistical analyses
- **Data Preprocessing**: Cleaned and prepared data for modeling
- **Machine Learning**: Trained models saved to the `models/` directory
- **Results**: All outputs, visualizations, and model performance metrics

### Custom Analysis:
- Modify notebook cells to explore different hypotheses
- Add new statistical tests or machine learning algorithms
- Create additional visualizations for specific insights
- All analysis components are in one place for easy modification

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `main.ipynb` | **MAIN FILE** - Complete analysis workflow with all components: EDA, statistical testing, preprocessing, and modeling |
| `Heart_Attack.csv` | Original raw dataset |
| `Heart_Attack_Cleaned.csv` | Preprocessed and cleaned dataset |
| `README.md` | Project documentation and overview |
| `INSTRUCTIONS.md` | Detailed setup and execution instructions |
| `requirements.txt` | Python package dependencies |
| `models/` | Directory containing saved trained models and preprocessing objects |
