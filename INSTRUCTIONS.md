# Heart Attack Detection - Setup and Execution Instructions

## 🚀 Quick Start Guide

**Note: All analysis is consolidated in `main.ipynb` - you only need to run this single notebook!**

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n heart-attack-analysis python=3.9
conda activate heart-attack-analysis

# Install required packages
pip install -r requirements.txt
```

#### Option B: Using pip with virtual environment
```bash
# Create virtual environment
python -m venv heart_attack_env

# Activate environment (Windows)
heart_attack_env\Scripts\activate

# Activate environment (macOS/Linux)
source heart_attack_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Verification

Before running the analysis, ensure you have the required data files:
- `Heart_Attack.csv` - Original dataset
- `Heart_Attack_Cleaned.csv` - Cleaned dataset (will be created if not present)

### 3. Running the Analysis

#### Complete Analysis in One Notebook
```bash
# Open the main notebook containing ALL analysis
jupyter notebook main.ipynb
```

**That's it!** Everything is in `main.ipynb`:
- Exploratory Data Analysis
- Statistical Testing (Welch's t-test)
- Data Preprocessing
- Machine Learning Models
- Visualizations and Results

## 📋 Execution Order

**Simple!** Just run `main.ipynb` from start to finish:

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Run All Cells Sequentially** (or use "Run All")
   - The notebook is organized in logical sections
   - Each section builds on the previous one
   - All outputs will be generated in order

### Notebook Sections (All in `main.ipynb`):
1. **Data Exploration** - Load and inspect the dataset
2. **Data Visualization** - Create comprehensive plots and charts
3. **Statistical Analysis** - Perform Welch's t-test and other statistical tests
4. **Data Preprocessing** - Clean and prepare data for modeling
5. **Machine Learning** - Train and evaluate models
6. **Results & Conclusions** - Summary of findings and model performance

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Missing Dependencies
```bash
# Error: ModuleNotFoundError
# Solution: Install missing packages
pip install [package_name]
```

#### Issue 2: Data File Not Found
```bash
# Error: FileNotFoundError: Heart_Attack.csv
# Solution: Ensure CSV files are in the root directory with main.ipynb
ls Heart_Attack*.csv
```

#### Issue 3: Rich Library Display Issues
```bash
# If Rich tables don't display properly in Jupyter
# Solution: Install Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

#### Issue 4: Memory Issues with Large Dataset
```python
# Add to the beginning of main.ipynb if memory issues occur
import gc
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)  # Limit displayed rows
```

## 📊 Expected Outputs

### 1. Data Exploration Outputs
- Dataset shape and basic information
- Missing value summary
- Statistical summary tables
- Data type information

### 2. Visualization Outputs
- Distribution histograms for all numerical features
- Correlation heatmap
- Box plots comparing groups
- Scatter plots showing feature relationships

### 3. Statistical Testing Results
- Welch's t-test results with p-values
- Confidence intervals
- Descriptive statistics by group
- Professionally formatted tables

### 4. Machine Learning Results
- Model performance metrics (R², MSE, MAE)
- Feature importance rankings
- Training and validation scores
- Saved model files in `models/` directory

## 🎯 Customization Options

### Modifying Analysis Parameters

#### Change Statistical Significance Level
```python
# In main.ipynb statistical testing section, modify alpha value
alpha = 0.01  # Change from default 0.05 for stricter testing
```

#### Adjust Model Parameters
```python
# In main.ipynb machine learning section, modify model hyperparameters
from sklearn.model_selection import GridSearchCV
# Add grid search for optimal parameters
```

#### Custom Visualizations
```python
# In main.ipynb visualization sections, add custom plots
plt.figure(figsize=(12, 8))  # Adjust figure size
sns.set_style("whitegrid")   # Change plot style
```

## 📝 Output File Locations

After running the complete analysis, you'll find:

### Generated Files:
- `models/best_linear_regression.pkl` - Trained model
- `models/feature_scaler.pkl` - Feature scaling object
- `models/feature_names.pkl` - Feature names for model
- Various plot images (if saving is enabled)

### Jupyter Notebook Outputs (from `main.ipynb`):
- Cell outputs with tables and visualizations
- Statistical test results
- Model performance metrics
- All analysis results in one consolidated notebook

## 🔄 Reproducing Results

To ensure reproducibility:

1. **Set Random Seeds**
```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
```

2. **Use Fixed Train-Test Splits**
```python
# In main.ipynb machine learning section
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

3. **Document Package Versions**
```bash
pip freeze > requirements_exact.txt
```

## 🚨 Important Notes

- **Data Privacy**: Ensure compliance with data privacy regulations when working with medical data
- **Clinical Validation**: Results should be validated by medical professionals before clinical use
- **Model Limitations**: The models are for educational/research purposes and should not replace professional medical diagnosis
- **Regular Updates**: Keep dependencies updated but test for compatibility

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure data files are in the correct location
4. Check Python version compatibility (3.7+)
5. Open an issue on the GitHub repository

## 🔗 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Rich Documentation](https://rich.readthedocs.io/en/stable/)