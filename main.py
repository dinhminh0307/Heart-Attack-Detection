#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

# Read the CSV file
df = pd.read_csv('Heart_Attack.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
df.head()


# In[2]:


# Display basic information about the dataset
print("Dataset Information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())


# In[3]:


# Statistical summary of the dataset
print("Statistical Summary:")
df.describe()


# In[4]:


# Check for any unusual values (zeros, negatives) in key medical metrics
print("Check for unusual values:")
print(f"Cholesterol min: {df['cholesterol'].min()}, max: {df['cholesterol'].max()}")
print(f"Blood pressure min: {df['trestbps'].min()}, max: {df['trestbps'].max()}")
print(f"Max heart rate min: {df['max heart rate'].min()}, max: {df['max heart rate'].max()}")

# Plot distributions of key features
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')

plt.subplot(2, 3, 2)
sns.histplot(df['cholesterol'], kde=True)
plt.title('Cholesterol Distribution')

plt.subplot(2, 3, 3)
sns.histplot(df['trestbps'], kde=True)
plt.title('Resting Blood Pressure')

plt.subplot(2, 3, 4)
sns.histplot(df['max heart rate'], kde=True)
plt.title('Max Heart Rate Distribution')

plt.subplot(2, 3, 5)
sns.countplot(x='sex', data=df)
plt.title('Sex Distribution (0=Female, 1=Male)')

plt.subplot(2, 3, 6)
sns.countplot(x='target', data=df)
plt.title('Target Distribution (0=No disease, 1=Disease)')

plt.tight_layout()
plt.show()


# In[5]:


# Check correlations between features
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Check class distribution
print("Target Class Distribution:")
print(df['target'].value_counts())
print(f"Class Balance: {df['target'].value_counts(normalize=True)}")


# # Data Cleaning Process
# 
# From our exploratory data analysis, we've identified several issues that need addressing:
# 
# 1. **Zero values in cholesterol and blood pressure**: These are likely errors since these metrics can't be zero in living patients.
# 2. **Outliers in cholesterol**: There are some very high values (up to 603) that might be outliers.
# 3. **No missing values**: The dataset appears to be complete, so we don't need to handle missing values.
# 4. **Target has 3 classes**: We have 0, 1, and 2 values in the target column. We'll keep this as is, but it's important to note that this is a multi-class classification problem.
# 
# Let's clean these issues one by one.

# In[6]:


# Step 1: Handle duplicates
print("Before removing duplicates:", df.shape)
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)


# In[7]:


# Step 2: Handle zero values in medical metrics
# First, let's see how many zeros we have in each column
print("Number of zeros in key medical columns:")
print(f"Cholesterol: {(df['cholesterol'] == 0).sum()}")
print(f"Blood pressure (trestbps): {(df['trestbps'] == 0).sum()}")

# Replace zeros in cholesterol with the median (medical metrics can't be zero)
median_cholesterol = df[df['cholesterol'] > 0]['cholesterol'].median()
df['cholesterol'] = df['cholesterol'].replace(0, median_cholesterol)

# Replace zeros in blood pressure with the median
median_trestbps = df[df['trestbps'] > 0]['trestbps'].median()
df['trestbps'] = df['trestbps'].replace(0, median_trestbps)

# Verify the replacements
print("\nAfter replacement:")
print(f"Cholesterol min: {df['cholesterol'].min()}, max: {df['cholesterol'].max()}")
print(f"Blood pressure min: {df['trestbps'].min()}, max: {df['trestbps'].max()}")


# In[8]:


# Step 3: Handle outliers
# For cholesterol, values above 500 are rare and may be outliers
# Let's examine the cholesterol distribution with a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['cholesterol'])
plt.title('Cholesterol Distribution - Boxplot')
plt.show()

# Calculate statistics for cholesterol
print("Cholesterol Statistics:")
print(f"95th percentile: {df['cholesterol'].quantile(0.95)}")
print(f"99th percentile: {df['cholesterol'].quantile(0.99)}")
print(f"Max value: {df['cholesterol'].max()}")

# Let's cap cholesterol values at the 99th percentile
cholesterol_cap = df['cholesterol'].quantile(0.99)
df['cholesterol'] = df['cholesterol'].clip(upper=cholesterol_cap)

# Verify the change
print(f"\nAfter capping outliers, cholesterol max: {df['cholesterol'].max()}")


# In[9]:


# Step 4: Check data types and ensure consistency
# Let's look at the data types of all columns
print("Data types before conversion:")
print(df.dtypes)

# The data already seems to be in numeric format, but let's check the unique values
# to make sure they're properly encoded
print("\nUnique values in categorical columns:")
categorical_cols = ['sex', 'Chest pain type', 'resting ecg', 'exercise angina', 'ST slope']
for col in categorical_cols:
    print(f"{col}: {sorted(df[col].unique())}")

# No need for encoding since all values are already numeric
# But we should document what the numeric values mean
print("\nCategorical columns interpretation:")
print("sex: 0 = Female, 1 = Male")
print("Chest pain type: Values 0-3 (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)")
print("resting ecg: Values 0-2 (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)")
print("exercise angina: 0 = No, 1 = Yes")
print("ST slope: Values 0-2 (0: Upsloping, 1: Flat, 2: Downsloping)")
print("target: 0 = No disease, 1 = Disease, 2 = Severe disease")


# In[10]:


# Step 5: Handle inconsistent categorical values
# Check how many records have these unexpected values
print("Records with unexpected category values:")
print(f"Chest pain type = 4: {(df['Chest pain type'] == 4).sum()}")
print(f"ST slope = 3: {(df['ST slope'] == 3).sum()}")

# Replace these with the most common value in each column (mode)
if (df['Chest pain type'] == 4).sum() > 0:
    mode_chest_pain = df[df['Chest pain type'] != 4]['Chest pain type'].mode()[0]
    df.loc[df['Chest pain type'] == 4, 'Chest pain type'] = mode_chest_pain
    print(f"Replaced Chest pain type 4 with {mode_chest_pain}")

if (df['ST slope'] == 3).sum() > 0:
    mode_st_slope = df[df['ST slope'] != 3]['ST slope'].mode()[0]
    df.loc[df['ST slope'] == 3, 'ST slope'] = mode_st_slope
    print(f"Replaced ST slope 3 with {mode_st_slope}")

# Verify the changes
print("\nAfter fixing inconsistent values:")
for col in categorical_cols:
    print(f"{col}: {sorted(df[col].unique())}")


# In[11]:


# Step 6: Final data validation
# Let's check that all our cleaning steps have been applied properly
print("Final Data Validation:")
print(f"Number of rows in cleaned dataset: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Cholesterol range: {df['cholesterol'].min()} - {df['cholesterol'].max()}")
print(f"Blood pressure range: {df['trestbps'].min()} - {df['trestbps'].max()}")

# Visualize the final distributions of the key medical metrics
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['cholesterol'], kde=True)
plt.title('Cleaned Cholesterol Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['trestbps'], kde=True)
plt.title('Cleaned Blood Pressure Distribution')

plt.subplot(1, 3, 3)
sns.countplot(x='target', data=df)
plt.title('Target Distribution')

plt.tight_layout()
plt.show()

# Display the final cleaned dataset
print("\nFinal Cleaned Dataset:")
df.head()


# # Data Cleaning Summary
# 
# In this analysis, we performed the following data cleaning steps:
# 
# 1. **Removed duplicates**: Reduced dataset from 1763 to 1491 rows by removing duplicate entries.
# 
# 2. **Handled invalid values**:
#    - Replaced zero values in cholesterol with the median (172 records affected)
#    - Replaced zero values in blood pressure (1 record affected)
#    - Fixed inconsistent category values:
#      - Replaced Chest pain type value 4 with 3 (625 records)
#      - Replaced ST slope value 3 with 1 (81 records)
# 
# 3. **Addressed outliers**:
#    - Capped cholesterol values at the 99th percentile (409)
# 
# 4. **Validated data types**:
#    - Confirmed all features have appropriate data types
#    - Documented meaning of categorical variables
# 
# The cleaned dataset contains 1491 records with no missing values, valid ranges for all medical metrics, and consistent categorical values.

# In[12]:


# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'Heart_Attack_Cleaned.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")


# In[13]:


import scipy.stats as stats

# Create contingency table
contingency_table = pd.crosstab(df['Chest pain type'], df['target'])

# Perform Chi-Square test of independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("Contingency Table:")
print(contingency_table)
print("\nChi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("p-value:", p)


# In[ ]:




