#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Data Preprocessing and Exploratory Data Analysis - COMPLETED
# 
# **Student Name:** Samson Silver  
# **Student ID:** 815337747  
# **Points:** 10 (8 points for Assignment + 2 points for survey)
# 
# ## Assignment Overview
# This completed notebook works with the "Salary Survey" dataset, which contains salary information and workplace characteristics from thousands of respondents. This dataset presents typical challenges found in real-world data science projects.
# 
# ## Dataset Information
# - **File:** `salary_survey.csv`
# - **Content:** Salary information and workplace characteristics
# - **Size:** 27,940 records with 18 columns

# ## Import Required Libraries
# Import all necessary libraries for data analysis, visualization, and preprocessing.

# In[ ]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import re
from collections import Counter

# Try to import missingno with graceful fallback
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
    print("missingno package imported successfully")
except ImportError:
    MISSINGNO_AVAILABLE = False
    print("missingno package not available, using seaborn/matplotlib alternatives")

# Optional: scikit-learn for future modeling
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    print("scikit-learn imported successfully")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available")

# Set plotting style and random seed
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("All libraries imported and configured successfully!")


# # Task 1: Data Description and Exploration (2 Points)
# 
# ## 1.1 Dataset Overview
# **Instructions:** 
# - Load the dataset from the CSV file `salary_survey.csv`
# - Display basic statistics (shape, columns, data types)
# - Create a comprehensive data dictionary explaining each variable
# - Identify potential target variable(s) for future modeling

# In[ ]:


# Load the dataset
try:
    df = pd.read_csv('salary_survey.csv')
    print("Dataset loaded successfully")
except UnicodeDecodeError:
    df = pd.read_csv('salary_survey.csv', encoding='latin-1')
    print("Dataset loaded with latin-1 encoding")

# Store original data
df_original = df.copy()

print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
display(df.head())


# In[ ]:


# Dataset info and statistics
print("=== DATASET INFO ===")
df.info()

print("\n=== DESCRIPTIVE STATISTICS ===")
display(df.describe(include='all'))

print("\nLast 5 rows:")
display(df.tail())


# In[ ]:


# Programmatic data dictionary analysis
print("=== DATA DICTIONARY ANALYSIS ===")

data_dict = []
for col in df.columns:
    col_info = {
        'Column': col[:50] + '...' if len(col) > 50 else col,
        'Data_Type': str(df[col].dtype),
        'Non_Null': df[col].count(),
        'Null_Count': df[col].isnull().sum(),
        'Null_Pct': round(df[col].isnull().mean() * 100, 1),
        'Unique': df[col].nunique()
    }

    # Infer type
    if df[col].dtype in ['int64', 'float64']:
        col_info['Type'] = 'Numeric'
    elif df[col].nunique() / len(df) < 0.05:
        col_info['Type'] = 'Categorical'
    else:
        col_info['Type'] = 'Free_Text'

    data_dict.append(col_info)

dict_df = pd.DataFrame(data_dict)
display(dict_df)

# Sample values for key columns
key_cols = ['How old are you?', 'What industry do you work in?', 
           'Please indicate the currency', 'What country do you work in?']

for col in key_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        if df[col].nunique() <= 15:
            print("  Value counts:")
            print(df[col].value_counts().head(10))
        else:
            print(f"  Sample values: {list(df[col].dropna().unique()[:5])}")


# ## Comprehensive Data Dictionary
# 
# Based on the analysis above:
# 
# ### **Survey Metadata**
# - **Timestamp**: Survey submission date/time (MM/DD/YYYY HH:MM:SS)
# 
# ### **Demographics**
# - **How old are you?**: Age groups (25-34, 35-44, etc.)
# - **What is your gender?**: Gender identity categories
# - **What is your race?**: Racial/ethnic identity (multiple selections)
# 
# ### **Employment**
# - **What industry do you work in?**: Industry categories
# - **Job title**: Free-text job titles
# - **Job title context**: Optional clarification
# 
# ### **Compensation (TARGET VARIABLES)**
# - **Annual salary**: Primary target - yearly salary (string format with commas)
# - **Additional compensation**: Bonuses, overtime, etc.
# - **Currency**: Currency codes (USD, GBP, EUR, etc.)
# - **Other currency**: Free-text currency specification
# - **Income context**: Optional income clarification
# 
# ### **Experience & Education**
# - **Overall experience**: Total work experience ranges
# - **Field experience**: Experience in current field
# - **Education level**: Highest education completed
# 
# ### **Location**
# - **Country**: Work country
# - **US State**: US state (if applicable)
# - **City**: Work city
# 
# ### **Target Variables for Modeling**
# 1. **Primary**: Annual salary (after cleaning)
# 2. **Secondary**: Total compensation (salary + additional)
# 3. **Alternative**: Salary categories/bands

# ### 📊 Your Analysis (Task 1.1)
# 
# The salary survey dataset contains **27,940 responses** across **18 columns**, representing a substantial sample for compensation analysis. Key observations:
# 
# **Data Quality Overview:**
# - Most demographic and employment fields have complete data
# - Optional fields show expected higher missingness
# - Multi-country, multi-currency dataset requiring normalization
# 
# **Key Patterns:**
# 1. **Demographics**: Concentration in 25-34 age group suggests tech/professional sample
# 2. **Industries**: Computing/Tech heavily represented
# 3. **Geography**: Primarily English-speaking countries
# 4. **Salary Format**: Requires cleaning (commas, string format)
# 
# **Modeling Potential**: Excellent for salary prediction with clear predictors (demographics, experience, location, industry) and targets (salary, total compensation).
# 
# **Collection Context**: Online survey likely distributed through professional networks, explaining demographic skew toward tech professionals.

# ### 🤖 AI-Assisted Analysis (Task 1.1)
# 
# **Dataset Characteristics:**
# 
# This salary survey represents modern crowdsourced compensation data with notable features:
# 
# **Strengths:**
# - **Large Sample**: 27,940+ responses provide strong statistical power
# - **Comprehensive Coverage**: Captures key salary determinants
# - **International Scope**: Multi-currency analysis capabilities
# - **Experience Granularity**: Overall vs. field-specific experience tracking
# 
# **Potential Challenges:**
# 1. **Selection Bias**: Tech professional skew from distribution channels
# 2. **Self-Reporting**: Accuracy depends on respondent honesty
# 3. **Currency/PPP**: Cross-country comparisons need adjustment
# 4. **Temporal Variation**: Responses span different economic periods
# 
# **Investigation Strategies:**
# - Analyze temporal patterns in responses
# - Examine outliers for data entry errors
# - Cross-validate demographics against census data
# - Investigate systematic missing data patterns
# 
# **Industry Context**: Reflects salary transparency trends in tech/professional sectors, enabling compensation equity analysis across demographics and regions.

# ## 1.2 Initial Data Quality Assessment
# **Instructions:**
# - Calculate missing value percentages and create visualizations
# - Identify formatting issues in numeric and categorical fields
# - Detect outliers using statistical methods
# - Assess data collection issues and their implications

# In[ ]:


# Missing values analysis
print("=== MISSING VALUES ANALYSIS ===")

missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': round(df.isnull().mean() * 100, 2),
    'Data_Type': df.dtypes
}).sort_values('Missing_Percentage', ascending=False)

print("Missing value summary:")
display(missing_stats)

# Visualization of missing values
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Bar chart of missing percentages
missing_pct = df.isnull().mean() * 100
missing_pct_sorted = missing_pct.sort_values(ascending=True)
axes[0].barh(range(len(missing_pct_sorted)), missing_pct_sorted.values)
axes[0].set_yticks(range(len(missing_pct_sorted)))
axes[0].set_yticklabels([col[:30] + '...' if len(col) > 30 else col for col in missing_pct_sorted.index])
axes[0].set_xlabel('Missing Percentage (%)')
axes[0].set_title('Missing Values by Column')
axes[0].grid(True, alpha=0.3)

# Missingness heatmap (alternative to missingno)
if MISSINGNO_AVAILABLE:
    plt.figure(figsize=(15, 8))
    msno.matrix(df)
    plt.title('Missingness Pattern Matrix')
    plt.show()
else:
    # Alternative heatmap using seaborn
    missing_data = df.isnull()
    sns.heatmap(missing_data.T, cbar=True, yticklabels=True, 
                cmap='viridis', ax=axes[1])
    axes[1].set_title('Missing Data Pattern (Yellow = Missing)')
    axes[1].set_xlabel('Row Index')

plt.tight_layout()
plt.show()

