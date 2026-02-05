# %% [markdown]
#  # Setup 
# Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from io import StringIO 
import requests

# %% [markdown]
# # Step 1: Data Loading and Idea Development
#
# Key Questions:
# - How well can we predict if students at this school will get recruited for a job?
# - An independent business metric would be student's job placements and how well they do on 
# the job/how much they contribute to the company.
# %%
# Load the data
job_url = ("https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
           "master/Placement_Data_Full_Class.csv")
jobs = pd.read_csv(job_url)
jobs.info()

# %% [markdown]
# # Step 2: Data Preparation and Initial Exploration
# %%
# Handle Missing Values
# jobs.isna().sum() 
# # Missing values in salary column - for the non-placed, can't remove these entries!
# Instead, fill with zeros
jobs['salary'] = jobs['salary'].fillna(0)
jobs.isna().sum() 
# %%
# Convert data types
# strings --> categorical columns
cat_cols = ['gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation']
jobs[cat_cols] = jobs[cat_cols].astype('category')


# strings --> booleans
jobs["workex"] = (
    jobs["workex"]
    .str.strip()
    .str.lower()
    .map({"yes": True, "no": False})
    .astype(bool)
)

jobs["status"] = (
    jobs["status"]
    .str.strip()
    .str.lower()
    .map({
        "placed": True,
        "not placed": False
    })
    .astype(bool)
)
jobs.dtypes.value_counts()
# %%
# Standardize numeric cols
numeric_cols = list(jobs.select_dtypes('number'))
jobs[numeric_cols] = MinMaxScaler().fit_transform(jobs[numeric_cols])

# %%
# One Hot Encoding
category_list = list(jobs.select_dtypes('category'))
jobs_encoded = pd.get_dummies(jobs, columns=category_list)
jobs_encoded.head()
# %%
# Establish the target variable:
jobs_encoded['placement'] = (
    jobs_encoded.status == True
).astype(bool)

jobs_encoded.placement.value_counts()
# %%
# Find the prevalence
prevalence = jobs_encoded.placement.mean()
print(f'Prevalence of job placement: {prevalence:.2%}')
# %%
# Train, Tune, & Test Split
# First drop the column(s) used to make target variable
jobs_clean = jobs_encoded.drop(columns=['status','salary'])
jobs_clean.info() # sucessfully dropped

# First Split
train, test = train_test_split(
    jobs_clean,
    train_size= 0.7,      # 70% of total entries
    stratify=jobs_clean.placement
)

# Second Split
tune, test = train_test_split(
    test,
    train_size=.5,
    stratify=test.placement
)

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"Test set shape: {tune.shape}")
# %% [markdown]
# # Step 3: Concerns
# I dropped the salary column as well since it would be very obvious which students
# were place and which weren't with that information, it would be like the model
# was cheating. Next, I worry that the dataset may be too small to make 
# accurate predictions. There also may be unseen data that affects job placement,
# since job placement is a complex factor that also has to deal with 
# immeasurable behavioral factors as well as raw test scores. 
