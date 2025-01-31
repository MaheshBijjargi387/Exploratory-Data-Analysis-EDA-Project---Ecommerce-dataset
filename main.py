# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Step 1: Load the Data

data_path = "healthcare_dataset.csv"
df = pd.read_csv(data_path)

# Step 2: Data Overview
print("Dataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nUnique Values per Column:\n")
print(df.nunique())

print("\nDuplicate Rows:", df.duplicated().sum())

# Step 3: Data Cleaning
# Drop duplicates
df = df.drop_duplicates()

# Convert dates to datetime format
date_cols = ['Date of Admission', 'Discharge Date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Standardize name formatting
df['Name'] = df['Name'].str.title()

# Fill missing values
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['number']):
    df[col].fillna(df[col].mean(), inplace=True)

# Step 4: Feature Engineering
df['Hospital Stay Duration'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Is Returning Patient'] = df.duplicated(subset=['Name'], keep=False)
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 40, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])



# Grouped data for different plots
billing_by_admission = df.groupby("Admission Type")["Billing Amount"].sum()
gender_distribution = df["Gender"].value_counts()
avg_billing_by_gender = df.groupby("Gender")["Billing Amount"].mean()
condition_count = df["Medical Condition"].value_counts().head(10)  # Top 10 conditions
age_distribution = df["Age"].value_counts().sort_index()
billing_by_hospital = df.groupby("Hospital")["Billing Amount"].sum().nlargest(10)

# 1. Total Billing Amount by Admission Type (Bar Chart)
plt.figure(figsize=(8, 5))
plt.bar(billing_by_admission.index, billing_by_admission.values, color="skyblue", edgecolor="black")
plt.title("Total Billing Amount by Admission Type")
plt.xlabel("Admission Type")
plt.ylabel("Total Billing Amount")
plt.xticks(rotation=45)
plt.show()

# 2. Gender Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
plt.pie(gender_distribution, labels=gender_distribution.index, autopct="%1.1f%%", colors=["lightcoral", "lightblue"])
plt.title("Gender Distribution")
plt.show()

# 3. Average Billing Amount by Gender (Bar Chart)
plt.figure(figsize=(8, 5))
plt.bar(avg_billing_by_gender.index, avg_billing_by_gender.values, color=["lightcoral", "lightblue"], edgecolor="black")
plt.title("Average Billing Amount by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Billing Amount")
plt.show()

# 4. Count of Patients by Medical Condition (Horizontal Bar Chart)
plt.figure(figsize=(10, 6))
plt.barh(condition_count.index, condition_count.values, color="purple", edgecolor="black")
plt.title("Top 10 Medical Conditions")
plt.xlabel("Number of Patients")
plt.ylabel("Medical Condition")
plt.show()

# 5. Age Distribution (Line Chart)
plt.figure(figsize=(10, 5))
plt.plot(age_distribution.index, age_distribution.values, marker="o", linestyle="-", color="green")
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.grid()
plt.show()

# 6. Total Billing Amount by Top 10 Hospitals (Bar Chart)
plt.figure(figsize=(10, 5))
plt.barh(billing_by_hospital.index, billing_by_hospital.values, color="orange", edgecolor="black")
plt.title("Total Billing Amount by Top 10 Hospitals")
plt.xlabel("Total Billing Amount")
plt.ylabel("Hospital")
plt.show()




# Filter the dataset to include only patients with Cancer
cancer_patients = df[df['Medical Condition'] == 'Cancer']
print(cancer_patients.head())

# Count the number of female patients with cancer
female_cancer_count = df[(df['Gender'] == 'Female') & (df['Medical Condition'] == 'Cancer')].shape[0]
print("Number of female patients with cancer:", female_cancer_count)

# Calculate the average billing amount for cancer patients
cancer_avg_billing = df[df['Medical Condition'] == 'Cancer']['Billing Amount'].mean()
print("Average billing amount for cancer patients:", cancer_avg_billing)

# Filter dataset for asthma patients
asthma_patients = df[df['Medical Condition'] == 'Asthma']
# Count occurrences of each admission type
most_common_admission = asthma_patients['Admission Type'].value_counts().idxmax()
print("Most common admission type for asthma patients:", most_common_admission)

asthma_urgent_avg_billing = df[(df['Medical Condition'] == 'Asthma') & (df['Admission Type'] == 'Urgent')]['Billing Amount'].mean()
print("Average billing amount for asthma patients with Urgent admissions:", asthma_urgent_avg_billing)


#5 create new feature
df['actual_stay_duration'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
print(df['actual_stay_duration'])















