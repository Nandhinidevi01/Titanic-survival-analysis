import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('Titanic-Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())
# Checking for missing values
print(df.isnull().sum())
# Filling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin', inplace=True)
# Removing duplicates
df.drop_duplicates(inplace=True)
print(df.isnull().sum())
# Converting categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
df['Age'].plot(kind='hist', bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

# Box Plots
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survived')
plt.show()

# Scatter Plots
plt.scatter(df['Age'], df['Fare'])
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Dropping non-numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Creating a new feature 'Family_Size'
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Dropping less important features
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
print(df.head())

# Summary Statistics
print(df.describe())

# Group By Analysis
print(df.groupby('Survived').mean())

# Correlation Analysis
sns.pairplot(df, hue='Survived', diag_kind='kde')
plt.show()
