
# Importing the Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Data Collection and Analysis

insurance_dataset = pd.read_csv('insurance.csv')

# Printing the first five rows of the dataframe

insurance_dataset.head()

# Number of rows and columns

insurance_dataset.shape


# Getting some information about datasets

insurance_dataset.info()


# Categorical columns are sex,smoker and region

# Checking for missing values
insurance_dataset.isnull().sum()


# Getting statistical measures

insurance_dataset.describe()

# Distribution of age value

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')


# Distribution of Gender column

plt.figure(figsize=(6,6))
sns.countplot(x='sex', data = insurance_dataset)
plt.title("Sex Distribution")
plt.show()


insurance_dataset['sex'].value_counts()

# Distribution of bmi

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('bmi Distribution')


# Children Distribution

plt.figure(figsize=(6,6))
sns.countplot(x='children', data = insurance_dataset)
plt.title('Children')
plt.show()


insurance_dataset['children'].value_counts()


# Smoker Distribution

plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data = insurance_dataset)
plt.title('Smoker')
plt.show()


insurance_dataset['smoker'].value_counts()


# Region Distribution

plt.figure(figsize=(6,6))
sns.countplot(x='region', data = insurance_dataset)
plt.title('Region')
plt.show()


insurance_dataset['region'].value_counts()

# Charges Value Distribution

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charge Distribution')
plt.show()

# Data Preprocessing : Categorical features encoding - Sex column

insurance_dataset.replace({'sex' : {'male':0, 'female':1}}, inplace=True)

# Categorical features encoding - Smoker column

insurance_dataset.replace({'smoker' : {'yes':0, 'no':1}}, inplace=True)

# Categorical features encoding - Region column

insurance_dataset.replace({'region' : {'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)



# Splitting the features and target variable

X = insurance_dataset.drop(columns='charges', axis=1)
y = insurance_dataset['charges']

print(X)
print(y)

# Splitting the dataset into Training and Testing Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


print(X.shape, X_train.shape, X_test.shape)


# Model Training

model = LinearRegression()

model.fit(X_train, y_train)


# Model Evaluation : Prediction on taining data

training_data_prediction = model.predict(X_train)

# R squared value

r2_train = metrics.r2_score(y_train, training_data_prediction)
print('R squared value : ', r2_train)


# Model Evaluation : Prediction on test data

test_data_prediction = model.predict(X_test)

r2_test = metrics.r2_score(y_test, test_data_prediction)
print('R squared value : ', r2_test)


# Building a Predictive System

input_data = (31,1,25.74,0,1,0)

# Changing data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])


