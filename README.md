<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
#### STEP 1:
Importing the libraries<BR>
#### STEP 2:
Importing the dataset<BR>
#### STEP 3:
Taking care of missing data<BR>
#### STEP 4:
Encoding categorical data<BR>
#### STEP 5:
Normalizing the data<BR>
#### STEP 6:
Splitting the data into test and train<BR>

###  PROGRAM:
```
ENTER YOUR NAME : SANDHIYA R
ENTER YOUR REGISTER NO : 212222230129
DATE : 25.02.2024
```
```
#import libraries

import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
d=pd.read_csv("Churn_Modelling.csv")
df=pd.DataFrame(d)
d.head()
d

#Finding Missing Values
print(d.isnull().sum())

d.info()
d.drop(['Surname', 'Geography','Gender'], axis=1)

#Check for Duplicates
print(d.duplicated().sum())

#Detect Outliers
#Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = d.quantile(0.25)
Q3 = d.quantile(0.75)

#Calculate the IQR
IQR = Q3 - Q1

#Normalize the dataset
#Create an instance of MinMaxScaler
scaler = MinMaxScaler()

#Define the columns to be normalized
columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

#Normalize the specified columns
d[columns] = scaler.fit_transform(d[columns])

#Display the normalized dataset
print("NORMALIZED DATASET\n",d)

#split the dataset into input and output
X = d.iloc[:,:-1].values
print("INPUT(X)\n",X)
y = d.iloc[:,-1].values
print("OUTPUT(y)\n",y)


#splitting the data for training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train\n")
print(X_train)
print("\nX_test\n")
print(X_test)
print("\nY_train\n")
print(y_train)
print("\nY_test\n")
print(y_test)
```

# OUTPUT:
### d.head()
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/60c5490b-fb05-4188-a282-9c54b39ee54e)
### d.isnull().sum()
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/82e978f7-ad98-41aa-bdd8-74ac109db495)
### d.sum()
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/c844b913-128c-44cf-aa30-4eec860395eb)
### After dropping categorical data
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/d6957cb5-1fd0-4b9c-8093-1c342dfd9635)
### Number of duplicates
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/030892a6-0d89-425d-9dbb-d13f333ba8bc)
### Identifying outliers
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/d861be26-f8d1-4304-93bf-a59e2e3bafc5)
### Normalized dataset
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/64271293-d348-4105-9233-7efd98c04446)
### X and Y
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/b2d5b3ef-d10d-4adf-bde7-69dd6775d2aa)
### X_train,X_test,Y_train,Y_test
![image](https://github.com/SandhiyaR1/Ex-1-NN/assets/113497571/9be50cad-bf08-4c49-ab7c-553d3142bf02)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


