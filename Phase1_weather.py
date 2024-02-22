import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("C:/Users/gokul/OneDrive/Desktop/Training and internship/nexus info/weather.csv")
print(df)
print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
#********** Data preprocessing*************
print(df.columns)
print(df.dtypes)
#Drop unnecessary columns
df.drop('Unnamed: 0',inplace=True,axis=1)
print(df.shape)
# handling missing value
 # check null value
print(df.isna().sum())
df.dropna(inplace=True)
print(df.shape)
#check duplicate value
print(df.duplicated().sum()) # we do not have duplicate value
# convert Date from object to date time
df['Date']=pd.to_datetime(df['Date'])

# *** separate into categorical and numerical columns
categorical_columns= [feature for feature in df.columns if df[feature].dtype== 'O']
numeric_cols= [feature for feature in df.columns if df[feature].dtype!='O']

print(categorical_columns)
# for value count
for feature in categorical_columns:
    print(df[feature].value_counts())
    
# converting categorical columns into numerical column  using data encoding (Label encoding)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for feature in categorical_columns:
    df[feature]= encoder.fit_transform(df[feature]) 
    
# print(df.head())

#********* Split dataset into feature and target feature *******
X= df.drop('RainTomorrow',axis=1) # Normal feature
print(X.columns)
Y=df['RainTomorrow'] # Target column


# Normalize numeric features
X[numeric_cols] = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std()
print(X[numeric_cols])

#************* EDA *******************
print(X[numeric_cols].corr())
plt.figure(figsize=(10,8))
print(sns.heatmap(X[numeric_cols].corr(),annot=True))
plt.show()

# Distribution of numeric features
plt.figure(figsize=(12, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(4, 5, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(col) 
plt.tight_layout()
plt.show()

#*********** Feature Engineering *********
# Create new features
# extract year, month, and day fro, the 'Date'
df['year']=pd.to_datetime(df['Date']).dt.year
df['month']=pd.to_datetime(df['Date']).dt.month
df['day']=pd.to_datetime(df['Date']).dt.day

# calculate the difference between max and min temp
df['Tempdiff']=df['MaxTemp']-df['MinTemp']

#perform imputation for missing value and fill using the mean value

from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy='mean')

numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                'Temp9am', 'Temp3pm', 'RISK_MM']
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


# Drop unnecessary columns
df.drop([ 'Date'], axis=1, inplace=True)
# Encode categorical variables
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split the dataset into features and target variable
x = df.drop('RainTomorrow', axis=1)  # indipendent data points
y = df['RainTomorrow']  # Target or dependent datapoints


# ************* MODEL TRAINING *************************


# split train test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.30,random_state=42)
print(x_train.shape , y_train.shape, x_test.shape,y_test.shape)

# Initialize the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model
rf_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#********* MODEL EVALUATION ***********
# Make predictions on the test set
y_pred = rf_classifier.predict(x_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:")
print(confusion_mat)
                                           




 
 






