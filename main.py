import sklearn as sklearn
import numpy as np
import pandas as pd
import sns as sns
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reads in the data and drops null value rows
df = pd.read_csv('D:\Ayaan\Documents\Heart Disease Classifier\heart_disease_uci.csv')
df = df.dropna()

# Drops unnecessary columns and sets new names for each column
df.drop('id', axis=1, inplace=True)
df.drop('dataset', axis=1, inplace=True)
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']

print(df.info())

# Transforms categorical data into numerical data
df.loc[df['chest_pain_type'] == 0, 'chest_pain_type'] = 'asymptomatic'
df.loc[df['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
df.loc[df['chest_pain_type'] == 2, 'chest_pain_type'] = 'typical'
df.loc[df['chest_pain_type'] == 3, 'chest_pain_type'] = 'non-anginal'

df.loc[df['rest_ecg_type'] == 0, 'rest_ecg_type'] = 'left ventricular hypertrophy'
df.loc[df['rest_ecg_type'] == 1, 'rest_ecg_type'] = 'normal'
df.loc[df['rest_ecg_type'] == 2, 'rest_ecg_type'] = 'ST-T wave abnormality'

df.loc[df['st_slope_type'] == 0, 'st_slope_type'] = 'downsloping'
df.loc[df['st_slope_type'] == 1, 'st_slope_type'] = 'flat'
df.loc[df['st_slope_type'] == 2, 'st_slope_type'] = 'upsloping'

df.loc[df['thalassemia_type'] == 0, 'thalassemia_type'] = 'nothing'
df.loc[df['thalassemia_type'] == 1, 'thalassemia_type'] = 'fixed defect'
df.loc[df['thalassemia_type'] == 2, 'thalassemia_type'] = 'normal'
df.loc[df['thalassemia_type'] == 3, 'thalassemia_type'] = 'reversable defect'

df.loc[df['target'] == 1, 'target'] = 2
df.loc[df['target'] == 1, 'target'] = 3
df.loc[df['target'] == 1, 'target'] = 4

data = pd.get_dummies(df, drop_first=False)

df_temp = data['thalassemia_type_fixed defect']
data = pd.get_dummies(df, drop_first=True)
data.head()

frames = [data, df_temp]
result = pd.concat(frames,axis=1)
print(result.head())

# Predicts whether the patient has heart disease or not using Logistic Regression and measures the accuracy
X = result.drop('target', axis=1)
y = df['target']
y = y.replace(2, 1)
y = y.replace(3, 1)
y = y.replace(4, 1)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train)).values
X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)).values
logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
y_pred = logistic.predict(X_test)
print(str(accuracy_score(Y_test,y_pred)) + "% accuracy")