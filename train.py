import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,  classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')


# TODO: Load regression dataset
df = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv")
df = pd.DataFrame(df)

print("Shape: ",df.shape)
print("Columns Names: ",df.columns)


target_var = "Outcome"
numerical_var = [feature for feature in df.columns if df[feature].dtype != 'O' and feature != target_var]
print("Number of Numerical Variable: ", len(numerical_var))
df.head()


##### 2
df[numerical_var].describe().T
print(df[numerical_var].isnull().sum())

# 1 - No missing Values

print("\n\n\n")

features_with_zero_value = [feature for feature in numerical_var if (df[feature] == 0).any()]
print(features_with_zero_value)
# No value in Medical database could be zero


(df[features_with_zero_value] == 0).sum()

# Handling values with 0
df[features_with_zero_value] = df[features_with_zero_value].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# 2 - Feature Separation
X = df.drop(target_var,axis = 1)
y = df[target_var]


# 3 - Outlier Detection IQR
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

features_with_outlier = [
    feature for feature in numerical_var
    if ((X[feature] < lower_bound[feature]) | (X[feature] > upper_bound[feature])).any()
]
print("Values with outliers",features_with_outlier)
#Not removing the outliers since it's a medical based dataset


# 4 - Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5 - Feature Enginnering
X["BMI_Age_Risk"] = X["BMI"] * X["Age"]
X["Glucose_BMI"] = X["Glucose"] * X["BMI"]


# 3
pipeline = Pipeline(steps=[
     ("model",RandomForestClassifier( 
         n_estimators = 200,
         max_depth = 3,
         random_state = 42
     ))
])


#5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)
pipeline.fit(X_train, y_train)

#6
cv_score = cross_val_score(pipeline, X_train, y_train, cv = 5)

print("CV score: ",cv_score)
print("Average: ", np.mean(cv_score))
print("Standard Deviation: ",np.std(cv_score))

#7

param_grid = {
    'model__n_estimators': [100,200,300],
    'model__max_depth': [5,10],
    'model__min_samples_split': [2,5]
}
grid = GridSearchCV(pipeline,
                    param_grid,
                    cv=5,
                    n_jobs=-1)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("\nBest Score:", grid.best_score_)

#8
best_model = grid.best_estimator_


#9
y_pred = best_model.predict(X_test)

print("\nAccuracy: ",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# pickle
with open("pipeline.pkl","wb") as f:
    pickle.dump(pipeline,f)