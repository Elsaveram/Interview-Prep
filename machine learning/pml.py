
#data manipulation tools
import numpy as np
import pandas as pd

#data visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn import datasets

###LINEAR & LOGISTIC REGRESSION WITH SKLEARN############################################################################
########################################################################################################################

#loading boston dataset from dictionary to dataframe
boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data, columns= boston.feature_names)
boston_df['target'] = boston.target

#loading iris from dictionary to dataframe
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

#Linear Regression & Logistic Regression
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

#creating a Linear regression in SKlearn
linreg_1 = LinearRegression()

#creating X, y for Boston dataset, for this toy example, we'll be using all the columns except for Y for a toy example.
b_feat_list = boston_df.columns[0:-1]
X = boston_df[b_feat_list]
y = boston_df.target

#fitting the LinearRegression object to your data
linreg_1.fit(X, y)

#getting the coefficients of your LinearRegression object
linreg_1.coef_

#getting the intercept
linreg_1.intercept_

#Getting the R2 score of your model
linreg_1.score(X, y)

#Predicting with your model
linreg_1.predict(X)

#passing in another set for your model to predict
X_test = X[0:30]
linreg_1.predict(X_test)

#creating a Linear regression in SKlearn
logit_1 = LogisticRegression()

#creating X, y from Iris dataset, fitting the LinearRegression object to your data & predicting
iris_feat_list = iris_df.columns[0:-1]
iris_X = iris_df[iris_feat_list]
iris_y = iris_df.target

logit_1.fit(iris_X, iris_y)
logit_1.predict(iris_X)

#Prediction probabilities
iris_predict_data = pd.DataFrame(logit_1.predict_proba(iris_X))
iris_predict_data

#Prediction of log odds
iris_log_predict = pd.DataFrame(logit_1.predict_log_proba(iris_X))
iris_log_predict

###DECISION TREES & TREE BASED MODELS WITH SKLEARN######################################################################
########################################################################################################################
from sklearn.tree import DecisionTreeClassifier

##DECISSION TREE
iris_tree = DecisionTreeClassifier()
iris_tree.fit(iris_X, iris_y)

#Examining feature importances of your tree model
iris_tree.feature_importances_

#A neat way to visualize your feature importance!
pd.Series(index = iris_feat_list, data = iris_tree.feature_importances_).sort_values().plot(kind = 'bar')

#Examining different classes in your class target
iris_tree.n_classes_

#counting how many features are in your model
iris_tree.n_features_

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

##RANDOM FOREST
boston_rf_1 = RandomForestRegressor()
boston_rf_1.fit(X, y)
#This will show you the base estimator of your ensemble, in this case, a decision tree regressor
boston_rf_1.base_estimator

#This will show you a list of ALL the estimators in your model
boston_rf_1.estimators_

#Predicting with model
boston_rf_1.predict(X)

##GRADIENT BOOSTING
iris_gbm = GradientBoostingClassifier()
iris_gbm.fit(iris_X, iris_y)
predicted_y=iris_gbm.predict(iris_X)

##MEtrics package
import sklearn.metrics as metrics

metrics.accuracy_score(iris_y, predicted_y)
metrics.confusion_matrix(iris_y, predicted_y)

##VALIDATION ###########################################################################################################
########################################################################################################################
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split
scores = cross_val_score(linreg_1, X, y, cv = 5)
scores

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
clf = linreg_1.fit(X_train, y_train)
clf.score(X_test, y_test)

#Predict the holdout set
predictions = cross_val_predict(linreg_1, X, y, cv = 5)
metrics.r2_score(y, predictions)

######KAGGLE WORKFLOW###################################################################################################

##Preprocess
##Save the ID column
train_ID = train_df['Id']
test_ID = test_df['Id']

# Now drop the 'Id' colum since we can not use it as a feature to train our model.
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

##Separate the response variable and the predictor variables into separate df
y_train = train_df['SalePrice']
X_train = train_df.drop('SalePrice', axis=1)
X_test = test_df.copy()

##Feature engineering:
#Combine training and test dataframes before feature engineering.
#For categorial features, this is fine because you want to avoid having new categories in the test set, which will cause
#different dimensions after dummify the data set.
#If you want to perform any transformation (normalization, standardization, etc) on the numerical features, you should
#fit on the training set and transform on the test set (use the training average for imputing missing values for example)

all_data = pd.concat([X_train, X_test], ignore_index=True)
all_data.shape

##CATEGORICAL FEATURES

#One-hot encoding
#Basic method: used with most linear algorithm
#Drop first column avoids collinearity

one_hot_df = pd.get_dummies(all_data, drop_first=True, dummy_na=True)
one_hot_df.head()

#Label encoding
#Useful for tree-based algorithms
#Does not increase dimensionality
from sklearn.preprocessing import LabelEncoder

label_df = all_data.copy()

for c in label_df.columns:
    if label_df[c].dtype == 'object':
        le = LabelEncoder()
        # Need to convert the column type to string in order to encode missing values
        label_df[c] = le.fit_transform(label_df[c].astype(str))

#Label count encoding
#For categorical features with many categories(rows:categories ration 20:1 or less)
#Rank categorical variables by count in the training set and transform the test set
#Iterate counter for each CV fold - fit on the new training set and transform on the new test set
#Useful for both linear or non-linear algorithms

class LabelCountEncoder(object):
    def __init__(self):
        self.count_dict = {}

    def fit(self, column):
        # This gives you a dictionary with level as the key and counts as the value
        count = column.value_counts().to_dict()
        # We want to rank the key by its value and use the rank as the new value
        # Your code here


    def transform(self, column):
        # If a category only appears in the test set, we will assign the value to zero.
        missing = 0
        # Your code here
        return column.map(lambda x:self.count_dict.get(x, missing))


    def fit_transform(self, column):
        self.fit(column)
        return self.transform(column)

l
abel_count_df = X_train.copy()

for c in label_count_df.columns:
    if label_count_df[c].dtype == 'object':
        lce = LabelCountEncoder()
        label_count_df[c] = lce.fit_transform(label_count_df[c])

##ORDINAL FEATURES
#Label Count encoding is good in general, however, some of the features are ordinal in nature.
#For example, we usually consider Excellent > Good > Average/Typical > Fair > Poor
#We can construct a dictionary like the following and map it to those columns:

ord_cols = ['ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'KitchenQual',
           'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1}

ord_df = X_train.copy()

for col in ord_cols:
    ord_df[col] = ord_df[col].map(lambda x: ord_dic.get(x, 0))


##PARAMETER TUNING
#Grid search
#Bayesian Optimization Method

##Ensesmble
#Averaging the predictions
#Stacking
