##############################################
# Reading data https://apps.who.int/nha/database/select/indicators/en
import pandas as pd

data = pd.read_excel("GHED_data.xlsx", 'Data')

print(data.head())
print(data.describe())
rdata = data.rename(columns={data.columns[1]:"fe3]"})
rdata.columns[1]

##############################################
# Editiing data
data.columns[0:7]
edata = data.filter(data.columns[0:7])
edata
##############################################
# Encoding (one hot and target)
ohdata = pd.get_dummies(data = edata, columns = [data.columns[3]], drop_first=True, dtype=int)
ohdata


import numpy as np



#############################################
# Multiple Logistic regression https://www.datasklr.com/logistic-regression/multinomial-logistic-regression

mlrdata = data.filter(data.columns[3:7])
mlrdata.dtypes
mlrdata = mlrdata.dropna(how='any')
X = mlrdata.filter(mlrdata.columns[1:4])

# data is string get values
y = pd.factorize(mlrdata.iloc[:,0])[0] + 1

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

import statsmodels.api as sm

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state = 5)

# if there is a plroblem with convergence try either normalizing data with sklearn's standardscaler or increasing maxiter
model1 = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=1000).fit(X_train, y_train)
preds = model1.predict(X_test)

params = model1.get_params()
print(params)

logit_model=sm.MNLogit(y_train,sm.add_constant(X_train))
logit_model
result=logit_model.fit()
stats1=result.summary()
stats2=result.summary2()
print(stats1)
print(stats2)




##############################################
# K-Nearest Neighbor algorithm  https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt

rmse_val = [] #to store rmse values for different k

for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


############################################
# Random forest  https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
import numpy as np

rfdata = pd.get_dummies(data = data.filter(data.columns[np.r_[0:6,10:25]]), columns = [data.columns[3]], drop_first=True, dtype=int)
rfdata = rfdata.dropna(how="any")
rfdata
# the value we want to predict
rfdata_label = np.array(pd.factorize(rfdata.iloc[:,1])[0]+1)
rfdata_label
rfdata_features = rfdata.filter(rfdata.columns[3:24])
rfdata_features["region"] = pd.factorize(rfdata.iloc[:,2])[0]+1
rfdata_features
feature_names = list(rfdata_features)
feature = np.array(rfdata_features)

train_features, test_features, train_labels, test_labels = train_test_split(feature, rfdata_label, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
predictions
test_labels
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
errors
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
np.mean(mape)


############################################
# Neural networks
