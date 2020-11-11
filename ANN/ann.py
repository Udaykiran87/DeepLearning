# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", # Just a name, any thing is ok (but to be more specific here i have indicated column name as country))
                         OneHotEncoder(), # The transformer class
                         [1])], # The column(s) to be applied on.
                         remainder = 'passthrough') # donot apply anything to the remaining columns
X = ct.fit_transform(X)

labelencoder_x_2=LabelEncoder()
X[:,4]=labelencoder_x_2.fit_transform(X[:,4]) # Encoding gender column

# to avoid dummy variable trap (remove first column as we can identify a country if we know 2  columns out of 3 columns)
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# Part 2- Now let's make the ANN!

# Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding input layer and the first hidden layer with dropout
classifier.add(Dense(6, kernel_initializer = "uniform", activation="relu",input_shape=(11,)))
classifier.add(Dropout(p = 0.1))

# Adding second hidden layer
classifier.add(Dense(6, kernel_initializer = "uniform", activation="relu"))
classifier.add(Dropout(p = 0.1))

# Adding output layer
# Incase dependent variables are more than 1, then the the number of output dim (first parameter will change accordingly 
# and activation function becomes "softmax" which also a sigmoid function but for more than 1 dependent variables)
classifier.add(Dense(1, kernel_initializer = "uniform", activation="sigmoid")) 

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Create your classifier here

# Part 3 - making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc_X.fit_transform(np.array([[0.0 , 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# making the condusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer = "uniform", activation="relu",input_shape=(11,)))
    classifier.add(Dense(6, kernel_initializer = "uniform", activation="relu"))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation="sigmoid")) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer = "uniform", activation="relu",input_shape=(11,)))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(6, kernel_initializer = "uniform", activation="relu"))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation="sigmoid")) 
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
