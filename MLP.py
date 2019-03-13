#predicting customer transaction using MLP in keras

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
dataset = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

x = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
kaggle = test_set.iloc[:, 1:].values


#get some data description
print(dataset['target'].describe())
print(dataset.isna().sum(axis = 0))

'''
#plotting variables according to the targeted class
var = 'var_150'
data = pd.concat([dataset[var], dataset['target']], axis=1)
f, ax = plt.subplots()
fig = sns.boxplot(x='target', y=var, data=data)
'''

#seperating the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#scaling the input data for the MLP(output is binary and does not require scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
'''
#initilizing the ANN
classifier = Sequential()

#add the hidden layers
classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'uniform', input_dim = 200))
#add the hidden layers
classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'uniform'))
#add the hidden layers
classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'uniform'))
#adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history_callback = classifier.fit(x_train, y_train, batch_size = 25, epochs = 5)
loss_history = history_callback.history['loss']
plt.plot(np.arange(0, len(loss_history)), loss_history)
plt.show()

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc_percent = (cm[0][0] + cm[1][1]) / sum(sum(cm)) * 100
print("Accuracy on a validation set is : \n", acc_percent, "%" )
'''
############################ Cross Validation #####################################
#evaluating the ann with k-fold cross validation using the keras wrapper
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'uniform', input_dim = 200))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 250, epochs = 100)
accuricies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuricies.mean()
variance = accuricies.std()
print(mean, "\n", variance)

#predicting the first results for the competiotion 
kaggle = sc.fit_transform(kaggle)
# Fitting the ANN to the Training set
history_callback = classifier.fit(x_train, y_train, batch_size = 250, epochs = 2)
loss_history = history_callback.history['loss']
plt.plot(np.arange(0, len(loss_history)), loss_history)
plt.show()
y_final = classifier.predict(kaggle)
#y_final = (y_final > 0.5)
for i in range(0, len(y_final)):
    if y_final[i] <= 0.5:
        y_final[i] = 0
    else:
        y_final[i] = 1
y_final = y_final.astype(int)
np.savetxt("output.csv", y_final, delimiter=",")
