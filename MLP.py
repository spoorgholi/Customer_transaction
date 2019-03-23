#predicting customer transaction using MLP in keras

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
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
kaggle = sc.transform(kaggle)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import UnitNorm
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
from keras.utils import plot_model

def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'normal', input_dim = 200, 
                         kernel_constraint = UnitNorm()))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'normal', 
                         kernel_constraint = UnitNorm()))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer = 'normal', 
                         kernel_constraint = UnitNorm()))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'normal', 
                         kernel_constraint = UnitNorm()))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    plot_model(classifier, to_file='model.png')
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 250, epochs = 1000)
################### K-fold cross calidation ########################
#accuricies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 2, n_jobs = 1)
#mean = accuricies.mean() 
#variance = accuricies.std() 
#print('Base Model:\n', 'mean is %2f' %(mean), "\n", "variance is %2f" %(variance))

# Fitting the ANN to the Training set
history_callback = classifier.fit(x_train, y_train, batch_size = 250, epochs = 4000)
loss_history = history_callback.history['loss']
########################### Live Plot ########################

######################### Output ###############################
plt.figure()
plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('Loss_epoch.png')
plt.show()

y_final = classifier.predict(kaggle)
unique, counts = np.unique(y_final, return_counts=True)
y_dict = dict(zip(unique, counts))
'''
for i in range(0, len(y_final)):
    if y_final[i] <= 0.5:
        y_final[i] = 0
    else:
        y_final[i] = 1
        
'''
y_final = list(chain(*y_final))
#write the outpu
sample = pd.read_csv("sample_submission.csv")
y_final = pd.DataFrame({'ID_code': sample['ID_code'] , 'target': y_final})
y_final.to_csv('output.csv', sep = ',', index = False)