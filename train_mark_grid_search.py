#import plaidml.keras
#plaidml.keras.install_backend()
import pandas
# first lets import the useful stuff
import tensorflow as tf
from tensorflow import keras
#import other stuff
#from keras import backend as K

import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD,Adagrad,Adadelta
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def normalize_biomass(y_train):
    return (y_train - np.min(y_train))/(np.max(y_train)-np.min(y_train))

def layers_to_tensor(layers):
    return np.stack([np.transpose(np.stack(layers[i]), [1, 2, 0]) for i in range(layers.shape[0])])

def create_model(optimizer='adam'):
    model = Sequential()
    layer = Conv2D(32,  # num filters
                   (3, 3),  # kernel size
                   activation='relu', input_shape=(10,10,32),
                   kernel_regularizer=regularizers.l1(0.01),
                   #activity_regularizer=regularizers.l1(0.002),
                   padding="valid"
                   )
    model.add(layer)

    layer = Conv2D(16,  # num filters
                   (3, 3),  # kernel size
                   activation='relu',
                   kernel_regularizer=regularizers.l1(0.01),
                   #activity_regularizer=regularizers.l1(0.002),
                   padding="valid"
                   )
    model.add(layer)

    layer = Conv2D(8,  # num filters
                   (3, 3),  # kernel size
                   activation='relu',
                   kernel_regularizer=regularizers.l1(0.01),
                   # activity_regularizer=regularizers.l1(0.002),
                   padding="valid"
                   )#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 )
    model.add(layer)


    layer = Conv2D(4,  # num filters
                   (3, 3),  # kernel size
                   activation='relu',
                   kernel_regularizer=regularizers.l1(0.01),
                   # activity_regularizer=regularizers.l1(0.002),
                   padding="valid"
                   )
    model.add(layer)

    model.add(Flatten())
    #model.add(Dense(num_classes))#, activation="relu"))
    model.add(Dense(2))
    #model.summary()
    model.compile(loss='mse',optimizer=optimizer,metrics=['mse'])
    return model


def train():
    # -- prepare data
    # - load data
    data = np.load("/Users/markrademaker/Projects/Naturalis/datasets/159317562268_1000_random.npz", allow_pickle=True)

    #Split into input (X) and output (Y)
    X=data['layers']
    Y=data['label']

    sample_indices = range(len(Y))
    indices_train,indices_test = train_test_split(sample_indices,test_size=0.4)

    #Split into  and train and test
    X_train = layers_to_tensor(X[indices_train])
    where_are_NaNs = np.isnan(X_train)
    X_train[where_are_NaNs] = -1

    y_train = np.stack(Y[indices_train])
    y_train = normalize_biomass(y_train)

    X_test = layers_to_tensor(X[indices_test])
    where_are_NaNs = np.isnan(X_test)
    X_test[where_are_NaNs] = -1

    y_test = np.stack(Y[indices_test])
    y_test = normalize_biomass(y_test)

    num_classes=y_test.shape[1]
    #print(X_train.shape[1:])

    #create model
    #model = create_model(X_train.shape[1:],num_classes)
    model = KerasRegressor(build_fn = create_model,verbose=1)

    #define grid search parameters
    optimizer = ['SGD', 'RMSPROP', 'Adagrad', 'Adadelta', 'Adam']
    batch_size = [30,50, 75]
    epochs = [50, 100, 150]

    param_grid= dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
    grid = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,cv=3)
    grid_result = grid.fit(X_train,y_train)

    #summarize results
    print("Best : %f using %s" % (grid_result.best_score_,grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    train()
