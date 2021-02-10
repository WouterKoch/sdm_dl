import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import numpy as np


# class My_Custom_Generator(keras.utils.Sequence):
#
#     def __init__(self, image_filenames, labels, batch_size):
#         self.image_filenames = image_filenames
#         self.labels = labels
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
#
#     def __getitem__(self, idx):
#         batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
#
#         return np.array([
#             resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
#             for file_name in batch_x]) / 255.0, np.array(batch_y)


if __name__ == '__main__':
    df = pd.read_csv('Plantae/59.5.csv', index_col=['decimalLatitude', 'decimalLongitude'])
    # targets = -len(df.columns)
    # df = df.merge(pd.read_csv('Bombus/59.5.csv'), how='outer')
    # targets += len(df.columns)

    target = pd.read_csv('Bombus/59.5.csv', index_col=['decimalLatitude', 'decimalLongitude'])

    df = pd.concat([df, target], axis=1).fillna(0).astype('int32').reset_index()
    targets = len(target.columns)
    target = None

    df['decimalLatitude'] = (df['decimalLatitude'] - 45) / 45
    df['decimalLongitude'] = (df['decimalLongitude']) / 33

    X, y = df.iloc[:, :].to_numpy().astype('float32'), df.iloc[:, -targets:].to_numpy().astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    x_features = X_train.shape[1]
    y_features = y_train.shape[1]

    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(x_features,)))
    # model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(y_features, activation='relu'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    # fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)
    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)

    model.save('trained_model_binary.h5')

    preds = model.predict(x=X_test)

    for p in preds:
        print(p)


    '''
    Get degrees of plants
    Get degrees of target
    
    Combine based on lat/lon
    Fill na with 0
    
    Make lat/lon regular columns
    
    Split target and rest
    
    train
    
    
    
    '''
