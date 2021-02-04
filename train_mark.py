import matplotlib.pyplot as plt
import pandas
import plaidml.keras
plaidml.keras.install_backend()
# first lets import the useful stuff
import keras
from keras.optimizers import RMSprop, Adam,SGD
from sklearn.model_selection import train_test_split
from keras.layers import *


def train():
    # -- prepare data
    # - load data
    #UNET label data
    data = np.load("/Users/markrademaker/Projects/Naturalis/datasets/160371949261_all_data.npz", allow_pickle=True)

    layers = data["layers"]

    label = data["label"]

    # train/test split
    sample_indices = range(len(label))
    indices_train, indices_test = train_test_split(sample_indices, test_size=0.4)

    # - put data in correct format for neural network
    X_train = layers_to_tensor(layers[indices_train])
    print(X_train.shape)
    where_are_NaNs = np.isnan(X_train)
    X_train[where_are_NaNs] = -1

    y_train = np.stack(label[indices_train])
    y_train = np.moveaxis(y_train,1,-1) # move label columns to last dimension as in X_train shape
    print(y_train.shape)
    y_train = normalize_biomass(y_train)
    print(y_train.shape)

    X_test = layers_to_tensor(layers[indices_test])
    where_are_NaNs = np.isnan(X_test)
    X_test[where_are_NaNs] = -1
    y_test = np.stack(label[indices_test])
    y_test = np.moveaxis(y_test,1,-1) # put label columns in last dimensions as in X_test
    y_test = normalize_biomass(y_test)
    print(y_test.shape)


    # -- create model
    print(X_train.shape[1:])
    print(X_train.shape)
    # Code for UNet Model
    model = create_UNET(X_train.shape[1:])
    predictions = model.predict(X_test)
    print(predictions)
    print("test mse - pre", np_mean_squared_error(predictions,y_test))

    model.compile(loss = "mse", optimizer = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-08),metrics=["mse"])
    batch_size = 75
    epochs = 25
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        #callbacks=[reducelr_callback],
                        shuffle=True,
                        # class_weight={0: 1.,
                        #               1: 20.,
                        #               }
                        )

    print(history.history.keys())
    plt.subplot(121)
    plt.plot(history.history["val_loss"], label="test")
    plt.plot(history.history["loss"], label="train")
    plt.xlabel('n epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['val_mean_squared_error'], label="test")
    plt.plot(history.history['mean_squared_error'], label="train")
    plt.xlabel('n epochs')
    plt.ylabel('mean squared error')
    plt.legend()
    plt.show()

    # -- evaluate
    #score = model.evaluate(X_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    #predictions = model.predict(X_test)
    #both = np.concatenate([predictions, y_test], axis=1)
    #print("test mse - post", np_mean_squared_error(predictions, y_test))
    #plt.figure()
    #plt.subplot(121)
    #plt.imshow(y_test, vmin=0, vmax=100)
    #plt.subplot(122)
    #plt.imshow(predictions, vmin=0, vmax=100)
    #plt.show()
    #pandas.DataFrame(data=both).to_csv("post_test.csv", index=None)
    #print("AUC", roc_auc_score(y_test[:, 1], predictions[:, 1]))
    #fpr, tpr, thresholds = roc_curve(y_test[:, 1], predictions[:, 1])
    #plt.plot(fpr, tpr)


def normalize_biomass(y_train):
    #return np.clip(np.linalg.norm(y_train))
    return (y_train - np.min(y_train))/(np.max(y_train)-np.min(y_train))
    #return np.clip((np.log10(y_train + 1e-10) - -10.) / 15., 0, 1)


def np_mean_squared_error(predictions, y_test):
    return np.mean(np.square(predictions - y_test))


def layers_to_tensor(layers):
    #print(np.stack([np.transpose(np.stack(layers[i]), [1, 2, 0]) for i in range(layers.shape[0])]))
    return np.stack([np.transpose(np.stack(layers[i]), [1, 2, 0]) for i in range(layers.shape[0])])


def create_UNET(input_img):
    filters= [16,32,64,128,256]
    inputTensor = Input(input_img)

    #contracting part 1
    conv1 = Conv2D(filters[0],(3,3),activation="relu",strides=1,padding='same')(inputTensor)
    conv1 = Conv2D(filters[0],(3,3),activation="relu",strides=1,padding='same')(conv1)
    #print(conv1)
    pool1 = keras.layers.MaxPooling2D((2,2))(conv1)
    #print(pool1)

    #part 2
    conv2 = Conv2D(filters[1],(3,3),activation="relu",strides=1,padding="same")(pool1)
    conv2 = Conv2D(filters[1],(3,3),activation="relu",strides=1,padding="same")(conv2)
    #print(conv2)
    pool2 = keras.layers.MaxPooling2D((2,2))(conv2)
    #print(pool2)

    #part 3
    conv3 = Conv2D(filters[2], (3, 3), activation="relu", strides=1, padding="same")(pool2)
    conv3 = Conv2D(filters[2], (3, 3), activation="relu", strides=1, padding="same")(conv3)
    #print(conv3)
    pool3 = keras.layers.MaxPooling2D((2, 2))(conv3)
    #print(pool3)

    #part 4
    conv4 = Conv2D(filters[3],(3,3),activation="relu",strides=1,padding="same")(pool3)
    conv4 = Conv2D(filters[3],(3,3),activation="relu",strides=1,padding="same")(conv4)
    #print(conv4)
    pool4 = keras.layers.MaxPooling2D((2,2))(conv4)
    #print(pool4)

    #bottleneck
    convm = Conv2D(filters[3],(3,3),activation="relu",padding="same",strides=1)(pool4)
    convm = Conv2D(filters[3],(3,3),activation="relu",padding="same",strides=1)(convm)
    #print(convm)

    #upsampling part 1
    deconv4 = Conv2DTranspose(filters[3],(3,3),strides=(2,2),padding="same")(convm)
    #print(deconv4)
    uconv4 = concatenate([deconv4,conv4])
    uconv4 = Conv2D(filters[3],(3,3),activation="relu",strides=1,padding="same")(uconv4)
    uconv4 = Conv2D(filters[3],(3,3),activation="relu",strides=1,padding="same")(uconv4)
    #print(uconv4)

    #part 2
    deconv3 = Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same")(uconv4)
    #print(deconv3)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(filters[2], (3, 3), activation="relu", strides=1, padding="same")(uconv3)
    uconv3 = Conv2D(filters[2], (3, 3), activation="relu", strides=1, padding="same")(uconv3)
    #print(uconv3)

    #part 3
    deconv2 = Conv2DTranspose(filters[1], (3, 3), strides=(2, 2),padding="same")(uconv3)
    #print(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(filters[1], (3, 3), activation="relu", strides=1, padding="same")(uconv2)
    uconv2 = Conv2D(filters[1], (3, 3), activation="relu", strides=1, padding="same")(uconv2)
    #print(deconv2)

    #part 4
    deconv1 = Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same")(uconv2)
    #print(deconv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(filters[0], (3, 3), activation="relu", strides=1, padding="same")(uconv1)
    uconv1 = Conv2D(filters[0], (3, 3), activation="relu", strides=1, padding="same")(uconv1)
    #print(uconv1)

    outputs = Conv2D(2,(1,1),padding="same")(uconv1)
    #print(outputs)#no activation function (keep raw values)
    model = keras.models.Model(inputs=[inputTensor],outputs=[outputs])
    model.summary()
    return model

if __name__ == '__main__':
    train()
