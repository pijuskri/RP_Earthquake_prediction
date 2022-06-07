import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math, scipy
import skimage
from keras import activations
from keras.regularizers import l2
from scipy import signal
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split

from scipy.interpolate import interp1d
from skimage.measure import block_reduce

epochs = 30
batch_size = 32 # 4 # 64
neurons = 16
dropout_rate = 0.1 #0.6
layers = 1
learning_rate = 0.001
checkpoint_path = "best_lstm_model.h5"
#CUDA_VISIBLE_DEVICES=""
def load_data():
    # load training data
    x_train = pickle.load(open("../datasets/100hz/x_train.pkl", "rb"))
    y_train = pickle.load(open("../datasets/100hz/y_train.pkl", "rb"))
    
    # load validation data
    x_validation = pickle.load(open("../datasets/100hz/x_validation.pkl", "rb"))
    y_validation = pickle.load(open("../datasets/100hz/y_validation.pkl", "rb"))

    # Combine train and val data
    x_train = np.concatenate([x_train, x_validation])
    y_train = np.concatenate([y_train, y_validation])

    # load test data
    x_test = pickle.load(open("../datasets/100hz/x_test.pkl", "rb"))
    y_test = pickle.load(open("../datasets/100hz/y_test.pkl", "rb"))

    print(np.min(x_test))

    return x_train, y_train, x_test, y_test

def load_data_from_df(location):
    df = pd.read_pickle(location)
    labels = df['label']
    print(len(labels))
    df = df.drop(columns=['label'])
    df = df.applymap(lambda x: np.array(x))
    arr = np.array(df.to_numpy().tolist())
    print(arr.shape)
    #np.random.shuffle(arr)
    x_train, x_test, y_train, y_test = train_test_split(arr, labels, test_size=0.20, random_state=42)
    return x_train, y_train, x_test, y_test

def evaluate_metrics(model, x_test, y_test):
    y_pred = model.predict(x_test).flatten()
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    test_errors = np.array([ abs(y_test - y_pred) for index, tup in enumerate(zip(y_test, y_pred))])
    error = np.mean(test_errors)
    print("average error", error)
    for index, tup in enumerate(zip(y_test, y_pred)):
        actual = tup[0]
        pred = tup[1]

        if pred >= 0.5:
            if actual == 1:
                TP += 1
            else:
                FP += 1
        else:
            if actual == 0:
                TN += 1
            else:
                FN += 1


    print("TP:", TP)
    print("TN:", TN)
    print("FN:", FN)
    print("FP:", FP)
    if (TP + FN) > 0: print("Sn:", (TP / (TP + FN)))
    if (TN + FP) > 0: print("Sp:", (TN / (TN + FP)))
    if (TN + FN) > 0: print("p0:", (TN / (TN + FN)))
    if (TP + FP) > 0: print("p1:", (TP / (TP + FP)))
    print("accuracy:", ((TP+TN) / (TP + TN + FP + FN)))

def plot_accuracy(history, metric="accuracy"):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model Accuracy - LSTM")
    plt.ylabel("Accuracy", fontsize="large")
    plt.xlabel("Total Number of Epochs", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

def plot_loss(history, metric="loss"):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model Loss - LSTM")
    plt.ylabel("Loss", fontsize="large")
    plt.xlabel("Total Number of Epochs", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()


@tf.autograph.experimental.do_not_convert
def lstm_model(x_train, y_train, x_test, y_test, input_shape):
    print(input_shape)

    model = tf.keras.models.Sequential()

    #model.add(tf.keras.layers.Dropout(dropout_rate, input_shape=input_shape))  # avoids overfitting

    for _ in range(0,layers-1):
        model.add(tf.keras.layers.LSTM(neurons, return_sequences=True, kernel_initializer='random_normal'))
        model.add(tf.keras.layers.Dropout(dropout_rate)) # avoids overfitting
    model.add(tf.keras.layers.LSTM(neurons, activation="tanh", input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate, input_shape=input_shape))

    # output = 1, binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        #learning_rate=0.0001
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #, epsilon=1e-05
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss="binary_crossentropy", # used for binary classification
        #metrics=['binary_accuracy'],
        metrics=['accuracy'],
    )

    # train model

    callbacks = [
        # save best model
        #tf.keras.callbacks.ModelCheckpoint(
        #    checkpoint_path, save_best_only=True, monitor="val_accuracy", verbose=1 #monitor="val_loss"
        #),
        #tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    ]

    #model = load_model(checkpoint_path)

    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        #validation_data=(x_test, y_test),
        validation_split = 0.1,
        verbose=1, # how to show progress bar
    )

    return model, history

def lower_hz(data):
    #should average instead
    #R = 3
    #return data.reshape(-1, R).mean(axis=1)
    #return signal.resample(data, data/)
    #return np.array_split(data, len(data)/2, 1)
    #return data[:,0::3]

    #R = 2
    #pad_size = math.ceil(float(b.size) / R) * R - b.size
    #b_padded = np.append(b, np.zeros(pad_size) * np.NaN)
    #np.nanmean(b_padded.reshape(-1, R), axis=1)
    #seems to make model very noisy
    def downsample(array, npts):
        interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
        downsampled = interpolated(np.linspace(0, len(array), npts))
        return downsampled

    #print(downsample([0.1,0.2,0.3,0.4], 3))
    out = []
    for i in range(len(data)):
        #out.append(downsample(data[i], 3000))
        out.append(block_reduce(data[i], block_size=(5, 1), func=np.mean))
    out = np.stack(out, axis=0)
    #print(out.shape)
    return out

def run_model(data_location="../datasets/sets/dataset_shallow.pkl"):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            pass
            #tf.config.set_logical_device_configuration(gpus[0],
            #    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            pass

    # to disable run time warnings
    tf.autograph.set_verbosity(0)

    # load data
    #x_train, y_train, x_test, y_test = load_data()
    x_train, y_train, x_test, y_test = load_data_from_df(data_location)

    # x_train = x_train.transpose((0,2,1))
    # x_test = x_test.transpose((0,2,1))
    # print(x_train.shape)
    # x_train = lower_hz(x_train)
    # x_test = lower_hz(x_test)
    # x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    print(x_train.shape)

    # create model - input_shape = (61, 58)
    model, history = lstm_model(x_train, y_train, x_test, y_test, input_shape=(x_train.shape[1:]))
    # model = load_model(checkpoint_path)

    # test model
    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # evaluation metrics
    evaluate_metrics(model, x_test, y_test)

    # plot metrics
    plot_loss(history)
    plot_accuracy(history)

if __name__ == '__main__':
    run_model()








