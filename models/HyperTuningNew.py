import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import keras_tuner as kt
from keras.regularizers import l2

from lstm_model import plot_loss, plot_accuracy

batch_size = 32
input_shape = (301, 56)
epochs = 50

def load_data():
    # load training data
    x_train = pickle.load(open("../datasets/100hz/x_train.pkl", "rb"))
    y_train = pickle.load(open("../datasets/100hz/y_train.pkl", "rb"))

    # load validation data
    x_validation = pickle.load(open("../datasets/100hz/x_validation.pkl", "rb"))
    y_validation = pickle.load(open("../datasets/100hz/y_validation.pkl", "rb"))

    # load test data
    x_test = pickle.load(open("../datasets/100hz/x_test.pkl", "rb"))
    y_test = pickle.load(open("../datasets/100hz/y_test.pkl", "rb"))


    return x_train, y_train, x_validation, y_validation, x_test, y_test

def lstm_model(hp):

    model = tf.keras.models.Sequential()

    #recurrent_regularizer = l2(0.01), kernel_regularizer = l2(0.01)
    #model.add(tf.keras.layers.LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_regularizer=l2(0.01)))
    #model.add(tf.keras.layers.Dropout(dropout_rate))
    #hp_neurons = hp.Int('neurons', min_value=32, max_value=128, step=32)
    #hp_neurons = hp.Choice('neurons', [32, 64, 128, 256, 512])
    hp_neurons = 64
    #hp_dropout = hp.Float('dropout', min_value=0.05, max_value=0.3, step=0.05)
    hp_dropout = 0.1
    #hp_regular = hp.Float('regularization', min_value=0, max_value=0.01, step=0.01)
    hp_lr = hp.Choice('learning rate', [0.00001, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.002])
    #recurrent_regularizer = l2(hp_regular), kernel_regularizer = l2(hp_regular))

    model.add(tf.keras.layers.Dropout(hp_dropout, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(hp_neurons, return_sequences=True, kernel_initializer='random_normal',input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(hp_dropout, input_shape=input_shape))  # avoids overfitting
    model.add(tf.keras.layers.LSTM(hp_neurons, input_shape=input_shape))



    # output = 1, binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        #optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="binary_crossentropy", # used for binary classification
        metrics=['accuracy'],
    )

    return model

if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3096)])
        except RuntimeError as e: pass

    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data()

    tuner = kt.Hyperband(lstm_model,
                         objective='val_accuracy', #val_accuracy, val_loss
                         max_epochs=epochs/2,
                         factor=3,
                         directory='tuning_dir',
                         project_name='intro_to_kt')
     # train model
    callbacks = [
        # save best model
        #tf.keras.callbacks.ModelCheckpoint(
        #    "best_lstm_model.h5", save_best_only=True, monitor="val_accuracy"
        #),
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    ]

    tuner.search(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        #callbacks=callbacks,
        validation_data=(x_validation, y_validation),
        # validation_split = 0.0,
        verbose=1,  # how to show progress bar
    )
    print("tuning done")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
        # validation_split = 0.0,
        verbose=1,  # how to show progress bar
    )

    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # plot metrics
    plot_loss(history)
    plot_accuracy(history)

