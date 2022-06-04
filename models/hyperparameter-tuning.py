import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

epochs = 20

batch_size = [64]
dropout_rate = [0.3, 0.4, 0.5, 0.6]
neurons = [60, 80]
learning_rate = [0.1, 0.01, 0.001]

hyper_params = (batch_size, dropout_rate, neurons, learning_rate)
hyper_params_grid = tuple(itertools.product(*hyper_params, repeat=1))


def load_data():
    # load training data
    x_train = pickle.load(open("x_train.pkl", "rb"))
    y_train = pickle.load(open("y_train.pkl", "rb"))

    # load validation data
    x_validation = pickle.load(open("x_validation.pkl", "rb"))
    y_validation = pickle.load(open("y_validation.pkl", "rb"))

    # load test data
    x_test = pickle.load(open("x_test.pkl", "rb"))
    y_test = pickle.load(open("y_test.pkl", "rb"))


    return x_train, y_train, x_validation, y_validation, x_test, y_test


if __name__ == '__main__':

    # load data
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data()


    # GRID SEARCH
    best_params = []
    max_val_accuracy = -1
    curr_val_accuracy = 0

    for b, d, n, lr in hyper_params_grid:

        print('batch_size:', b, 
            'dropout:', d, 
            'neurons:', n,
            'lr:', lr
            )

        model = tf.keras.models.Sequential()
        
        model.add(tf.keras.layers.LSTM(n, input_shape=(61, 58)))
        model.add(tf.keras.layers.Dropout(d))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 
        


        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy", # used for binary classification
            metrics=['accuracy'], 
        )

        # fit model
        history = model.fit(
            x_train,
            y_train,
            batch_size=b,
            epochs=epochs,
            validation_data=(x_validation, y_validation),
            verbose=1, # how to show progress bar
        )

        curr_val_accuracy = history.history['val_loss'][-1]

        if curr_val_accuracy > max_val_accuracy:
          max_val_accuracy = curr_val_accuracy
          best_params = [b, d, n, lr]
          print('best_params', best_params)


    print("===================== DONE ===========================")
    print(best_params)







