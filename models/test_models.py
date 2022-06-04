import numpy as np
import pickle
import tensorflow as tf

def load_data():
   # load training data
    x_train = pickle.load(open("../datasets/100hz/x_train.pkl", "rb"))
    y_train = pickle.load(open("../datasets/100hz/y_train.pkl", "rb"))

    # load test data
    x_test = pickle.load(open("../datasets/100hz/x_test.pkl", "rb"))
    y_test = pickle.load(open("../datasets/100hz/y_test.pkl", "rb"))

    return x_train, y_train, x_test, y_test


def test_model(model, x_train, y_train, x_test, y_test):
    
    # test model
    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Guess Accuracy = ", 1 - np.sum(y_test) / y_test.shape[0])

    y_pred = model.predict(x_test) >= .4
    confusionMatrix = tf.math.confusion_matrix(y_test, y_pred) / y_test.shape[0]
    print(confusionMatrix)
    print("Test accuracy (confusion matrix) = ", confusionMatrix[0,0] + confusionMatrix[1,1])

    y_pred = model.predict(x_test) >= .5
    confusionMatrix = tf.math.confusion_matrix(y_test, y_pred) / y_test.shape[0]
    print(confusionMatrix)
    print("Test accuracy (confusion matrix) = ", confusionMatrix[0,0] + confusionMatrix[1,1])

    y_pred = model.predict(x_test) >= .6
    confusionMatrix = tf.math.confusion_matrix(y_test, y_pred) / y_test.shape[0]
    print(confusionMatrix)
    print("Test accuracy (confusion matrix) = ", confusionMatrix[0,0] + confusionMatrix[1,1])

    y_pred = model.predict(x_test) >= .7
    confusionMatrix = tf.math.confusion_matrix(y_test, y_pred) / y_test.shape[0]
    print(confusionMatrix)
    print("Test accuracy (confusion matrix) = ", confusionMatrix[0,0] + confusionMatrix[1,1])



if __name__ == '__main__':
    # to disable run time warnings
    tf.autograph.set_verbosity(0)

    # load data
    x_train, y_train, x_test, y_test = load_data()

    lstm_model = tf.keras.models.load_model("best_lstm_model.h5")

    print("================================= LSTM =================================")
    test_model(lstm_model, x_train, y_train, x_test, y_test)

   

    




