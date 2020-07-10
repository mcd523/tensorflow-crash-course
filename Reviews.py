import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb


# Single-class problem
class Reviews:
    def __init__(self):
        return

    def run(self):
        # get data from imdb datastore
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        # print review encoded as numbers (words in review 1-1 map with integer)
        print(train_data[0])
        # print if review is positive or negative (1 == positive, 0 == negative)
        print(train_labels[0])
        # get mapping of word to integers
        word_index = imdb.get_word_index()
        # handling to get review content from word_index
        reverse_word_index = dict(
            [(value, key) for (key, value) in word_index.items()]
        )
        decoded_review = ' '.join(
            [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
        )
        print(decoded_review)

        x_train = self.vectorize_sequences(train_data)
        x_test = self.vectorize_sequences(test_data)

        # turn labels into numpy array of floats
        y_train = np.asarray(train_labels).astype('float32')
        y_test = np.asarray(test_labels).astype('float32')

        # create model of 2 layers containing 16 nodes and an output layer that returns probability of correct guess
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # compile model with desired optimizer, loss function, and relevant metrics
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # split data into training set and validation set (10k each)
        x_val = x_train[:10000]
        partial_x_train = x_train[10000:]
        y_val = y_train[:10000]
        partial_y_train = y_train[10000:]

        print(partial_x_train.shape)
        print(partial_y_train.shape)

        # run training data through model to modify weights in each node, then run validation set through model to determine
        # accuracy of non-training data
        history = model.fit(
            partial_x_train,
            partial_y_train,
            epochs=20, # determines how many times we run the training data through the model, be aware of overtraining if this value is too high, you'll start to see the accuracy for the training data increase but accuracy for the validation data will plateau or decrease, probably because the model is focusing too much on the specific data from the training set
            batch_size=512,
            validation_data=(x_val, y_val)
        )

        # run model on test data to see how effective it is for data not previously seen
        results = model.evaluate(x_test, y_test)
        print(results)
        prediction = model.predict(x_test)
        print(prediction)


    # turn integer array of review into a vector with correct dimensions for model input (25000, 10000) <- 25k reviews of 10k length eachh
    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

