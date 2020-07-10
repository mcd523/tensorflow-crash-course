import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt


# Single-label, multi-class problem (ie. each newswire will belong to 1 of 46 classes, the network will match them)
class Reuters:
    def __init__(self):
        return

    def run(self):
        # get data from imdb datastore
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
        # print review encoded as numbers (words in review 1-1 map with integer)
        print(train_data[0])
        # print if review is positive or negative (1 == positive, 0 == negative)
        print(train_labels[0])
        # get mapping of word to integers
        word_index = reuters.get_word_index()
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

        # One-hot encoding could also be achieved by calling tf.keras.utils.np_utils.to_categorical(labels)
        one_hot_train_labels = self.to_one_hot(train_labels)
        one_hot_test_labels = self.to_one_hot(test_labels)

        # create model of 2 layers containing 64 nodes and an output layer that returns probability distribution that
        # the input is each of the 46 labels. 64 node layers were used instead of 16 in the imdb problem because 16
        # nodes may be too few to try and differentiate 46 different output classes
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(46, activation='softmax')
        ])

        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy', # can't use binary_crossentropy because we have 46 possible outputs as opposed to 2
            metrics=['accuracy']
        )

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]
        y_val = one_hot_train_labels[:1000]
        partial_y_train = one_hot_train_labels[1000:]

        history = model.fit(
            partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val)
        )

        # plt.clf()  # clears plot
        self.handle_loss_plot(history)
        plt.clf()  # clears plot
        self.handle_accuracy_plot(history)



    # turn integer array of review into a vector with correct dimensions for input (x, 10000) <- x reviews of 10k length
    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    # creates matrix with dimensions num_labels x num_dimensions where each label is represented by an array of
    # num_dimensions - 1 zeros and one 1
    def to_one_hot(self, labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
            results[i, label] = 1.
        return results

    def handle_loss_plot(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def handle_accuracy_plot(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
