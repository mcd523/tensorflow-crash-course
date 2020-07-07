import math
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


class ClothesExample:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def run(self):
        print('Creating model')
        dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        print('Loaded fashion data')

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        num_train_examples = metadata.splits['train'].num_examples
        num_test_examples = metadata.splits['test'].num_examples
        print("Number of training examples: {}".format(num_train_examples))
        print("Number of test examples:     {}".format(num_test_examples))

        # The map function applies the normalize function to each element in the train
        # and test datasets
        train_dataset = train_dataset.map(self.normalize)
        test_dataset = test_dataset.map(self.normalize)

        # The first time you use the dataset, the images will be loaded from disk
        # Caching will keep them in memory, making training faster
        train_dataset = train_dataset.cache()
        test_dataset = test_dataset.cache()

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(self.batch_size)
        test_dataset = test_dataset.cache().batch(self.batch_size)

        model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(num_train_examples / self.batch_size))

        test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
        print('Accuracy on test dataset:', test_accuracy)

        for test_images, test_labels in test_dataset.take(1):
            test_images = test_images.numpy()
            test_labels = test_labels.numpy()
            predictions = model.predict(test_images)
            print("Prediction: {}".format(class_names[np.argmax(predictions[0])]))
            print("Label: {}".format(class_names[test_labels[0]]))

            plt.figure(figsize=(6, 3))
            plt.imshow(test_images[0].squeeze(), cmap=plt.cm.binary)
            plt.colorbar()
            plt.grid(False)
            plt.show()

    def normalize(self, images, labels):
        images = tf.cast(images, tf.float32)
        images /= 255
        return images, labels

    def plot_image(self, i, predictions_array, true_labels, images, class_names):
        predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        print(predictions_array)
        print(img)
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
