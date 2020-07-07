import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt


class AnimalsExample:
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    def __init__(self, batch_size, image_shape):
        self.batch_size = batch_size
        self.image_shape = image_shape

    def run(self):
        base_dir = os.path.join("/Users/mcdahl/Downloads", 'cats_and_dogs_filtered')
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

        train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
        train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

        num_cats_tr = len(os.listdir(train_cats_dir))
        num_dogs_tr = len(os.listdir(train_dogs_dir))

        num_cats_val = len(os.listdir(validation_cats_dir))
        num_dogs_val = len(os.listdir(validation_dogs_dir))

        total_train = num_cats_tr + num_dogs_tr
        total_val = num_cats_val + num_dogs_val

        print('total training cat images:', num_cats_tr)
        print('total training dog images:', num_dogs_tr)

        print('total validation cat images:', num_cats_val)
        print('total validation dog images:', num_dogs_val)
        print("--")
        print("Total training images:", total_train)
        print("Total validation images:", total_val)

        train_data_gen = self.train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                   directory=train_dir,
                                                                   shuffle=True,
                                                                   target_size=(self.image_shape, self.image_shape),  # (150,150)
                                                                   class_mode='binary')
        val_data_gen = self.validation_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                      directory=validation_dir,
                                                                      shuffle=False,
                                                                      target_size=(self.image_shape, self.image_shape),  # (150,150)
                                                                      class_mode='binary')

        sample_training_images, _ = next(train_data_gen)
        self.plot_images(sample_training_images[:5])  # Plot images 0-4
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        EPOCHS = 25
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=int(np.ceil(total_train / float(self.batch_size))),
            epochs=EPOCHS,
            validation_data=val_data_gen,
            validation_steps=int(np.ceil(total_val / float(self.batch_size)))
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('./foo.png')
        plt.show()

    # This function plots images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plot_images(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()