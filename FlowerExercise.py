import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FlowerExercise:
    train_image_generator = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size

    def run(self):
        base_dir = os.path.join('/Users/mcdahl/Downloads', 'flower_photos')

        classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

        # for cl in classes:
        #     img_path = os.path.join(base_dir, cl)
        #     images = glob.glob(img_path + '/*.jpg')
        #     print("{}: {} Images".format(cl, len(images)))
        #     train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]
        #
        #     for t in train:
        #         if not os.path.exists(os.path.join(base_dir, 'train', cl)):
        #             os.makedirs(os.path.join(base_dir, 'train', cl))
        #         shutil.move(t, os.path.join(base_dir, 'train', cl))
        #
        #     for v in val:
        #         if not os.path.exists(os.path.join(base_dir, 'val', cl)):
        #             os.makedirs(os.path.join(base_dir, 'val', cl))
        #         shutil.move(v, os.path.join(base_dir, 'val', cl))

        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')

        total_train = 0
        total_val = 0
        for c in classes:
            total_train += len(os.listdir(os.path.join(train_dir, c)))
            total_val += len(os.listdir(os.path.join(val_dir, c)))

        print("Training size: {}".format(total_train))
        print("Validation size: {}".format(total_val))
        train_data_gen = self.train_image_generator.flow_from_directory(
            batch_size=self.batch_size,
            directory=train_dir,
            shuffle=True,
            target_size=(self.image_size, self.image_size),
            class_mode='binary'
        )
        val_data_gen = self.validation_image_generator.flow_from_directory(
            batch_size=self.batch_size,
            directory=val_dir,
            shuffle=True,
            target_size=(self.image_size, self.image_size),
            class_mode='binary'
        )

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.image_size, self.image_size, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        EPOCHS = 5
        history = model.fit(
            train_data_gen,
            steps_per_epoch=int(np.ceil(train_data_gen.n / float(self.batch_size))),
            epochs=EPOCHS,
            validation_data=val_data_gen,
            validation_steps=int(np.ceil(val_data_gen.n / float(self.batch_size)))
        )

        self.handlePlot(history, EPOCHS)

    def handlePlot(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

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
