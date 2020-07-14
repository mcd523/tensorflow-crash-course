import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


class HousingPrices:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs

    def run(self):
        (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

        # each piece of data has 13 fields with differing values, the following code normalizes the data around the mean
        # and standard deviation of the specific field
        mean = train_data.mean(axis=0)
        train_data -= mean
        std = train_data.std(axis=0)
        train_data /= std
        test_data -= mean
        test_data /= mean

        all_mae_histories = self.handle_k_folding(train_data, train_targets)
        average_mae_history = [
            np.mean([x[i] for x in all_mae_histories]) for i in range(self.num_epochs)
        ]
        self.handle_mae_plot(average_mae_history, smooth=True)

    # builds and returns compiled model
    def build_model(self, shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(
            optimizer='rmsprop',
            loss='mse',  # mean-squared error
            metrics=['mae']  # mean-absolute error, in this case a value of 1 means error is $1,000
        )
        return model

    def handle_k_folding(self, train_data, train_targets):
        k = 4  # number of folds
        num_val_samples = len(train_data) // k
        all_mae_histories = []
        for i in range(k):
            print('processsing fold #', i)
            val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                (train_data[:i * num_val_samples],
                 train_data[(i + 1) * num_val_samples:]),
                axis=0
            )
            partial_train_targets = np.concatenate(
                (train_targets[:i * num_val_samples],
                 train_targets[(i + 1) * num_val_samples:]),
                axis=0
            )
            model = self.build_model(train_data.shape[1])
            history = model.fit(
                partial_train_data,
                partial_train_targets,
                validation_data=(val_data, val_targets),
                epochs=self.num_epochs,
                batch_size=1,
                verbose=0
            )
            val_mae = history.history['val_mae']
            all_mae_histories.append(val_mae)
        return all_mae_histories

    def handle_mae_plot(self, average_mae_history, factor=0.9, smooth=False):
        smooth_mae_history = average_mae_history
        if smooth:
            print('We are smooth')
            smoothed_points = []
            for point in smooth_mae_history[10:]:  # remove first 10 points from graph to help see actual trend after
                if smoothed_points:                # initial MAE dropoff over the first few epochs
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            smooth_mae_history = smoothed_points
        plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
        plt.xlabel('Epochs')
        plt.ylabel('Validation MAE')
        plt.show()
