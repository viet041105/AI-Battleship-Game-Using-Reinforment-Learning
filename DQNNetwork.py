import tensorflow as tf
import numpy as np
import os

class DQNNetwork:
    def __init__(self, board_width, board_height, num_ships, model_file=None):
        tf.keras.backend.clear_session()
        self.board_width = board_width
        self.board_height = board_height
        self.num_ships = num_ships
        self.num_input_dimension = self.num_ships + 1
        self.board_size = self.board_width * self.board_height
        self.epsilon = 1.0 
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.995 
        input_shape = (board_height * board_width * (self.num_input_dimension),)
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Reshape((board_height, board_width, self.num_input_dimension)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(self.num_input_dimension, (1, 1), padding="same", activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.board_size, activation='softmax')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy'
        )
        if model_file is not None and os.path.exists(model_file):
            print('Attempting to load model', model_file)
            self.restoreModel(model_file)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def getBoardProbabilities(self, input_dimensions):
        return self.model.predict(input_dimensions)

    def runTrainStep(self, input_dimensions, labels, learning_rate):
        labels = np.array(labels).flatten()
        history = self.model.fit(
            input_dimensions,
            labels,
            epochs=1,
            verbose=0
        )
        return history.history['loss'][0]

    def saveModel(self, model_path):
        self.model.save(model_path + '.keras', save_format='keras')

    def restoreModel(self, model_path):
        keras_path = model_path + '.keras'
        if os.path.exists(keras_path):
            self.model = tf.keras.models.load_model(keras_path)
            print(f'Model loaded from {keras_path}')
        else:
            print(f'No model found at {keras_path}. Starting with a new model.')