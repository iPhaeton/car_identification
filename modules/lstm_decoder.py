from keras.layers import Input, LSTM, BatchNormalization, Dense
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

class LSTM_decoder:
    def __init__(self, X, y, weights_path = None):
        self.X = X
        self.y = y
        self.y_onehot = to_categorical(y)

        input_shape = X.shape[1:]
        output_shape = self.y_onehot.shape[1]

        inputs = Input(shape=input_shape)
        X = BatchNormalization()(inputs)
        X = LSTM(1024)(X)
        outputs = Dense(output_shape, activation='softmax')(X)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.train.AdamOptimizer(0.000001),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
        )

        if weights_path != None:
            model.load_weights(weights_path)

        self.model = model

    def predict(self):
        predictions = self.model.predict(self.X, verbose = True)
        return np.argmax(predictions, axis=1)