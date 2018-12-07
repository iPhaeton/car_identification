from keras.layers import Input, Dropout, Dense, Bidirectional, TimeDistributed, LSTM, Activation, BatchNormalization
from keras.models import Model
import numpy as np
import os
import fnmatch
import tensorflow as tf

class LSTMEncoder:
    def __init__(self, regularizer, source_path, weights_path = None, rs = 0, lr = 0.00001):
        self.source_path = source_path
        self.get_metadata(source_path)

        inputs = Input(shape=self.input_shape)
        X = Dropout(0.2)(inputs)
        X = Dense(4096, activation='relu', kernel_regularizer=regularizer(rs))(X)
        X = Dropout(0.2)(X)
        X = Dense(2048, activation='relu', kernel_regularizer=regularizer(rs))(X)
        X = Dense(2048, activation='relu', kernel_regularizer=regularizer(rs))(X)
        X = Bidirectional(LSTM(1024, return_sequences=True, kernel_regularizer=regularizer(rs)))(X)
        outputs = TimeDistributed(Dense(self.num_classes, activation='softmax'))(X)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.train.AdamOptimizer(lr),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
        )
        
        if weights_path != None:
            model.load_weights(weights_path)
        
        self.model = model

    def get_metadata(self, path):
        features = np.load(path + 'features_train_0.npy')
        input_shape = features.shape[1:]
        classes = np.load(path + 'classes_train_0.npy')
        num_classes = classes.shape[2]
        steps_per_epoch = len(fnmatch.filter(os.listdir(path),'*features_train_*'))
        validation_steps = len(fnmatch.filter(os.listdir(path),'*features_dev_*'))

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def get_steps(self, mode):
        steps = None
        if mode == 'dev':
            steps = self.validation_steps
        else:
            steps = self.steps_per_epoch
        return steps

    def generator(self, path, mode, num_batches, random=True):
        counter = 0
        indices = None
        if random == True:
            indices = np.random.permutation(list(range(num_batches)))
        else:
            indices = list(range(num_batches))
        
        while True:
            if counter >= num_batches:
                counter = 0
                if random == True:
                    indices = np.random.permutation(list(range(num_batches)))

            features = np.load(path + 'features_' + mode + '_' + str(indices[counter]) + '.npy')
            classes = np.load(path + 'classes_' + mode + '_' + str(indices[counter]) + '.npy')
            
            counter += 1
            yield features, classes

    def encode(self, mode, save_path=None):
        encoder = Model(inputs=self.model.input, outputs=self.model.layers[6].output)

        steps = self.get_steps(mode)

        encoded_features = encoder.predict_generator(
            self.generator(self.source_path, mode, steps, random=False),
            steps=steps,
            verbose=True,
        )

        if save_path != None:
            np.save(save_path, encoded_features)

        return encoded_features

    def train(self, epochs, callbacks):
        self.model.fit_generator(
            self.generator(self.source_path, 'train', self.steps_per_epoch),
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=self.generator(self.source_path, 'dev', self.validation_steps),
            validation_steps=self.validation_steps,
            callbacks=callbacks,
        )

    def predict(self, mode):
        steps = self.get_steps(mode)
        predictions = self.model.predict_generator(
            self.generator(self.source_path, mode, steps, random=False),
            steps=steps,
            verbose=True,
        )
        return np.argmax(predictions, axis=2)


class LSTMEncoderWithBatchnorm(LSTMEncoder):
    def __init__(self, regularizer, source_path, weights_path = None, rs = 0, lr = 0.00001):
        self.source_path = source_path
        self.get_metadata(source_path)

        inputs = Input(shape=self.input_shape)
        X = Dropout(0.2)(inputs)
        X = Dense(4096, kernel_regularizer=regularizer(rs))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(2048, kernel_regularizer=regularizer(rs))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dense(2048, kernel_regularizer=regularizer(rs))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Bidirectional(LSTM(1024, return_sequences=True, kernel_regularizer=regularizer(rs)))(X)
        outputs = TimeDistributed(Dense(self.num_classes, activation='softmax'))(X)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.train.AdamOptimizer(lr),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
        )
        
        if weights_path != None:
            model.load_weights(weights_path)
        
        self.model = model
