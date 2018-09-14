#!/usr/bin/env python3
"""Train a Bidirectional LSTM."""

import argparse
import csv
import json
import logging
import os
import sys
import time

from keras import backend as K
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, CuDNNLSTM, Bidirectional
from keras.utils import multi_gpu_model
import numpy as np
import pandas as pd
import tensorflow as tf

BATCH_SIZE = 32
DATA_FEATURES = [
    'smart_5_raw',
    'smart_187_raw',
    'smart_188_raw',
    'smart_197_raw',
    'smart_198_raw',
]
DATA_LABELS = [
    'failure'
]
FILE_REPORT = 'report.json'
FILE_EPOCH_STATS = 'epoch_stats.log'

LOG = logging.getLogger(__name__)


class EpochStatsLogger(Callback):
    """TODO."""

    def on_train_begin(self, logs={}):
        """TODO."""
        filename = os.path.basename(sys.argv[0])[:-3]
        backend = K.backend()
        self.f = open(FILE_EPOCH_STATS, 'w')
        self.log_writer = csv.writer(self.f)
        self.log_writer.writerow([
            'epoch', 'elapsed', 'loss', 'acc', 'val_loss', 'val_acc'
        ])

    def on_train_end(self, logs={}):
        """TODO."""
        self.f.close()

    def on_epoch_begin(self, epoch, logs={}):
        """TODO."""
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        """TODO."""
        self.log_writer.writerow([
            epoch, time.time() - self.start_time, logs.get('loss'),
            logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')
        ])


def parse_args():
    """TODO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', required=True, type=int,
        help='Number of iterations (epochs) over the corpus.'
    )
    parser.add_argument(
        '--data_file', required=True, type=str,
        help='Filename of data to read.'
    )
    return parser.parse_args()


def load_data(filename):
    """Load the raw drive data for features and label and 80/20 split."""
    LOG.info('loading data')
    raw_data = pd.read_csv(
        filename,
        usecols=DATA_FEATURES + DATA_LABELS
    ).dropna()

    LOG.info('splitting data 80/20')
    train = raw_data.sample(frac=0.8, random_state=200)
    test = raw_data.drop(train.index)

    train_x = train[DATA_FEATURES]
    train_y = np.array(train[DATA_LABELS])
    test_x = test[DATA_FEATURES]
    test_y = np.array(test[DATA_LABELS])

    LOG.info('Train Features Shape: %s' % str(train_x.shape))
    LOG.info('Test Features Shape: %s' % str(test_x.shape))

    return train_x, train_y, test_x, test_y


def get_model():
    """TODO."""
    LOG.info('building model')
    model = Sequential()
    model.add(Embedding(len(DATA_FEATURES), 128))
    model.add(Bidirectional(CuDNNLSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # TODO: Increase the batch size * # GPUs
    # https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012

    # TODO: https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
    # Consider one of those?
    if len(K.tensorflow_backend._get_available_gpus()) > 1:
        LOG.info("Using Multi-GPU Model")
        model = multi_gpu_model(
            model,
            gpus=len(K.tensorflow_backend._get_available_gpus())
        )

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    LOG.info(model.summary())
    return model


def launch(filename, epochs):
    """TODO."""
    train_x, train_y, test_x, test_y = load_data(filename)
    model = get_model()

    LOG.info(
        'training for %s epochs with batch size %s and %s features'
        % (epochs, BATCH_SIZE, len(DATA_FEATURES))
    )

    model.fit(
        train_x, train_y, batch_size=BATCH_SIZE,
        epochs=epochs, validation_data=[test_x, test_y],
        callbacks=[EpochStatsLogger()]
    )


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    options = parse_args()
    sys.exit(launch(options.data_file, options.epochs))
