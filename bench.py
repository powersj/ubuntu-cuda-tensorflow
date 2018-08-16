#!/usr/bin/env python3
"""Run matrix multiplication with Tensorflow on a GPU and CPU.

This is a short demo of various matrix sizes to show how a GPU can
become more efficient at calculating matrix multiplication with
Tensorflow and CUDA.
"""

import argparse
import sys
import time

from tabulate import tabulate
import tensorflow as tf

MATRIX_SHAPES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]


def create_matrix(matrix_shape):
    """Create a matrix of specified shape.

    Args:
        matrix_shape: tuple of (x, y)
    Returns:
        tensor of random values from a uniform distribution

    """
    return tf.random_uniform(
        shape=matrix_shape,
        minval=0,
        maxval=1,
        dtype=tf.float16
    )


def time_tf_matrixmul(device, matrix_shape):
    """Run tensorflow with device and matrix size.

    Args:
        device: the name of the tensorflow device to use
        matrix_shape: tuple of (x, y)
    Returns:
        time in seconds for matrix multiplication operation

    """
    with tf.Session() as session:
        with tf.device(device):
            matrix_a = create_matrix(matrix_shape)
            matrix_b = create_matrix(matrix_shape)

            start = time.time()
            session.run(tf.matmul(matrix_a, matrix_b))
            end = time.time()

    return round(end - start, 2)


def bench(device):
    """Run matrix multiplication on various sizes.

    Args:
        device: the name of the tensorflow device to use
    """
    results = []
    for shape in MATRIX_SHAPES:
        print('%s with %sx%s' % (device, shape, shape))
        time = time_tf_matrixmul(device, (shape, shape))
        results.append([device, shape, time])

    print(tabulate(results, headers=['Shape', 'Time']))


def launch():
    """Set up arguments and launch testing on requested devices."""
    parser = argparse.ArgumentParser(prog='bench')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--cpu', action='store_true', help='run with cpu'
    )
    group.add_argument(
        '--gpu', action='store_true', help='run with gpu'
    )

    args = parser.parse_args()
    if args.gpu:
        bench('/gpu:0')
    if args.cpu:
        bench('/cpu:0')


if __name__ == '__main__':
    sys.exit(launch())
