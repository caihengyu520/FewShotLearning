#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 15:00
# @Author  : HJH
# @Site    :
# @File    : covert_cifar10.py
# @Software: PyCharm

import tensorflow as tf
import os
import sys
import numpy as np
import pickle as p
from PIL import Image
_NUM_TRAIN_FILES = 5
LABELS_FILENAME = 'label.txt'
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def _int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _image_to_tfexample(image_data, image_format, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _int64_feature(class_id)
    }))


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    with tf.gfile.Open(filename, 'rb') as f:
        # get python version
        if sys.version_info < (3,):
            data = p.load(f)
        else:
            data = p.load(f, encoding='bytes')

    images = data[b'data']
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, 32, 32))
    labels = data[b'labels']

    with tf.Graph().as_default():
        for j in range(num_images):
            sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                filename, offset + j + 1, offset + num_images))
            sys.stdout.flush()

            image = np.squeeze(images[j]).transpose((1, 2, 0))
            image = Image.fromarray(image)
            image = image.resize((227, 227))
            # image.save('../images/image/' + str(j) + '.png')
            image = image.tobytes()
            label = labels[j]

            example = _image_to_tfexample(image, b'png', label)
            tfrecord_writer.write(example.SerializeToString())

    return offset + num_images


def _get_output_filename(dataset_dir, split_name):
    return '%s/cifar10_%s.tfrecord' % (dataset_dir, split_name)


def run(dataset_dir):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0
        for i in range(_NUM_TRAIN_FILES):
            filename = os.path.join(dataset_dir,
                                    'data_batch_%d' % (i + 1))  # 1-indexed.
            offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        filename = os.path.join(dataset_dir,
                                'test_batch')
        _add_to_tfrecord(filename, tfrecord_writer)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the Cifar10 dataset!')


if __name__ == '__main__':
    run('cifar-10-batches-py/')
