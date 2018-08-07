from array import array
import numpy as np
import struct
import sys
import os

class MNISTLoader:

    def __init__(self, path):
        self.path = path

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'
        self.train_images, self.train_labels = [], []

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'
        self.test_images, self.test_labels = [], []

        self.num_classes = 10
        self.rows = 28
        self.cols = 28
        self.channels = 1

        self.load_data()

    def load_data(self):
        imgs, labels = self.load(os.path.join(self.path, self.train_img_fname),
            os.path.join(self.path, self.train_lbl_fname))
        self.train_images = self.process_images(imgs)
        self.train_labels = self.process_labels(labels)
        print('Train data:', self.train_images.shape, self.train_labels.shape)

        imgs, labels = self.load(os.path.join(self.path, self.test_img_fname),
            os.path.join(self.path, self.test_lbl_fname))
        self.test_images = self.process_images(imgs)
        self.test_labels = self.process_labels(labels)
        print('Test data:', self.test_images.shape, self.test_labels.shape)

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @staticmethod
    def process_images(images):
        return np.array(images) / 255.

    @staticmethod
    def process_labels(labels):
        return np.array(labels)[:, np.newaxis]
