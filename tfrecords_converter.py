"""
    @author Ivan Chernukha
    Date: 30/10/2017

    Simple script to convert KITTI, Cityscapes, CamVid, Mapillary datasets
    into TFRecords.

    TODO: add requirements.txt
    TODO: add ok messages for datasets
"""
import numpy as np
import skimage.io as io
import tensorflow as tf
import time
from PIL import Image
import os
import warnings
import shutil


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('kitti_path','', 'full path to kitti lane folder')
tf.app.flags.DEFINE_string('cityscapes_path','', 'full path to cityscapes folder')
tf.app.flags.DEFINE_string('camvid_path','', 'full path to CamVid folder')
tf.app.flags.DEFINE_string('mapillary_path','', 'full path to mapillary folder')
tf.app.flags.DEFINE_string('output','tfrecords_datasets/', 'output folder')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_image_sets(path, train, val, test):
    for image_set in ['train', 'val', 'test']:
        df = eval(image_set)
        with open(os.path.join(path, image_set + '.txt'), 'w') as f:
            for im, l in df:
                f.writelines(im + ' ' + l + '\n')


def get_filename_pairs(dataset_name, path):
    """
    Returns full path of images and image annotations.
    :param path: dataset folder path
    :return: List of tuples [(img_path1, ann_path1), (img_path2,ann_path2)..]
    """
    train = []; val = []; test = []
    train_images = []; train_labels = []
    if dataset_name == 'kitti':
        trainpath = os.path.join(path, 'training/')
        testpath = os.path.join(path, 'testing/')

        for f in os.listdir(os.path.join(trainpath, 'image_2/')):
            train_images.append(os.path.join(trainpath, 'image_2/', f))

            fsplit = f.split('_')
            train_labels.append(os.path.join(trainpath, 'gt_image_2/',
                                             fsplit[0] + '_road_' + fsplit[1]))

        # Hold out 10% for validation
        val_images = train_images[int(len(train_images) * .9):]
        val_labels = train_labels[int(len(train_labels) * .9):]

        train_images = train_images[:int(len(train_images) * .9)]
        train_labels = train_labels[:int(len(train_labels) * .9)]

        train = zip(train_images, train_labels)
        val = zip(val_images, val_labels)
        test_images = [os.path.join(testpath, 'image_2/',f)
                       for f in os.listdir(os.path.join(testpath, 'image_2/'))]

        print('KITTI train, val images and labels loaded.')
        print("Train: {0}\nVal: {1}\nTest:{2}\n"
              .format(len(train_images), len(val_images), len(test_images)))
        warnings.warn('KITTI test split don"t contain labels!')
        test = zip(test_images,test_images)

        create_image_sets(path, train, val, test)

    elif dataset_name == 'mapillary':
        pass
    elif dataset_name == 'cityscapes':
        pass
    elif dataset_name == 'camvid':
        trainpath = os.path.join(path, 'train/')
        valpath = os.path.join(path, 'val/')
        testpath = os.path.join(path, 'test/')
        with open(os.path.join(path, 'train.txt'), 'r') as f:
            # It is already written as pair of 'img/path ann/path', but local
            train_images = [tuple(map(str, i.split(' '))) for i in f]
            for o, a in train_images:
                # Transform an overlapping path to united full path of imgs and anns
                im = path + '/'.join([i for i in o.split('/') if i not in path.split('/')])
                an = path + '/'.join([i for i in a.split('/') if i not in path.split('/')])
                train.append(tuple((im.strip(), an.strip())))
        with open(os.path.join(path, 'val.txt'), 'r') as f:
            val_images = [tuple(map(str, i.split(' '))) for i in f]
            for o, a in val_images:
                im = path + '/'.join([i for i in o.split('/') if i not in path.split('/')])
                an = path + '/'.join([i for i in a.split('/') if i not in path.split('/')])
                val.append(tuple((im.strip(), an.strip())))
        with open(os.path.join(path, 'test.txt'), 'r') as f:
            test_images = [tuple(map(str, i.split(' '))) for i in f]
            for o, a in test_images:
                im = path + '/'.join([i for i in o.split('/') if i not in path.split('/')])
                an = path + '/'.join([i for i in a.split('/') if i not in path.split('/')])
                test.append(tuple((im.strip(), an.strip())))

    return train, val, test


def convert_dataset_to_tfrecord(dataset_name, path):
    """
    Convert a dataset to tfrecord format and output to given folder.
    :param dataset_name:
    """
    # TODO: Check path assert
    if not os.path.exists(path):
        raise ('Please, provide a valid path to the dataset {0}!'.format(dataset_name))

    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    train, val, test = get_filename_pairs(dataset_name, path)

    for split in ['train', 'val', 'test']:
        tfrecords_filename = FLAGS.output + '{0}-{1}.tfrecords'.format(dataset_name, split)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        # Get the filename pairs we received from get_filename_pairs
        filename_pairs = eval(split)
        for img_path, annotation_path in filename_pairs:
            img = np.array(Image.open(img_path))
            annotation = np.array(Image.open(annotation_path))

            # The reason to store image sizes - we have to know sizes
            # of images to later read raw serialized string,
            # convert to 1d array and convert to respective
            # shape that image used to have.
            height = img.shape[0]
            width = img.shape[1]

            img_raw = img.tostring()
            annotation_raw = annotation.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img_raw),
                'mask_raw': _bytes_feature(annotation_raw)}))

            writer.write(example.SerializeToString())

        writer.close()


def main(args):
    if os.path.isdir(FLAGS.kitti_path):
        convert_dataset_to_tfrecord('kitti', FLAGS.kitti_path)
    if os.path.isdir(FLAGS.mapillary_path):
        convert_dataset_to_tfrecord('mapillary', FLAGS.mapillary_path)
    if os.path.isdir(FLAGS.cityscapes_path):
        convert_dataset_to_tfrecord('cityscapes', FLAGS.cityscapes_path)
    if os.path.isdir(FLAGS.camvid_path):
        convert_dataset_to_tfrecord('camvid', FLAGS.camvid_path)

if __name__ == '__main__':

    tf.app.run()

