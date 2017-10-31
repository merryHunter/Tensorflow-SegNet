"""
    @author Ivan Chernukha
    Date: 30/10/2017
"""
import tensorflow as tf
import time
import skimage.io as io

### EXAMPLE USAGE

class DatasetReader:
    """
    Abstract class for reading datasets in TFRecord format.
    """
    def __init__(self, dataset_name, path_to_tfrecords, img_height, img_width, annot_dims):
        self.annot_dims = annot_dims
        self.img_width = img_width
        self.img_height = img_height
        self.path_to_tfrecords = path_to_tfrecords
        self.dataset_name = dataset_name

    def _read_and_decode(self, split_name, filename_queue, batch_size, min_queue_examples):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        # TODO: check 3rd dim!
        annotation_shape = tf.stack([height, width, self.annot_dims]) # KITTI - 3

        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)

        # Adjust image size

        # resized_image = tf.random_crop(image, [height, width, 3])
        # resized_annotation = tf.random_crop(annotation, [height, width, 3])
        # image.set_shape([self.img_height, self.img_width, 3])
        # annotation.set_shape([self.img_height, self.img_width, 3])
        
        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=self.img_height,
                                                               target_width=self.img_width)

        resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                                                    target_height=self.img_height,
                                                                    target_width=self.img_width)
        if split_name != 'test':
            images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                         batch_size=batch_size,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         num_threads=1,
                                                         min_after_dequeue=min_queue_examples
                                                         )
        else:
            images, annotations = tf.train.batch([resized_image, resized_annotation],
                                                         batch_size=1,
                                                         capacity=1,
                                                         num_threads=1,
                                                         )

        return images, annotations

    def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        batch_size, shuffle):
        pass

    def generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        batch_size, shuffle):
        pass
    
    def get_images_labels(self, split_name, batch_size=5):
        tfrecords_filename = self.path_to_tfrecords + self.dataset_name + '-' + split_name + '.tfrecords'
        print(tfrecords_filename)
        filename_queue = tf.train.string_input_producer(
            [tfrecords_filename], num_epochs=None)
        min_fraction_of_examples_in_queue = 0.4
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
        min_queue_examples = 1
        if split_name != 'test':
            min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                   min_fraction_of_examples_in_queue)
            print ('Filling queue with %d CamVid images before starting to train. '
                 'This will take a few minutes.' % min_queue_examples)
        return self._read_and_decode(split_name, filename_queue, batch_size, min_queue_examples)

    
    def getNumberTestTFRecords(self):
        """ Count number of records in test TFRecord of self.dataset. """
        return sum(1 for _ in tf.python_io.tf_record_iterator(self.path_to_tfrecords
                                                              + self.dataset_name + '-test.tfrecords'))
    
    
    
    def demo(self):
        # tfrecords_filename = 'output/kitti-train.tfrecords'
        # tfrecords_filename = 'output/kitti-test.tfrecords'
        # tfrecords_filename = 'output/camvid-train.tfrecords'
        tfrecords_filename = self.path_to_tfrecords
        filename_queue = tf.train.string_input_producer(
            [tfrecords_filename], num_epochs=None)

        # Even when reading in multiple threads, share the filename
        # queue.
        image_, anno_ = self._read_and_decode(filename_queue)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:

            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                step = 0
                while not coord.should_stop():
                    start_time = time.time()
                    for i in xrange(3):
                        img_, ann_ = sess.run([image_, anno_])
                        print(img_[:, :, :].shape)
                        print('current batch')
                        # We selected the batch size of two
                        # So we should get two image pairs in each batch
                        # Let's make sure it is random
                        if self.annot_dims == 1:
                            io.imshow(img_[1, :, :, :])
                            io.show()

                            io.imshow(ann_[0, :, :, 0])
                            io.show()
                        else:
                            io.imshow(img_[1, :, :, :])
                            io.show()

                            io.imshow(ann_[1][:, :, :])
                            io.show()
                    duration = time.time() - start_time
                step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for epochs, %d steps.' % ( step))
            finally:
                # When done, ask the threads to stop.
                print step
                coord.request_stop()
            coord.request_stop()
            coord.join(threads)


# reader = DatasetReader(dataset_name='kitti', path_to_tfrecords='output/kitti-train.tfrecords',
#         img_height=375, img_width=1242, annot_dims=3)
# reader = DatasetReader(dataset_name='camvid', path_to_tfrecords='output/camvid-train.tfrecords',
#                        img_height=360, img_width=480, annot_dims=1)
# reader.demo()