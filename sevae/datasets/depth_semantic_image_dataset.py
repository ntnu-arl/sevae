import torch
import tensorflow as tf
import numpy as np
import os
import sys
from math import ceil

sys.path.append('.')
# from data_processing.utilities import get_files_ending_with



# TFRecord is used here to read from a serialized dataset. 

# There can really be a better way to do this, but need to find out what and how. Explore: using pickle to store the data.
# It is usually very painful to have to use both tensorflow and pytorch in the same project. Better to stick to one.

IMAGES_PER_TF_RECORD = 1000

class DepthSemanticImageDataset(torch.utils.data.IterableDataset):
    """DepthSemanticImageDataset wraps a torch.nn.IterableDataset around a tf.data.TFRecordDataset. 

    This allows the the depth images to fit in memory as the TFRecordDataset retrieves depth images sequencially during training.
    """

    def __init__(self, tfrecord_folder, batch_size=32, shuffle=False, one_tfrecord=False, test_dataset=False):
        super(DepthSemanticImageDataset).__init__()
        self.tfrecord_folder = tfrecord_folder
        self.dataset, self.data_len = self.load_tfrecords(
            is_shuffle_and_repeat=shuffle, batch_size=batch_size, one_tfrecord=one_tfrecord, test_dataset=test_dataset)

    def read_tfrecord(self, serialized_example):
        print("Reading dataset")
        feature_description_v2 = {
            'depth_image_raw': tf.io.FixedLenFeature([], tf.string),
            'depth_image_filtered': tf.io.FixedLenFeature([], tf.string),
            'semantic_image_raw': tf.io.FixedLenFeature([], tf.string),
            'num_envs': tf.io.FixedLenFeature([], tf.int64),
            'num_images_per_env': tf.io.FixedLenFeature([], tf.int64),}
        
        feature_description_old = {
            'depth_image_raw': tf.io.FixedLenFeature([], tf.string),
            'semantic_image_raw': tf.io.FixedLenFeature([], tf.string),
            'num_envs': tf.io.FixedLenFeature([], tf.int64),
            'num_images_per_env': tf.io.FixedLenFeature([], tf.int64),}
        try:
            example = tf.io.parse_single_example(
                serialized_example, feature_description_v2)
        except:
            example = tf.io.parse_single_example(
                serialized_example, feature_description_old)
        print("parsing depth image")
        parsed_depth_tensor = tf.cast(tf.io.parse_tensor(
            example['depth_image_raw'], out_type=np.float), tf.float32) / 1000.0
        print("parsing depth image filtered")
        parsed_depth_tensor_filtered = tf.cast(tf.io.parse_tensor(
            example['depth_image_filtered'], out_type=np.float), tf.float32) / 1000.0
        print("parsing semantic image")
        parsed_semantic_tensor = tf.cast(tf.io.parse_tensor(
            example['semantic_image_raw'], out_type=np.float), tf.int64)
        # If semantic information is not aviailable, set it to a matrix of zeros
        if parsed_semantic_tensor.shape[0] == 1:
            parsed_semantic_tensor = tf.zeros_like(parsed_depth_tensor)
        print("parsing num envs")
        parsed_num_envs = example['num_envs']
        print("parsing images per env")
        parsed_num_images_per_env = example['num_images_per_env']
        return parsed_num_envs, parsed_num_images_per_env, parsed_depth_tensor, parsed_depth_tensor_filtered, parsed_semantic_tensor

    def load_tfrecords(self, is_shuffle_and_repeat=True, shuffle_buffer_size=5000, prefetch_buffer_size_multiplier=2, batch_size=32, one_tfrecord=False, test_dataset=False):
        print('Loading tfrecords from folder ', self.tfrecord_folder)
        assert os.path.exists(self.tfrecord_folder), "TF Record folder path: {} does not exist".format(self.tfrecord_folder)
        assert os.path.isdir(self.tfrecord_folder), "TF Record folder path: {} is not a directory".format(self.tfrecord_folder)

        self.tfrecord_fnames = [os.path.join(self.tfrecord_folder, f) for f in os.listdir(self.tfrecord_folder) if f.endswith('.tfrecord')]
        
        print("Found {} tfrecords".format(len(self.tfrecord_fnames)))
        print("TFRecord names: ", self.tfrecord_fnames)
        
        assert len(self.tfrecord_fnames) > 0
        if is_shuffle_and_repeat:
            np.random.shuffle(self.tfrecord_fnames)
        else:
            # 176 tfrecords for train, 20 for test
            self.tfrecord_fnames = sorted(self.tfrecord_fnames)

        if one_tfrecord:
            self.tfrecord_fnames = self.tfrecord_fnames[:1]
            print(self.tfrecord_fnames)

        dataset = tf.data.TFRecordDataset(self.tfrecord_fnames)
        dataset = dataset.map(self.read_tfrecord,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_shuffle_and_repeat:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(
            buffer_size=prefetch_buffer_size_multiplier * batch_size)

        print('Iterating length... ', end="\t")
        return dataset, IMAGES_PER_TF_RECORD *len(self.tfrecord_fnames)

    def __iter__(self):
        print("gotcha")
        return self.dataset.__iter__()

    def __len__(self):
        return self.data_len


def collate_batch(batch):
    """Used as the 'collate_fn' when creating a torch.nn.DataLoader."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_envs = torch.Tensor(np.array(batch[0][0]))
    num_images_per_env = torch.Tensor(np.array(batch[0][1]))
    depth_images = torch.Tensor(np.array(batch[0][2]))
    filtered_images = torch.Tensor(np.array(batch[0][3]))
    semantic_images = torch.Tensor(np.array(batch[0][4]))
    

    return num_envs, num_images_per_env, depth_images, filtered_images, semantic_images
