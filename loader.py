import os
from os.path import join as fpath
import tensorflow as tf
from tensorflow.data import Dataset as tfds
from imageio import read


def filepath_to_label(fp):
    return fp.split(os.sep)[-2]


def is_image(fp, extensions=('jpg', 'jpeg')):
    return os.path.splitext(fp.lower())[1:] in extensions


def get_image_filepaths(image_dir, extensions=('jpg', 'jpeg')):
    """Get image file paths from directory of subdirectory-labeled images."""

    image_paths = []
    for label in os.listdir(image_dir):
        if not os.path.isdir(fpath(image_dir, label)):
            continue

        for fn in os.listdir(fpath(image_dir, label)):

            if not is_image(fn, extensions):
                continue

            image_paths.append(fpath(image_dir, label, fn))
    return image_paths


@tf.function
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


@tf.function
def extract_label(file_path):
    return tf.strings.split(file_path, sep=os.sep)


def load(file_paths, augmentation_func=None, size=None, shuffle_buffer=False):
    labels = [filepath_to_label(fp) for fp in file_paths]
    class_names = list(set(labels))
    label_encoder = dict((label, k) for k, label in enumerate(class_names))
    encoded_labels = [label_encoder[label] for label in labels]

    # just file paths
    ds_file_paths = tfds.from_tensor_slices(file_paths)
    ds_labels = tfds.from_tensor_slices(encoded_labels)

    if augmentation_func is None:
        ds_images = ds_file_paths.map(load_image)
    else:
        image_generator = map(read, file_paths)
        augmented_image_generator = map(augmentation_func, image_generator)
        ds_images = tfds.from_generator(augmented_image_generator, tf.float32)

    if size is not None:
        ds_images = ds_images.map(lambda img: tf.image.resize(img, size))

    ds = tfds.zip((ds_images, ds_labels, ds_file_paths))

    if shuffle_buffer is not False:
        ds = ds.shuffle(shuffle_buffer)
    return ds
