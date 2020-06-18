from pathlib import Path
from os import sep as file_path_seperator
import tensorflow as tf
from imageio import imread

tfds = tf.data.Dataset


def filepath_to_label(fp):
    return Path(fp).parent.name


def is_image(fp, extensions=('jpg', 'jpeg')):
    return Path(fp).suffix[1:] in extensions


def get_image_filepaths(image_dir, extensions=('jpg', 'jpeg')):
    """Get image file paths from directory of subdirectory-labeled images."""

    image_dir = Path(image_dir)
    assert image_dir.exists()

    return [str(fp) for fp in image_dir.glob('*/*')
            if is_image(fp, extensions)]


@tf.function
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img


@tf.function
def extract_label(file_path):
    return tf.strings.split(file_path, sep=file_path_seperator)


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
        def generate_augmented_epoch():
            image_generator = map(imread, file_paths)
            return map(augmentation_func, image_generator)
        ds_images = tfds.from_generator(generate_augmented_epoch, tf.float32)

    if size is not None:
        ds_images = ds_images.map(lambda img: tf.image.resize(img, size))

    # scale pixel values to [0, 1]
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input
    ds_images.map(lambda img: tf.image.convert_image_dtype(img, tf.float32))

    # zip, shuffle, batch, and return
    ds = tfds.zip((ds_images, ds_labels, ds_file_paths))
    if shuffle_buffer is not False:
        ds = ds.shuffle(shuffle_buffer)
    return ds


def test(args):
    from tempfile import gettempdir
    import numpy as np
    from augmentation import Augmenter
    from os import system as system_call

    temp_image_file_path = Path(gettempdir(), 'temp_image.jpg')

    file_paths = get_image_filepaths(image_dir=args.image_dir)
    np.random.shuffle(file_paths)

    ds = load(
        file_paths=file_paths,
        augmentation_func=Augmenter(),
        size=(100, 100),
        shuffle_buffer=min(10 * args.batch_size, len(file_paths)),
    )

    for image, label, file_path in ds:
        print(f"label: {label}\n"
              f"file path: {file_path}\n"
              f"pixel range: {(tf.reduce_min(image), tf.reduce_max(image))}")
        tf.io.write_file(temp_image_file_path, tf.io.encode_jpeg(image))
        system_call(f'open {temp_image_file_path}')
        user_says = input("Press enter to see next image (q to quit).")
        if user_says.strip() == 'q':
            return


if __name__ == '__main__':
    from config import get_user_args
    test(get_user_args())
