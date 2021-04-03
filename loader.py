from augmenter import augment

from pathlib import Path
from os import sep as file_path_seperator
import numpy as np
import tensorflow as tf
# from imageio import imread

tfds = tf.data.Dataset
MAP_PARALLELISM = tf.data.experimental.AUTOTUNE


def filepath_to_label(fp):
    return Path(fp).parent.name


def is_image(fp, extensions=('jpg', 'jpeg')):
    return Path(fp).suffix[1:] in extensions


def get_image_filepaths(image_dir, png=False):
    """Get image file paths from directory of subdirectory-labeled images."""

    extensions = ('png',) if png else ('jpg', 'jpeg')
    image_dir = Path(image_dir)
    assert image_dir.exists()

    image_filepaths = [str(fp) for fp in image_dir.glob('*/*')
                       if is_image(fp, extensions)]
    assert len(image_filepaths)
    return image_filepaths


@tf.function
def load_jpeg(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img


@tf.function
def load_png(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    return img


@tf.function
def load_grayscale_jpeg(image_path):
    img = tf.io.read_file(image_path)
    img = tf.squeeze(tf.io.decode_jpeg(img, channels=1))
    return tf.stack([img, img, img], axis=2)


@tf.function
def load_grayscale_png(image_path):
    img = tf.io.read_file(image_path)
    img = tf.squeeze(tf.io.decode_png(img, channels=1))
    return tf.stack([img, img, img], axis=2)


@tf.function
def extract_label(file_path):
    return tf.strings.split(file_path, sep=file_path_seperator)


def shape_setter(shape):
    """fix for tf problems when shape confusingly isn't set"""
    @tf.function
    def shape_setter_func(img):
        img.set_shape(shape)
        return img
    return shape_setter_func


def load(file_paths, augmentation_func=None, size=None, class_names=None,
         include_filepaths=False, grayscale=False, png=False):
    labels = [filepath_to_label(fp) for fp in file_paths]
    if class_names is None:
        class_names = list(set(labels))
    label_encoder = dict((label, k) for k, label in enumerate(class_names))
    encoded_labels = [label_encoder[label] for label in labels]

    # just file paths
    ds_file_paths = tfds.from_tensor_slices(file_paths)
    ds_labels = tfds.from_tensor_slices(encoded_labels)

    if grayscale:
        load_func = load_grayscale_png if png else load_grayscale_jpeg
    else:
        load_func = load_png if png else load_jpeg

    ds_images = ds_file_paths.map(load_func, num_parallel_calls=MAP_PARALLELISM)

    if augmentation_func is not None:
        ds_images = ds_images.cache()
        ds_images = ds_images.map(
            map_func=lambda img: tf.numpy_function(func=augmentation_func,
                                                   inp=[img], Tout=[tf.uint8]),
            num_parallel_calls=MAP_PARALLELISM
        )
    # else:
    #     def generate_augmented_epoch():
    #         image_generator = map(imread, file_paths)
    #         return map(augmentation_func, image_generator)
    #     ds_images = tfds.from_generator(generate_augmented_epoch, tf.float32)

    ds_images = ds_images.map(shape_setter([None, None, 3]),
                              num_parallel_calls=MAP_PARALLELISM)

    if size is not None:
        @tf.function
        def resize(img):
            return tf.image.resize(img, list(size))

        # from IPython import embed;embed()  ### DEBUG
        # ds_images = ds_images.map(resize, num_parallel_calls=MAP_PARALLELISM)
        ds_images = ds_images.map(lambda img: tf.image.resize(img, list(size)),
                                  num_parallel_calls=MAP_PARALLELISM)
        ds_images = ds_images.map(shape_setter(list(size) + [3]),
                                  num_parallel_calls=MAP_PARALLELISM)

    # scale pixel values to [0, 1]
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input
    ds_images = ds_images.map(
        map_func=lambda img: tf.image.convert_image_dtype(img, tf.float32),
        num_parallel_calls=MAP_PARALLELISM
    )

    # zip, shuffle, batch, and return
    if include_filepaths:
        ds = tfds.zip((ds_images, ds_labels, ds_file_paths))
    else:
        ds = tfds.zip((ds_images, ds_labels))
    return ds, class_names


def prepare_data(args):
    assert args.test_dir and args.test_part == 0 or 0 < args.test_part < 1
    assert args.val_dir and args.val_part == 0 or 0 < args.val_part < 1
    assert args.test_part + args.val_part < 1

    file_paths = get_image_filepaths(args.image_dir, args.png)
    np.random.shuffle(file_paths)
    labels = [filepath_to_label(fp) for fp in file_paths]
    class_names = list(set(labels))

    # split file paths into train and val sets
    train_file_paths, val_file_paths, test_file_paths = [], [], []
    label_distribution, train_label_distribution = {}, {}
    for label in class_names:
        label_fps = [fp for fp, l in zip(file_paths, labels) if l == label]

        n_val = int(np.ceil(len(label_fps) * args.val_part))
        n_test = int(np.ceil(len(label_fps) * args.test_part))
        n_train = len(label_fps) - n_val - n_test

        train_file_paths += label_fps[:n_train]
        if args.val_part > 0:
            assert n_val > 0
            val_file_paths += label_fps[n_train:n_train + n_val]

        if args.test_part > 0:
            assert n_test > 0
            test_file_paths += label_fps[n_train + n_val:]

        # record how many examples there are of each label
        label_distribution[label] = len(label_fps)
        train_label_distribution[label] = n_train

    if args.val_dir is not None:
        val_file_paths = get_image_filepaths(args.val_dir, args.png)
    if args.test_dir is not None:
        test_file_paths = get_image_filepaths(args.test_dir, args.png)

    np.random.shuffle(train_file_paths)
    np.random.shuffle(val_file_paths)
    np.random.shuffle(test_file_paths)

    ds_train, train_class_names = load(
        file_paths=train_file_paths,
        augmentation_func=augment,
        size=args.image_dimensions,
        grayscale=args.grayscale,
        png=args.png,
    )

    ds_val, val_class_names = load(
        file_paths=val_file_paths,
        augmentation_func=None,
        size=args.image_dimensions,
        grayscale=args.grayscale,
        png=args.png,
    )

    ds_test, test_class_names = load(
        file_paths=test_file_paths,
        augmentation_func=None,
        size=args.image_dimensions,
        grayscale=args.grayscale,
        png=args.png,
    )

    assert set(test_class_names) == set(val_class_names) == \
           set(train_class_names) == set(class_names)

    ds_train = ds_train.shuffle(
        buffer_size=min(10 * args.batch_size, len(train_file_paths)))

    ds_train = ds_train.batch(args.batch_size)
    ds_val   =   ds_val.batch(args.batch_size)
    ds_test  =  ds_test.batch(args.batch_size)

    def optimize(ds):
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        ds = ds.with_options(options)
        return ds

    # ds_train = ds_train.optimize()
    ds_val = ds_val.cache()

    def count_batches(file_paths):
        return int(np.ceil(len(file_paths)/args.batch_size))

    # prefetch training and val sets, do not prefetch test set
    # ds_train = ds_train.prefetch(count_batches(train_file_paths))
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.prefetch(count_batches(val_file_paths))
    return ds_train, ds_val, ds_test, class_names, train_label_distribution


def load_test(args):
    from tempfile import gettempdir
    import numpy as np
    from augmenter import augment
    from os import system as system_call

    temp_image_file_path = str(Path(gettempdir(), 'temp_image.jpg'))

    file_paths = get_image_filepaths(args.image_dir, args.png)
    np.random.shuffle(file_paths)

    ds, class_names = load(
        file_paths=file_paths,
        augmentation_func=augment,
        size=(100, 100),
        include_filepaths=True,
        grayscale=args.grayscale,
        png=args.png,
    )
    ds = ds.shuffle(buffer_size=min(10 * args.batch_size, len(file_paths)))

    for augmented_image, label, original_path in ds:
        print(f"label: {class_names[label]}\n"
              f"file path: {original_path}\n" + \
              "pixel range: {} - {}".format(tf.reduce_min(augmented_image),
                                            tf.reduce_max(augmented_image)))

        original_image = load_jpeg(original_path)
        original_image = tf.cast(original_image, tf.uint8)
        augmented_image = tf.cast(augmented_image, tf.uint8)

        w = max(original_image.shape[1], augmented_image.shape[1])
        h = max(original_image.shape[0], augmented_image.shape[0])
        original_image = tf.image.resize_with_pad(original_image, h, w)
        augmented_image = tf.image.resize_with_pad(augmented_image, h, w)

        # write a side-by-side image comparison to disk
        side_by_side = tf.concat([original_image, augmented_image], axis=1)
        side_by_side = tf.io.encode_jpeg(tf.cast(side_by_side, tf.uint8))
        tf.io.write_file(temp_image_file_path, side_by_side)

        system_call(f'open {temp_image_file_path}')
        user_says = input("Press enter to see next image (q to quit).")
        if user_says.strip() == 'q':
            return


def benchmark_input(args):
    from tensorflow_datasets.core import benchmark

    ds_train, ds_val, ds_test, _, _ = prepare_data(args)

    def report_stats(ds, report_title='Benchmark Statistics',
                     num_iter=None, batch_size=args.batch_size):
        stats = benchmark(ds, num_iter=num_iter, batch_size=batch_size)
        print(f"\n{report_title}\n{'-'*len(report_title)}")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f'{k}:')
                for kk, vv in v.items():
                    print(f'\t{kk}: {vv}')
            else:
                print(f'{k}: {v}')
        return stats

    report_stats(ds=ds_train, report_title="Train Statistics")
    # report_stats(ds=ds_val, report_title="Val Statistics")
    # report_stats(ds=ds_test, report_title="Test Statistics")
    print()


if __name__ == '__main__':
    print("\nTo run load_test or benchmark use main.py and --test_load "
          "or --benchmark_input respectively.\n")
