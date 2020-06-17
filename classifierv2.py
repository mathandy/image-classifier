from loader import load, get_image_filepaths
from augmentation import Augmenter
import numpy as np


def prepare_data(args):
    file_paths = get_image_filepaths(image_dir=args.image_dir)
    np.random.shuffle(file_paths)

    n_train = int(np.ceil(len(file_paths) * (1 - args.val_part)))
    train_file_paths = file_paths[:n_train]
    val_file_paths = file_paths[n_train:]

    ds_train = load(
        file_paths=train_file_paths,
        augmentation_func=Augmenter(),
        size=(299, 299),
        shuffle_buffer=min(10 * args.batch_size, n_train),
    )

    ds_val = load(
        file_paths=val_file_paths,
        augmentation_func=None,
        size=(299, 299),
        shuffle_buffer=False,
    )

    ds_test = None
    if args.test_dir is not None:
        test_file_paths = get_image_filepaths(image_dir=args.test_dir)
        ds_test = load(
            file_paths=test_file_paths,
            augmentation_func=None,
            size=(299, 299),
            shuffle_buffer=False,
        )
    return ds_train, ds_val, ds_test


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_dir',
        help='Path to subdirectory-labeled image directory.'
    )
    parser.add_argument(
        '--val_part', default=0.1, type=float,
        help='Portion of training images to reserve for validation.'
    )
    parser.add_argument(
        '--test_dir', default=None,
        help='Path to subdirectory-labeled image directory for testing.'
    )
    parser.add_argument(
        '--batch_size', default=128,
        help='Batch size.'
    )
    args = parser.parse_args()

    ds_train, ds_val, ds_test = prepare_data(args)

    TRAIN! and TEST!
    

if __name__ == '__main__':
    main()
