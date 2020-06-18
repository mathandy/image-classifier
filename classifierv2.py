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
        size=args.image_dimensions,
        shuffle_buffer=min(10 * args.batch_size, n_train),
    )

    ds_val = load(
        file_paths=val_file_paths,
        augmentation_func=None,
        size=args.image_dimensions,
        shuffle_buffer=False,
    )

    ds_test = None
    if args.test_dir is not None:
        test_file_paths = get_image_filepaths(image_dir=args.test_dir)
        ds_test = load(
            file_paths=test_file_paths,
            augmentation_func=None,
            size=args.image_dimensions,
            shuffle_buffer=False,
        )
    return ds_train, ds_val, ds_test


def main(args):
    ds_train, ds_val, ds_test = prepare_data(args)

    TRAIN! and TEST!


if __name__ == '__main__':
    from config import get_user_args
    main(get_user_args())
