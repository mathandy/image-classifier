from loader import load, get_image_filepaths, filepath_to_label
from augmenter import augment
from available_tfhub_models import tf_hub_model_input_size, tf_hub_model_urls
from model import build_model

import numpy as np


def prepare_data(args):

    file_paths = get_image_filepaths(image_dir=args.image_dir)
    labels = [filepath_to_label(fp) for fp in file_paths]
    class_names = list(set(labels))

    # split file paths into train and val sets
    train_file_paths, val_file_paths = [], []
    label_distribution, train_label_distribution = {}, {}
    for label in class_names:
        label_fps = [fp for fp, l in zip(file_paths, labels) if l == label]
        np.random.shuffle(label_fps)

        n_train = int(np.ceil(len(label_fps) * (1 - args.val_part)))
        train_file_paths += label_fps[:n_train]
        val_file_paths += label_fps[n_train:]
        assert len(val_file_paths) > 0

        # record how many examples there are of each label
        label_distribution[label] = len(label_fps)
        train_label_distribution[label] = n_train

    ds_train, train_class_names = load(
        file_paths=train_file_paths,
        augmentation_func=augment,
        size=args.image_dimensions,
        shuffle_buffer=min(10 * args.batch_size, len(train_file_paths)),
    )

    ds_val, val_class_names = load(
        file_paths=val_file_paths,
        augmentation_func=None,
        size=args.image_dimensions,
        shuffle_buffer=False,
    )

    if args.test_dir is not None:
        test_file_paths = get_image_filepaths(image_dir=args.test_dir)
        ds_test, test_class_names = load(
            file_paths=test_file_paths,
            augmentation_func=None,
            size=args.image_dimensions,
            shuffle_buffer=False,
        )
    else:
        ds_test, test_class_names = ds_val, val_class_names

    assert test_class_names == val_class_names == train_class_names == class_names
    return ds_train, ds_val, ds_test, class_names


# def train(args):
#     from model import build_model
#     from loader import load, get_image_filepaths
#     from augmenter import augment
#     import numpy as np
#
#     file_paths = get_image_filepaths(image_dir=args.image_dir)
#     np.shuffle(file_paths)
#     ds, class_names = load(
#         file_paths=file_paths,
#         augmentation_func=augment,
#         size=(100, 100),
#         shuffle_buffer=min(10 * args.batch_size, len(file_paths)),
#     )
#
#     model = build_model(model_path=args.model, n_classes=len(class_names))
#     model.fit()


def main(args):
    ds_train, ds_val, ds_test, class_names = prepare_data(args)
    model = build_model(model_path=args.model, n_classes=len(class_names))

    ds_train = ds_train.map(lambda image, label, file_path: (image, label))
    ds_val   =   ds_val.map(lambda image, label, file_path: (image, label))
    model.fit(ds_train,
              batch_size=args.batch_size,
              validation_data=ds_val,
              class_weight=None,
              epochs=3)


if __name__ == '__main__':
    from config import get_user_args
    main(get_user_args())
