from loader import load, get_image_filepaths, filepath_to_label
from augmenter import augment
from model import build_model

import tensorflow as tf
import numpy as np


def prepare_data(args):

    file_paths = get_image_filepaths(image_dir=args.image_dir)
    np.random.shuffle(file_paths)
    labels = [filepath_to_label(fp) for fp in file_paths]
    class_names = list(set(labels))

    # split file paths into train and val sets
    train_file_paths, val_file_paths = [], []
    label_distribution, train_label_distribution = {}, {}
    for label in class_names:
        label_fps = [fp for fp, l in zip(file_paths, labels) if l == label]

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

    assert set(test_class_names) == set(val_class_names) == \
           set(train_class_names) == set(class_names)
    return ds_train, ds_val, ds_test, class_names


def main(args):

    # prepare data
    ds_train, ds_val, ds_test, class_names = prepare_data(args)
    ds_train = ds_train.map(lambda image, label, file_path: (image, label))
    ds_val   =   ds_val.map(lambda image, label, file_path: (image, label))
    # ds_test = ds_test.map(lambda image, label, file_path: (image, label))
    ds_train = ds_train.batch(args.batch_size)
    ds_val = ds_val.batch(args.batch_size)
    # ds_test = ds_test.batch(args.batch_size)

    # define metrics
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    # for class_id, class_name in enumerate(class_names):
    #     metrics.append(tf.keras.metrics.Precision(
    #         class_id=class_id, name='pr_' + class_name))
    #     metrics.append(tf.keras.metrics.Recall(
    #         class_id=class_id, name='re_' + class_name))

    # class_weights = dict((k, ))

    # build model
    model = build_model(model_name=args.model, n_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )

    # train model
    model.fit(ds_train,
              batch_size=args.batch_size,
              validation_data=ds_val,
              class_weight=None,
              epochs=100)


if __name__ == '__main__':
    from config import get_user_args
    main(get_user_args())
