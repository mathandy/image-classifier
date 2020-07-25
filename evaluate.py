from main import Classifier
from loader import get_image_filepaths, filepath_to_label, load
from model import build_model

from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf


class Args(object):
    """Used to convert dict objects to namespace-like objects

    Credit:
        https://stackoverflow.com/questions/2597278
    """
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def prepare_test_data(image_dir, image_dimensions):

    file_paths = get_image_filepaths(image_dir=image_dir)
    labels = [filepath_to_label(fp) for fp in file_paths]
    class_names = list(set(labels))
    label_distribution = dict((l, labels.count(l)) for l in class_names)

    ds, class_names = load(
        file_paths=file_paths,
        augmentation_func=None,
        size=image_dimensions,
        shuffle_buffer=False,
    )

    return ds, class_names, label_distribution


def score(train_args, model_dir, image_dir, batch_size=None):

    ds, class_names, label_counts = prepare_test_data(
        image_dir=image_dir,
        image_dimensions=train_args.image_dimensions,
    )

    # set class weights to compensate for class imbalance
    class_weights = None
    if not train_args.no_class_weights:
        print(f"\nTrain Label Counts\n{label_counts}\n")
        class_weights = [1/c for c in label_counts.values()]

    # build model and load weights
    model = build_model(model_name=train_args.model, n_classes=len(class_names))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classifier = Classifier(
        model=model,
        optimizer=None,
        loss=loss,
        class_weights=class_weights,
        class_names=class_names,
        metric_dict={},
        logdir=model_dir,
    )
    classifier.model.load_weights(str(Path(model_dir, 'model.h5')))

    # make sure model is fixed (in a hackish way)
    for layer in classifier.model.layers:
        try:
            layer.trainable = False
            for sublayer in layer.layers:
                sublayer.trainable = False
        except:
            pass

    with open(Path(model_dir, 'evaluation-results.txt'), 'w') as f:
        f.write(','.join(
            ['file_path', 'ground truth'] + list(class_names)
        ) + '\n')
        for image, label, file_path in ds:
            # pass through model
            logits = classifier.model(np.expand_dims(image, 0))
            probabilities = tf.squeeze(tf.nn.softmax(logits))

            # parse and report results
            readable_probabilities = ['%f' % p for p in probabilities.numpy()]
            class_name = class_names[label.numpy()]
            fp = Path(file_path.numpy().decode())
            fp = str(Path(fp.parent.name, fp.name))
            f.write(','.join([fp, class_name] + readable_probabilities) + '\n')

    # fix and batch
    ds = ds.map(lambda image, label, file_path: (image, label))
    ds = ds.batch(batch_size)

    # report
    test_results, test_loss, test_cm = classifier.score(ds)
    test_acc = np.trace(test_cm) / np.array(test_cm).sum()
    test_results.update({'Loss': test_loss,
                         'Confusion Matrix': f'\n{test_cm}',
                         'Accuracy': test_acc})
    classifier.report(test_results, "Test Results",
                      write_to_log=Path(model_dir, 'evaluation-scores.txt'))


def get_user_args():
    """Fetch model parameters (including from CLI flags)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', '-m', type=Path, default=None,  # default set below
        help='The directly containing train_args.p and model.h5.'
    )
    parser.add_argument(
        '--image_dir', '-i', type=Path, default=None,  # default set below
        help='Path to a subdirectory-labeled image directory.'
    )
    args = parser.parse_args()
    return args


def get_train_args(logdir):
    with Path(logdir, 'train_args.p').open('rb') as f:
        train_args = pickle.load(f)

    if isinstance(train_args, dict):
        train_args = Args(train_args)
    return train_args


if __name__ == '__main__':
    eval_args_ = get_user_args()
    train_args_ = get_train_args(eval_args_.model_dir)
    score(train_args=train_args_,
          model_dir=eval_args_.model_dir,
          image_dir=eval_args_.image_dir)
