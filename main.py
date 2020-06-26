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


class Classifier:
    def __init__(self, model, optimizer, loss, metric_dict, class_weights, args):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.class_weights = class_weights
        self.metric_dict = metric_dict
        self.args = args

    def get_report(self, results, title=None):
        report = ''
        if title is not None:
            report =  title + '\n'
        report += '\n'.join(f'{k}: {v}' for k, v in results.items())
        return f'\n{report}\n'

    def update_metrics(self, y_true, y_pred):
        for metric in self.metric_dict.values():
            metric.update_state(y_true, y_pred)

    def get_metric_results(self):
        results = {}
        for metric_name, metric in self.metric_dict:
            results[metric_name] = metric.result()
        return results

    def reset_metrics(self):
        for metric in self.metric_dict.values():
            metric.reset_states()

    def train_step(self, x_batch, y_batch, sample_weights=None):
        with tf.GradientTape() as tape:
            # ANDY - figure out if this updates batch norm layers even
            # though they are marked not trainable
            # TO SEE, check self.model.trainable_weights
            logits = self.model(x_batch, training=True)
            loss_value = self.loss(y_batch, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        if sample_weights is not None:
            grads = grads * sample_weights
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return logits

    def score(self, batches, reset=True):
        self.reset_metrics()
        for x_batch, y_batch in batches:
            logits = self.model(x_batch, training=False)
            predictions = tf.argmax(logits, 1)
            self.loss.update_state(y_batch, predictions)
            self.update_metrics(y_batch, predictions)
        results = self.get_metric_results()
        self.reset_metrics()
        return results

    def compute_sample_weights(self, y_true):
        if self.class_weights is None:
            return None
        raise NotImplementedError

    def train(self, training_data, validation_data=None, class_weights=None):
        epochs = 2
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            epoch_train_loss = 0
            for step, (x_train_batch, y_train_batch) in enumerate(training_data):
                if class_weights is None:
                    sample_weights = None
                else:
                    raise NotImplementedError
                train_batch_logits = self.train_step(
                    x_batch=x_train_batch,
                    y_batch=y_train_batch,
                    sample_weights=self.compute_sample_weights(y_train_batch)
                )
                train_batch_predictions = tf.argmax(train_batch_logits, 1)
                epoch_train_loss += \
                    self.update_metrics(y_train_batch, train_batch_predictions)
            train_results = self.get_metric_results()
            self.reset_metrics()
            train_results.update({'Loss': epoch_train_loss})
            print(self.get_report(train_results, "Training Results"))

            if validation_data is not None:
                self.score(validation_data)
                val_results = self.get_metric_results()
                self.reset_metrics()
                print(self.get_report(val_results, "Validation Results"))


def main(args):

    # prepare data
    ds_train, ds_val, ds_test, class_names = prepare_data(args)
    ds_train = ds_train.map(lambda image, label, file_path: (image, label))
    ds_val   =   ds_val.map(lambda image, label, file_path: (image, label))
    # ds_test = ds_test.map(lambda image, label, file_path: (image, label))
    ds_train = ds_train.batch(args.batch_size)
    ds_val = ds_val.batch(args.batch_size)
    # ds_test = ds_test.batch(args.batch_size)

    # set class weights to compensate for class imbalance
    class_weights = None
    if args.balance_class_weights:
        raise NotImplementedError

    # define metrics
    metrics = {'Accuracy': tf.keras.metrics.SparseCategoricalAccuracy()}

    # build model
    model = build_model(model_name=args.model, n_classes=len(class_names))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classifier = Classifier(
        model=model,
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metric_dict=metrics,
        args=args,
    )

    # train model
    classifier.train(
        training_data=ds_train,
        validation_data=ds_val,
        class_weights=class_weights,
    )

    # test
    classifier.score(ds_test)

if __name__ == '__main__':
    from config import get_user_args
    main(get_user_args())
