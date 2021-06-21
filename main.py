from loader import prepare_data
from model import build_model
import tensorflow as tf
import tensorflow_addons as tfa
from pandas import DataFrame
import numpy as np
from pathlib import Path
from time import time
import pickle as pickle
from sklearn.neighbors import NearestCentroid


class Classifier:
    def __init__(self, model, optimizer, loss, metric_dict=None,
                 class_weights=None, class_names=None, logdir='logs'):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.metric_dict = metric_dict
        self.class_names = class_names
        self.logdir = Path(logdir)
        self.min_val_loss = np.inf
        self.patience = 5
        self.remaining_patience = self.patience
        self.log = self.logdir / 'log.txt'

    def report(self, results_dict, title=None, write_to_log=True):
        s = f"\n{title}\n"
        s += '\n'.join(f'{k}: {v}' for k, v in results_dict.items())
        print(s)
        if write_to_log:
            if isinstance(write_to_log, str) or isinstance(write_to_log, Path):
                file_path = Path(write_to_log)
            else:
                file_path = self.log
            with file_path.open('a+') as f:
                f.write(s)

    def update_metrics(self, y_true, y_pred):
        for metric_name, metric in self.metric_dict.items():
            if metric_name == 'Confusion Matrix':
                metric.update_state(tf.one_hot(y_true, metric.num_classes),
                                    tf.one_hot(y_pred, metric.num_classes))
            else:
                metric.update_state(y_true, y_pred)

    def get_metric_results(self, reset=False):
        results = {}
        for metric_name, metric in self.metric_dict.items():
            results[metric_name] = metric.result()

        if reset:
            self.reset_metrics()
        return results

    def reset_metrics(self):
        for metric in self.metric_dict.values():
            metric.reset_states()

    def train_step(self, x_batch, y_batch, sample_weights=None,
                   update_metrics=True):
        with tf.GradientTape() as tape:
            # Note: to check trainable weights, self.model.trainable_weights
            logits = self.model(x_batch, training=True)
            loss_value = self.loss(y_batch, logits,
                                   sample_weight=sample_weights)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        if np.isnan(loss_value.numpy()):
            from IPython import embed; embed()  ### DEBUG
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        if update_metrics:
            self.update_metrics(y_batch, tf.argmax(logits, 1))
        return loss_value

    def nearest_centroid_accuracy(self, batches, reset_before=True, reset_after=True):
        if reset_before:
            self.reset_metrics()

        xs, ys = zip(*[(self.model(x, training=False).numpy(), y.numpy())
                       for x, y in batches])
        xs, ys = np.concatenate(xs, 0), np.concatenate(ys, 0)

        # if np.isnan(xs).any():
        #     x, y = next(iter(batches))
        #     f = self.model(x, training=False)
        #     return np.nan

        n = len(xs)//2
        nc_classifier = NearestCentroid()
        nc_classifier.fit(xs[:n], ys[:n])
        accuracy = nc_classifier.score(xs[n:], ys[n:])

        return accuracy

    def compute_val_loss(self, batches):
        loss = 0.
        num_samples = 0
        for x_batch, y_batch in batches:
            logits = self.model(x_batch, training=False)
            loss += self.loss(y_batch, logits)
            num_samples += x_batch.shape[0]
        loss /= num_samples
        return loss

    def score(self, batches, reset_before=True, reset_after=True):
        if reset_before:
            self.reset_metrics()

        # evaluate
        n_classes = self.model.output_shape[-1]
        confusion_matrix = tf.zeros((n_classes, n_classes), dtype=tf.int32)
        loss = 0.
        num_samples = 0
        for x_batch, y_batch in batches:
            logits = self.model(x_batch, training=False)
            predictions = tf.argmax(logits, 1)
            loss += self.loss(y_batch, logits)
            confusion_matrix = confusion_matrix + tf.math.confusion_matrix(
                y_batch, predictions, num_classes=n_classes)
            self.update_metrics(y_batch, predictions)
            num_samples += x_batch.shape[0]
        loss /= num_samples
        metric_results = self.get_metric_results(reset=reset_after)

        confusion_matrix = DataFrame(confusion_matrix.numpy(),
                                     columns=self.class_names,
                                     index=self.class_names)
        return metric_results, loss, confusion_matrix

    def compute_sample_weights(self, y_true):
        # https://github.com/tensorflow/tensorflow/issues/10021
        if self.class_weights is None:
            return None
        return tf.gather(self.class_weights, y_true)

    def train(self, training_data, validation_data=None, epochs=1, triplet_loss=False):
        atime = time()
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            epoch_train_loss = 0
            for step, (x_train_batch, y_train_batch) in enumerate(training_data):
                epoch_train_loss += self.train_step(
                    x_batch=x_train_batch,
                    y_batch=y_train_batch,
                    sample_weights=self.compute_sample_weights(y_train_batch)
                )

            train_results = self.get_metric_results(reset=True)
            train_results.update({'Loss': epoch_train_loss})
            elapsed, atime = time() - atime, time()
            self.report(train_results,
                        "Epoch {} ({:.3f} s) Training Results"
                        "".format(epoch, atime))

            if validation_data is not None:
                if triplet_loss:
                    val_acc = self.nearest_centroid_accuracy(validation_data)
                    val_loss = self.compute_val_loss(validation_data)
                    val_results = {"Validation Accuracy": val_acc,
                                   "Validation Loss": val_loss}
                else:
                    val_results, val_loss, val_cm = self.score(validation_data)
                    val_acc = np.trace(val_cm) / np.array(val_cm).sum()
                    val_results.update({'Validation Loss': val_loss,
                                        'Confusion Matrix': f'\n{val_cm}',
                                        'Accuracy': val_acc})
                    elapsed, atime = time() - atime, time()
                self.report(val_results,
                            "Epoch {} ({:.3f} s) Validation Results"
                            "".format(epoch, atime))

                # early stopping and model saving
                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    self.remaining_patience = self.patience
                    if val_acc > 0.8:
                        self.model.save(Path(self.logdir, 'model.h5'))
                else:
                    if self.remaining_patience == 0:
                        break  # stop early
                    self.remaining_patience -= 1

# TODO: See batch norm todo above (or maybe i want them to train)
# TODO: check that implementation of class weights doesn't have softmax issue
# TODO: finish adding the rest of the tf hub models
# TODO: build a grid search tool (that goes through models, lr, etc.)


def train_and_test(args):
    start_time = time()

    # create logdir and record args (in both txt and pickle format)
    args.logdir.mkdir(parents=True)
    with Path(args.logdir, 'train_args.txt').open('a+') as f:
        f.write('\n'.join(f'{k}:{v}' for k, v in vars(args).items()))
    with Path(args.logdir, 'train_args.p').open('wb') as f:
        pickle.dump(args, f)

    # prep data
    ds_train, ds_val, ds_test, class_names, label_counts = prepare_data(args)
    # ds_val = ds_val.prefetch()

    # save class names
    with Path(args.logdir, 'class_names.txt').open('w') as f:
        f.write(','.join(class_names))

    # set class weights to compensate for class imbalance
    class_weights = None
    if not (args.no_class_weights or args.triplet_loss):
        print(f"\nTrain Label Counts\n{label_counts}\n")
        class_weights = [1/c for c in label_counts.values()]

    # define metrics
    metrics = {
        # 'Accuracy': tf.keras.metrics.Accuracy,
        # 'BinaryAccuracy': tf.keras.metrics.BinaryAccuracy,
        # 'CategoricalAccuracy': tf.keras.metrics.CategoricalAccuracy,
    }

    # build model
    num_dense_outputs = args.tl_dims if args.triplet_loss else len(class_names)
    model = build_model(model_name=args.model, n_classes=num_dense_outputs,
                        input_dimensions=args.image_dimensions,
                        is_embedding=args.triplet_loss)
    if args.triplet_loss:
        # loss = tfa.losses.TripletSemiHardLoss()
        # raise Exception("ANDY: Note that this loss function is causing NaNs")
        loss = tfa.losses.TripletHardLoss(margin=args.tl_margin, soft=args.tl_soft)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    classifier = Classifier(
        model=model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        class_weights=class_weights,
        class_names=class_names,
        metric_dict=metrics,
        logdir=args.logdir,
    )

    # train model
    # with tf.profiler.experimental.Profile(str(args.logdir)):
    classifier.train(
        training_data=ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        triplet_loss=args.triplet_loss,
    )

    # test
    best_model_path = Path(args.logdir, 'model.h5')
    try:
        classifier.model.load_weights(str(best_model_path))
    except (FileNotFoundError, OSError):
        print(f"Warning:  "
              f"Best model weights not found, this is to be expected if "
              f"the model never reached 80% validation accuracy.  "
              f"Path checked: {best_model_path}")

    if args.triplet_loss:
        test_acc = classifier.nearest_centroid_accuracy(ds_test)
        test_loss = classifier.compute_val_loss(ds_test)
        test_results = {"Validation Accuracy": test_acc,
                        "Validation Loss": test_loss}
    else:
        test_results, test_loss, test_cm = classifier.score(ds_test)
        test_acc = np.trace(test_cm) / np.array(test_cm).sum()
        test_results.update({'Test Loss': test_loss,
                             'Test Confusion Matrix': f'\n{test_cm}',
                             'Test Accuracy': test_acc,
                             'Total Train+Test Time': time() - start_time})
    classifier.report(test_results, "Test Results")


def main(args):
    if args.benchmark_input:
        from loader import benchmark_input
        benchmark_input(args)
    elif args.test_load:
        from loader import load_test
        load_test(args)
    else:
        train_and_test(args)


if __name__ == '__main__':
    from config import get_user_args
    main(get_user_args())
