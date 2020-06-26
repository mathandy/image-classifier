from loader import prepare_data
from model import build_model
import tensorflow as tf
# from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from pandas import DataFrame


class Classifier:
    def __init__(self, model, optimizer, loss, metric_dict=None,
                 class_weights=None, class_names=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.metric_dict = metric_dict
        self.class_names = class_names

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
            # TODO: figure out if this updates batch norm layers even
            # though they are marked not trainable
            # TO SEE, check self.model.trainable_weights
            logits = self.model(x_batch, training=True)
            loss_value = self.loss(y_batch, logits,
                                   sample_weight=sample_weights)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        if update_metrics:
            self.update_metrics(y_batch, tf.argmax(logits, 1))
        return loss_value

    def score(self, batches, reset_before=True, reset_after=True):
        if reset_before:
            self.reset_metrics()

        n_classes = self.model.output_shape[-1]
        confusion_matrix = tf.zeros((n_classes, n_classes), dtype=tf.int32)
        loss = 0.
        for x_batch, y_batch in batches:
            logits = self.model(x_batch, training=False)
            predictions = tf.argmax(logits, 1)
            loss += self.loss(y_batch, logits) / x_batch.shape[0]
            confusion_matrix = confusion_matrix + tf.math.confusion_matrix(
                y_batch, predictions, num_classes=n_classes)
            self.update_metrics(y_batch, predictions)
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

    def train(self, training_data, validation_data=None):
        epochs = 2
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
            print("\nTraining Results")
            print('\n'.join(f'{k}: {v}' for k, v in train_results.items()))

            if validation_data is not None:
                val_results, val_loss, val_cm = self.score(validation_data)
                val_results.update({"Val Loss": val_loss})
                print("\nValidation Results")
                print('\n'.join(f'{k}: {v}' for k, v in val_results.items()))
                print(f'Validation Confusion Matrix:\n{val_cm}')

# TODO: See batch norm todo above
# TODO: implement class weights
# TODO: finish adding the rest of the tf hub models
# TODO: build a grid search tool (that goes through models, lr, etc.)


def main(args):

    def fix_and_batch(ds):
        ds = ds.map(lambda image, label, file_path: (image, label))
        ds = ds.batch(args.batch_size)
        return ds

    # prepare data
    ds_train, ds_val, ds_test, class_names, label_counts = prepare_data(args)
    ds_train = fix_and_batch(ds_train)
    ds_val   = fix_and_batch(ds_val)
    ds_test  = fix_and_batch(ds_test) if args.test_dir is not None else None

    # set class weights to compensate for class imbalance
    class_weights = None
    if args.balance_class_weights:
        n_data = sum(c for c in label_counts.values())
        class_weights = [c/n_data for c in label_counts.values()]

    # define metrics
    metrics = {
        'Accuracy': tf.keras.metrics.CategoricalAccuracy(),
        # 'Confusion Matrix': MultiLabelConfusionMatrix(len(class_names)),
    }

    # build model
    model = build_model(model_name=args.model, n_classes=len(class_names))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classifier = Classifier(
        model=model,
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        class_weights=class_weights,
        class_names=class_names,
        metric_dict=metrics,
    )

    # train model
    classifier.train(
        training_data=ds_train,
        validation_data=ds_val,
    )

    # load_test
    if args.test_dir is not None:
        test_results, test_loss, test_cm = classifier.score(ds_test)
        test_results.update({"Val Loss": test_loss})
        print("\nValidation Results")
        print('\n'.join(f'{k}: {v}' for k, v in test_results.items()))
        print(f'Test Set Confusion Matrix:\n{test_cm}')


if __name__ == '__main__':
    from config import get_user_args
    main(get_user_args())
