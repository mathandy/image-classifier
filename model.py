import tensorflow as tf
import tensorflow_hub as hub


BATCH_NORM_MOMENTUM = 0.99  # 0.997 is a choice hinted at by tfhub
N_CLASSES = 5


def build_model(model_path, n_classes, input_dimensions=None, input_channels=3):
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input
    model = tf.keras.Sequential([
        hub.KerasLayer(
            model_path,
            trainable=True,
            arguments=dict(batch_norm_momentum=BATCH_NORM_MOMENTUM),
            output_shape=N_CLASSES)
    ])

    if input_dimensions is None:
        module_spec = hub.load_module_spec(model_path)
        input_dimensions = hub.get_expected_image_size(module_spec)

    model.build([None] + list(input_dimensions) + [input_channels])
    return model





