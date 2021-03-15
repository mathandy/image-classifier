import tensorflow as tf
import tensorflow_hub as hub
from available_tf_hub_models import tf_hub_model_urls, tf_hub_model_input_size


# BATCH_NORM_MOMENTUM = 0.99  # 0.997 is a choice hinted at by tfhub

def get_expected_image_shape(model_path, model_name):

    try:
        module_spec = hub.load_module_spec(model_path)
        input_dimensions = hub.get_expected_image_size(module_spec)
    except Exception as e:
        print(f"\nFailed to get module spec from TF Hub.\n\n{e}\n")
        input_size = tf_hub_model_input_size[model_name]
        input_dimensions = (input_size, input_size)
        print(f"Using input dimensions {input_dimensions}\n")

    return input_dimensions


def get_pretrained_featurizer(model_name, input_dimensions=None,
                              input_channels=3):
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input

    model_path = model_name
    if not model_path[:4] == 'http':
        model_path = tf_hub_model_urls[model_name]

    # get image shape
    if input_dimensions is None:
        input_dimensions = get_expected_image_shape(model_path, model_name)
    image_shape = list(input_dimensions) + [input_channels]

    # load headless pretrained model as keras layer
    headless_pretrained_base = hub.KerasLayer(
        model_path,
        trainable=False,
        # arguments=dict(batch_norm_momentum=batch_norm_momentum),\
        input_shape=image_shape)
    return headless_pretrained_base, image_shape


def build_model(model_name, n_classes, input_dimensions=None, input_channels=3,
                headless=False):
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input

    # load headless pretrained model as keras layer
    headless_pretrained_base, image_shape = get_pretrained_featurizer(
        model_name=model_name,
        input_dimensions=input_dimensions,
        input_channels=input_channels
    )

    # build classifier model
    if headless:
        model = headless_pretrained_base
    else:
        model = tf.keras.Sequential([
            headless_pretrained_base,
            tf.keras.layers.Dense(n_classes)
        ])
    model.build(image_shape)
    return model