import tensorflow as tf
import tensorflow_hub as hub
from available_tf_hub_models import tf_hub_model_urls, tf_hub_model_input_size


# BATCH_NORM_MOMENTUM = 0.99  # 0.997 is a choice hinted at by tfhub
VERBOSITY = 0


def get_expected_image_shape(model_path, model_name):
    model_path = hub.resolve(model_path)
    try:
        module_spec = hub.load_module_spec(model_path)
        input_dimensions = hub.get_expected_image_size(module_spec)
    except Exception as e:
        if VERBOSITY:
            print(f"\nFailed to get module spec from TF Hub.\n\n{e}\n")
        input_size = tf_hub_model_input_size[model_name]
        input_dimensions = (input_size, input_size)
        if VERBOSITY:
            print(f"Using input dimensions {input_dimensions}\n")

    return input_dimensions


def get_pretrained_featurizer(model_name, input_dimensions=None,
                              input_channels=3, trainable=False):
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input

    model_path = model_name
    if not model_path[:4] == 'http':
        model_path = tf_hub_model_urls[model_name]
    model_path = hub.resolve(model_path)

    # get image shape
    if input_dimensions is None:
        input_dimensions = get_expected_image_shape(model_path, model_name)
    h, w = input_dimensions
    image_shape = [None, h, w, input_channels]

    # load headless pretrained model as keras layer
    headless_pretrained_base = hub.KerasLayer(model_path, trainable=trainable)
    return headless_pretrained_base, image_shape


def get_model(model_name, n_classes, input_dimensions=None, input_channels=3,
                headless=False, is_embedding=False, trainable=False):
    assert not (headless and is_embedding)
    # see the common image input conventions
    # https://www.tensorflow.org/hub/common_signatures/images#input

    # load headless pretrained model as keras layer
    headless_pretrained_base, image_shape = get_pretrained_featurizer(
        model_name=model_name,
        input_dimensions=input_dimensions,
        input_channels=input_channels,
        trainable=trainable,
    )

    # build classifier model
    if headless:
        model = headless_pretrained_base
    elif is_embedding:
        model = tf.keras.Sequential([
            headless_pretrained_base,
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(n_classes),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
        ])
    else:
        model = tf.keras.Sequential([
            headless_pretrained_base,
            tf.keras.layers.Dense(n_classes)
        ])
    return model, image_shape


def build_model(model_name, n_classes, input_dimensions=None, input_channels=3,
                headless=False, is_embedding=False, trainable=False):
    model, image_shape = get_model(model_name=model_name,
                                   n_classes=n_classes,
                                   input_dimensions=input_dimensions,
                                   input_channels=input_channels,
                                   headless=headless,
                                   is_embedding=is_embedding,
                                   trainable=trainable)

    model.build(image_shape)
    return model


if __name__ == '__main__':
    from available_tf_hub_models import tf_hub_models
    from tempfile import gettempdir
    from pathlib import Path
    from time import time
    VERBOSITY = 0
    input_dims = None
    input_channels = 3
    n_classes = 1
    trainable = True
    log_path = Path(gettempdir()) / f"image_classifier_fails_{str(time()).replace('.', '')}.txt"

    fails = []
    for name, url, input_size in tf_hub_models:
        try:
            model = build_model(model_name=name,
                                n_classes=n_classes,
                                input_dimensions=input_dims,
                                input_channels=input_channels,
                                trainable=trainable)
            print(f'success ({name}).')
            del model
        except Exception as e:
            print(f'FAIL ({name}): {e}')
            fails.append(name)
            with log_path.open('a') as f:
                f.write(f'{name}\n')

    if fails:
        print("The following models weren't fetched or built successfully.")
        for m in fails:
            print(m)

