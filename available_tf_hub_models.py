"""
This module creates a list of named tuples `tf_hub_models` with the
available pretrained model name, url, and the input size it was trained
with.

Note this isn't currently all models (though does contain all
fine-tunable TF2 "imagenet/mobilenet_v2" variants as of Jun 17, 2020).
"""

from collections import namedtuple


# pretrained mobile-net available in many width-multiplier and input size
_mobilnet_models = [
    (140, 224),
    (130, 224),
    (100, 224),
    (75, 224),
    (50, 224),
    (35, 224),

    (100, 192),
    (75, 192),
    (50, 192),
    (35, 192),

    (100, 160),
    (75, 160),
    (50, 160),
    (35, 160),

    (100, 128),
    (75, 128),
    (50, 128),
    (35, 128),

    (100, 96),
    (75, 96),
    (35, 96),
]

TFHubModel = namedtuple('TF_Hub_Model', ['name', 'url', 'input_size'])


def tf_hub_model(name, input_size, url=None):
    if url is None:
        url = get_hub_url(model_name)
    return TFHubModel(name, url, input_size)


def get_hub_url(model_name):
    return f"https://tfhub.dev/google/imagenet/{model_name}/feature_vector/4"


# add all mobilenet models to list
tf_hub_models = []
for width_multiplier, input_size in _mobilnet_models:
    model_name = "mobilenet_v2_%03d_%d" % (width_multiplier, input_size)
    tf_hub_models.append(tf_hub_model(model_name, input_size))

tf_hub_models += [
    tf_hub_model("inception_v3", 96),  # online claims this is 299

    tf_hub_model("resnet_v2_50", 96),
    tf_hub_model("resnet_v2_101", 224),
    tf_hub_model("resnet_v2_152", 224),

    tf_hub_model("pnasnet_large", 331),
    tf_hub_model("nasnet_large", 331),
    tf_hub_model("resnet_v2_152", 224),
]

# add efficientnet models
# note using 114 instead of 224 to avoid OOM error
def enet_url(i):
    return f'https://tfhub.dev/tensorflow/efficientnet/b{i}/feature-vector/1'
tf_hub_models += \
    [tf_hub_model(f'EfficientNet-B{i}', 224, enet_url(i)) for i in range(8)]

# add BiT-S (ImageNet-1k) and BiT-M (ImageNet-21k) models (expected size unkown)
tf_hub_models += [
    tf_hub_model('BiT-S-R50x1', 224, 'https://tfhub.dev/google/bit/s-r50x1/1'),
    tf_hub_model('BiT-S-R50x3', 224, 'https://tfhub.dev/google/bit/s-r50x3/1'),
    tf_hub_model('BiT-S-R101x1', 224, 'https://tfhub.dev/google/bit/s-r101x1/1'),
    tf_hub_model('BiT-S-R101x3', 224, 'https://tfhub.dev/google/bit/s-r101x3/1'),
    tf_hub_model('BiT-S-R152x4', 224, 'https://tfhub.dev/google/bit/s-r152x4/1'),

    tf_hub_model('BiT-M-R50x1', 224, 'https://tfhub.dev/google/bit/m-r50x1/1'),
    tf_hub_model('BiT-M-R50x3', 224, 'https://tfhub.dev/google/bit/m-r50x3/1'),
    tf_hub_model('BiT-M-R101x1', 224, 'https://tfhub.dev/google/bit/m-r101x1/1'),
    tf_hub_model('BiT-M-R101x3', 224, 'https://tfhub.dev/google/bit/m-r101x3/1'),
    tf_hub_model('BiT-M-R152x4', 224, 'https://tfhub.dev/google/bit/m-r152x4/1'),

    tf_hub_model('BiT-M-R50x1-invertebrate', 224,
                 'https://tfhub.dev/google/experts/bit/r50x1/in21k/invertebrate/1'),

]

# make some convenient dictionaries
tf_hub_model_urls = dict((name, url) for name, url, _ in tf_hub_models)
tf_hub_model_input_size = dict((name, sz) for name, _, sz in tf_hub_models)


if __name__ == '__main__':
    for x in tf_hub_models:
        print(x.name)
