from pathlib import Path
from available_tf_hub_models import tf_hub_model_input_size


def get_user_args():
    """Fetch model parameters (including from CLI flags)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', default='../bug-data/Multi_Inverts_Master', type=Path,
        help='Path to subdirectory-labeled image directory.'
    )
    parser.add_argument(
        '--val_part', default=0.1, type=float,
        help='Portion of training images to reserve for validation.'
    )
    parser.add_argument(
        '--test_dir', default=None,
        help='Path to subdirectory-labeled image directory for testing.'
    )
    parser.add_argument(
        '--batch_size', default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--image_dimensions', default=None, nargs=2, type=int,
        help='Resize all images to these dimensions (after augmentation).'
    )
    parser.add_argument(
        '--model', default='inception_v3',
        help='Name of TF Hub model to use.'
    )
    parser.add_argument(
        '--balance_class_weights', '-b', default=False, action='store_true',
        help='Use class weights to compensate for class imbalance.'
    )
    args = parser.parse_args()
    return process_args(args)


def process_args(args):

    if args.image_dimensions is None:
        sz = tf_hub_model_input_size[args.model]
        args.image_dimensions = (sz, sz)

    return args