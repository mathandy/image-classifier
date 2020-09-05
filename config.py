from pathlib import Path
from available_tf_hub_models import tf_hub_model_input_size
from time import time


def get_user_args():
    """Fetch model parameters (including from CLI flags)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', '-i', default=Path('../bug-data/Multi_Inverts_Master'),
        type=Path,
        help='Path to subdirectory-labeled image directory.'
    )
    parser.add_argument(
        '--grayscale', default=False, action='store_true',
        help='Input images are grayscale.'
    )
    parser.add_argument(
        '--png', default=False, action='store_true',
        help='Input images are PNGs (otherwise assumes JPEGs).'
    )
    parser.add_argument(
        '--val_dir', default=None, type=Path,
        help='Path to subdirectory-labeled image directory for validation.'
             'Alternatively, use the val_part argument.'
    )
    parser.add_argument(
        '--test_dir', default=None, type=Path,
        help='Path to subdirectory-labeled image directory for testing.'
             'Alternatively, use the test_part argument.'
    )
    parser.add_argument(
        '--val_part', default=0.1, type=float,
        help='Portion of training images to reserve for validation.'
             'Alternatively, use the val_dir argument.'
    )
    parser.add_argument(
        '--test_part', default=0.1, type=float,
        help='Portion of training images to reserve for the final test. '
             'Alternatively, use the test_dir argument.'
    )
    parser.add_argument(
        '--logdir', '-o', type=Path, default=None,  # default set below
        help='Where to store output.'
    )
    parser.add_argument(
        '--run_name', '-n', type=Path, default=None,  # default set below
        help='Name of TF Hub model to use.'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='Batch size.'
    )
    parser.add_argument(
        '--epochs', default=1000, type=int,
        help='Number of epochs to train for.'
    )
    parser.add_argument(
        '--image_dimensions', default=None, nargs=2, type=int,
        help='Resize all images to these dimensions (after augmentation).'
    )
    parser.add_argument(
        '--model', '-m', default='BiT-M-R101x1',
        help='Name of TF Hub model to use.'
    )
    parser.add_argument(
        '--no_class_weights', default=False, action='store_true',
        help='Do not use class weights to compensate for class imbalance.'
    )
    parser.add_argument(
        '--learning_rate', '-r', default=0.001, type=float,
        help='Name of TF Hub model to use.'
    )
    parser.add_argument(
        '--benchmark_input', default=False, action='store_true',
        help='Benchmark input pipeline.'
    )
    parser.add_argument(
        '--test_load', default=False, action='store_true',
        help='Show images (with augmentations).'
    )
    args = parser.parse_args()
    return process_args(args)


def process_args(args):

    is_test = args.test_load or args.benchmark_input

    # image dimensions
    if args.image_dimensions is None:
        sz = tf_hub_model_input_size[args.model]
        args.image_dimensions = (sz, sz)

    # logdir and run_name
    assert args.logdir is None or args.run_name is None or is_test
    if not is_test and args.logdir is None:
        if args.run_name is None:
            args.run_name = str(time()).replace('.', '-')
        else:
            from warnings import warn
            warn("Using temp dir for named run.")
        args.logdir = Path('..', 'classifier-logs', args.run_name)

    # train-val-test split parameters
    if args.test_dir is not None:
        args.test_part = 0
    if args.val_dir is not None:
        args.val_dir = 0
    return args
