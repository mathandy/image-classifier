from pathlib import Path
from available_tf_hub_models import tf_hub_model_input_size
from time import time
from tempfile import gettempdir
from augmenter import augmentation_choices


def get_user_args():
    """Fetch model parameters (including from CLI flags)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', default=None,
        type=Path,
        help='Path to subdirectory-labeled image directory (all images, '
             'no test/train/val split, use --split_image_dir for that case).'
    )
    parser.add_argument(
        '--split_image_dir', '-i', default=None,
        type=Path,
        help='Path to subdirectory-labeled image directory split into '
             'train, test, and val dirs.'
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
        '--batch_size', '-b', default=32, type=int,
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
        '--model', '-m', default='BiT-M-R50x1',
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
    parser.add_argument(
        '--triplet_loss', '-t', action='store_true',
        help='Use triplet loss instead of cross-entropy.'
    )
    parser.add_argument(
        '--tl_dims', default=256,
        help='The triplet loss embedding dimensionality.  Only '
             'applicable if --triplet_loss flag used.'
    )
    parser.add_argument(
        '--tl_margin', default=0.2, type=float,
        help='The triplet loss margin.  Only '
             'applicable if --triplet_loss flag used.'
    )
    parser.add_argument(
        '--tl_soft', default=False, action='store_true',
        help='Use soft triplet loss margin.  Only '
             'applicable if --triplet_loss flag used.'
    )
    parser.add_argument(
        '--standardize', action='store_true', default=False,
        help='Standardize each image to have mean 0 and variance 1.'
    )
    parser.add_argument(
        '--augmentation', '-a', default='strong', choices=augmentation_choices,
        help="Which augmentation pipeline to use.\nChoices:{'\n'.join(augmentation_choices}."
    )
    args = parser.parse_args()
    return process_args(args)


def process_args(args):

    is_test = args.test_load or args.benchmark_input

    assert bool(args.image_dir) ^ bool(args.split_image_dir)
    if args.split_image_dir:
        train_dir = args.split_image_dir / 'train'
        assert train_dir.exists()
        args.image_dir = train_dir
        val_dir = args.split_image_dir / 'val'
        test_dir = args.split_image_dir / 'test'
        if not (val_dir.exists() or test_dir.exists()):
            raise FileNotFoundError(
                "When using --split_image_dir, a test or val directory "
                "must exists.")
        if val_dir.exists():
            args.val_dir = val_dir
        if test_dir.exists():
            args.test_dir = test_dir

    # image dimensions
    if args.image_dimensions is None:
        sz = tf_hub_model_input_size[args.model]
        args.image_dimensions = (sz, sz)

    # logdir and run_name
    assert args.logdir is None or args.run_name is None or is_test
    if args.logdir is None and not is_test:
        if args.run_name is None:
            args.run_name = str(time()).replace('.', '-')
        args.logdir = Path(gettempdir()) / 'classifier-logs' / args.run_name
        print(f"\n\nWARNING: Storing log in temp dir: {args.logdir}\n")

    # train-val-test split parameters
    if args.test_dir is not None:
        args.test_part = 0
    if args.val_dir is not None:
        args.val_part = 0
    return args
