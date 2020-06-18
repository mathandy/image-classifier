from pathlib import Path


def get_user_args():
    """Fetch model parameters (including from CLI flags)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', default='../bug-data/BC_Night', type=Path,
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
        '--image_dimensions', default=(150, 150), nargs=2,
        help='Resize all images to these dimensions (after augmentation).'
    )
    args = parser.parse_args()
    return args
