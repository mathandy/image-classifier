import tensorflow as tf
from loader import get_image_filepaths, load_png, load_jpeg
from model import build_model
from pathlib import Path


def parse_cli_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path, help='directory of images')
    parser.add_argument('output_dir', type=Path, help='where to store output')
    parser.add_argument(
        '--image_dimensions', default=None, nargs=2, type=int,
        help='Resize all images to these dimensions (after augmentation).'
    )
    parser.add_argument(
        '--model', '-m', default='BiT-M-R101x1',
        help='Name of TF Hub model to use.'
    )
    # parser.add_argument(
    #     '--grayscale', default=False, action='store_true',
    #     help='Input images are grayscale.'
    # )
    parser.add_argument(
        '--png', default=False, action='store_true',
        help='Input images are PNGs (otherwise assumes JPEGs).'
    )
    return parser.parse_args()


def subdir_labelled_filepath_to_label(fp):
    return Path(fp).parent.name


def extract_features(args):

    # create output_dir and record args (in both txt and pickle format)
    args.output_dir.mkdir(parents=True)
    with Path(args.output_dir, 'args.txt').open('a+') as f:
        f.write('\n'.join(f'{k}:{v}' for k, v in vars(args).items()))

    # prep data
    file_paths = get_image_filepaths(args.image_dir, args.png)
    labels = [subdir_labelled_filepath_to_label(fp) for fp in file_paths]
    class_names = sorted(list(set(labels)))
    encoder = dict((n, i) for i, n in enumerate(class_names))
    encoded_labels = [encoder[l] for l in labels]
    assert len(file_paths) > 1

    # create label key
    with (args.output_dir / 'label_key.txt').open('w') as f:
        f.write('NAME,LABEL,COUNT\n')
        for name in class_names:
            count = labels.count(name)
            encoding = encoder[name]
            f.write(f'{name},{encoding},{count}\n')

    # build model
    model = build_model(model_name=args.model, n_classes=len(class_names),
                        input_dimensions=args.image_dimensions,
                        headless=True)

    load_image = load_png if args.png else load_jpeg
    with (args.output_dir / 'feature_vectors.csv').open('w+') as fv_out:
        with (args.output_dir / 'labels.csv').open('w+') as lbl_out:
            for i, (fp, l) in enumerate(zip(file_paths, encoded_labels)):
                image = load_image(fp)
                resized_image = tf.image.resize(image, args.image_dimensions)
                features = model(tf.expand_dims(resized_image, 0))
                fv_out.write(','.join(str(x) for x in features.numpy().ravel()) + '\n')
                lbl_out.write(f'{l},{fp}\n')

                if i % 100 == 0:
                    print(f'working on {i}/{len(file_paths)}')


if __name__ == '__main__':
    extract_features(parse_cli_args())
