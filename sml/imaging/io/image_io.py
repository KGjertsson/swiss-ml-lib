from pathlib import Path

from keras.preprocessing import image as kimage
import numpy as np


class ImageReader:
    """Assumes 2D data."""

    def __init__(self, data_dir, verbose=False):
        assert isinstance(data_dir, Path), \
            'expected argument \'data_dir\' of type \'Path\' but found ' \
            '\'{}\''.format(type(data_dir))
        self.data_dir = data_dir
        self.verbose = verbose

    @staticmethod
    def _calculate_mean_shape_of_rgb(image_files):
        image_shapes = [kimage.img_to_array(kimage.load_img(image_file)).shape
                        for image_file in image_files]
        mean_rows = np.mean([image_shape[0] for image_shape in image_shapes])
        mean_cols = np.mean([image_shape[1] for image_shape in image_shapes])
        mean_chs = np.mean([image_shape[2] for image_shape in image_shapes])

        return mean_rows, mean_cols, mean_chs

    def load_as_numpy_array(self, image_file_names=None, image_shape=None,
                            transforms=()):
        # read specific files in specific order if desired,
        # otherwise read all images in arbitrary order
        if image_file_names is not None:
            image_files = [self.data_dir / image_file
                           for image_file in image_file_names]
        else:
            image_files = [self.data_dir / image_file
                           for image_file in self.data_dir.glob('*')]

        if image_shape is None:
            if self.verbose:
                print('calculating image shape...')
            image_shape = self._calculate_mean_shape_of_rgb(image_files)

        # unable to do np.zeros(shape) directly due to perceived memory error
        images = list()
        for image_file in image_files:
            image = kimage.load_img(image_file, target_size=image_shape)
            image = kimage.img_to_array(image)
            for transform_fun, transform_kwargs in transforms:
                image = transform_fun(image, **transform_kwargs)
            images.append(image)

        return np.asarray(images)
