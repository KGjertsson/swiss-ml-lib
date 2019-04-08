from ..encodings import SmlLabelEncoder
from ...io.image_io import ImageReader


class Image2DPipeline:

    def __init__(self, data_dir, image_information, image_shape=None,
                 verbose=False, transforms=()):
        self.data_dir = data_dir
        self.image_information = image_information
        self.verbose = verbose
        self.image_shape = image_shape
        self.transforms = transforms

    def load_images_with_annotations(self):
        # initialize image reader
        image_reader = ImageReader(self.data_dir, self.verbose)

        # load images and labels, encode str labels to one hot encoded format
        images = image_reader.load_as_numpy_array(
            self.image_information['Image'].values, self.image_shape,
            self.transforms)
        labels, label_encoder = SmlLabelEncoder().str_categories_to_one_hot(
            self.image_information['Id'].values)

        # apply transforms
        return images, labels, label_encoder
