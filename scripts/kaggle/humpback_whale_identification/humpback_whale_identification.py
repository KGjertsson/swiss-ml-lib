from pathlib import Path

import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
from skimage import color

from sml.imaging.models.pre_trained_siamese_identification import \
    PreTrainedSiameseIdentification
from sml.imaging.data.piplines.image_2d_pipeline import Image2DPipeline


# TODO: 1. Add affine transforms

def main():
    # define variables
    train_image_dir = Path(
        '~/git/swiss-ml-lib/data/humpback_whale_identification/train'
    ).expanduser()
    # test_image_dir = Path('data/test')
    train_annotations_file = Path(
        '~/git/swiss-ml-lib/data/humpback_whale_identification/train.csv'
    ).expanduser()
    # n_images = 25361
    n_classes = 5005
    model_type = 'Xception'
    desired_image_shape = (128, 128, 3)

    # define transformations to apply to images before training,
    # current transforms are as follows:
    #   1. transform image to gray-scale from rgb
    #   2. pre process image according to keras standards
    transforms = [
        (color.rgb2gray, {}),
        (preprocess_input, {'mode': 'tf'})
    ]

    # load train annotations
    annotations = pd.read_csv(train_annotations_file)
    annotations = annotations.head()

    # drop new whale
    annotations = annotations[annotations['Id'] != 'new_whale']

    # load and pre-process image data
    pipeline = Image2DPipeline(
        train_image_dir,
        annotations,
        desired_image_shape,
        verbose=True,
        transforms=transforms)
    train_images, train_labels, label_encoder = \
        pipeline.load_images_with_annotations()

    # define keras model
    keras_model = PreTrainedSiameseIdentification(
        branch_model_identifier='Xception',
        image_shape=desired_image_shape,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy', 'acc'],
        dense_layer_configs=[
            {
                'nodes': 1024,
                'dropout_prob': 0.5,
                'activation': 'relu'
            },
            {
                'nodes': n_classes,
                'dropout_prob': 0.0,
                'activation': 'softmax'
            }
        ]).make_model()

    history = keras_model.fit(
        train_images,
        train_labels,
        epochs=100,
        validation_split=0.0,
        verbose=1)
    keras_model.save(model_type + '_keras_model')

    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('history')

    # test = os.listdir(test_image_dir)
    # col = ['Image']
    # test_df = pd.DataFrame(test, columns=col)
    #
    # test_images = image_io.load_images(test_image_dir, test_df['Image'].values, desired_image_shape)
    # predictions = keras_model.predict(np.array(test_images), verbose=1)
    #
    # for i, pred in enumerate(predictions):
    #     test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
    #
    # test_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
