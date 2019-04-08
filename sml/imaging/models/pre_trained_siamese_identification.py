from keras.layers import Input, Lambda, GlobalAveragePooling2D, Concatenate, \
    Reshape, Conv2D, Flatten
from keras.models import Model
import keras.backend as K
from keras.utils import plot_model

# from .pre_trained_cnn_with_dense import PreTrained
from sml.imaging.models.pre_trained_cnn_with_dense import PreTrained


class PreTrainedSiameseIdentification(PreTrained):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch_model = None
        self.head_model = None

    def _make_head_model(self, _=None):
        if not self._head_model:
            assert self._branch_model, 'expected self._branch model to exist'
            nbr_kernels = 32

            # define input
            input_a = Input(shape=self._branch_model.output_shape[1:])
            input_condense_a = GlobalAveragePooling2D()(input_a)

            input_b = Input(shape=self._branch_model.output_shape[1:])
            input_condense_b = GlobalAveragePooling2D()(input_b)

            # extract features
            x_1 = Lambda(lambda _x: _x[0] * _x[1])(
                [input_condense_a, input_condense_b])
            x_2 = Lambda(lambda _x: _x[0] + _x[1])(
                [input_condense_a, input_condense_b])
            x_3 = Lambda(lambda _x: K.abs(_x[0] - _x[1]))(
                [input_condense_a, input_condense_b])
            x_4 = Lambda(lambda _x: K.square(_x))(x_3)

            x = Concatenate()([x_1, x_2, x_3, x_4])
            x = Reshape(
                (4, self._branch_model.output_shape[3], 1), name='reshape1')(x)

            # filter features
            x = Conv2D(
                nbr_kernels, (4, 1), activation='relu', padding='valid')(x)
            x = Reshape(
                (self._branch_model.output_shape[3], nbr_kernels, 1))(x)
            x = Conv2D(
                1, (1, nbr_kernels), activation='linear', padding='valid')(x)
            x = Flatten(name='flatten')(x)

            self._head_model = Model(
                inputs=[input_a, input_b], outputs=x, name='head')
        return self._head_model

    def make_model(self):
        self.branch_model = self._make_branch_model()
        self.head_model = self._make_head_model()

        img_a = Input(shape=self.image_shape)
        img_b = Input(shape=self.image_shape)

        x_a = self.branch_model(img_a)
        x_b = self.branch_model(img_b)

        x = self.head_model([x_a, x_b])
        self.model = Model([img_a, img_b], x)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return self.model


if __name__ == '__main__':
    MODEL = PreTrainedSiameseIdentification(
        branch_model_identifier='InceptionV3',
        image_shape=(100, 100, 3),
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc'],
        dense_layer_configs=[
            {
                'nodes': 1024,
                'dropout_prob': 0.5,
                'activation': 'relu'
            },
            {
                'nodes': 5005,
                'dropout_prob': 0.0,
                'activation': 'softmax'
            }
        ])
    MODEL.make_model()
    MODEL.model.summary()
    plot_model(MODEL.model, to_file='pre_trained_siamese_identification.png')
    plot_model(MODEL.branch_model,
               to_file='pre_trained_siamese_identification_branch.png')
    plot_model(MODEL.head_model,
               to_file='pre_trained_siamese_identification_head.png')
