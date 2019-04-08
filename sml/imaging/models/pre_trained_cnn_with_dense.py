from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model
from keras.utils import plot_model

from .abstract_model import AbstractModel

BASE_MODELS = {
    'Xception': Xception,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'NASNetMobile': NASNetMobile,
    'NASNetLarge': NASNetLarge
}


class PreTrained(AbstractModel):

    def __init__(self, branch_model_identifier, image_shape, optimizer, loss,
                 metrics, dense_layer_configs=None):
        self.branch_model_identifier = branch_model_identifier
        self.image_shape = image_shape
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.dense_layer_configs = dense_layer_configs
        self.model = None

        self._branch_model = None
        self._head_model = None

    def _make_branch_model(self):
        if not self._branch_model:
            self._branch_model = BASE_MODELS[self.branch_model_identifier](
                weights='imagenet',
                include_top=False,
                input_shape=self.image_shape)
        return self._branch_model

    def _make_head_model(self, branch_output):
        if self.dense_layer_configs:
            x = GlobalAveragePooling2D()(branch_output)
            for dense_layer_config in self.dense_layer_configs:
                x = Dense(dense_layer_config['nodes'])(x)
                x = Dropout(dense_layer_config['dropout_prob'])(x)
                x = Activation(dense_layer_config['activation'])(x)
        else:
            x = branch_output

        return x

    def make_model(self):
        branch_model = self._make_branch_model()

        x = branch_model.output
        x = self._make_head_model(x)
        self.model = Model(inputs=branch_model.input, outputs=x)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        return self.model


if __name__ == '__main__':
    model = PreTrained(branch_model_identifier='InceptionV3',
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
    model.make_model()
    model.model.summary()
    plot_model(model.model, to_file='pre_trained_dense_model.png')
