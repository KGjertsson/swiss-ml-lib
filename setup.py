from distutils.core import setup

INSTALL_REQUIRES = ['numpy', 'keras', 'matplotlib', 'scikit-image', 'pydot',
                    'boto', 'pandas', 'xgboost',  'lightgbm', 'opencv-python']

# install with conda:
# conda install pytorch torchvision -c pytorch
# conda install -c pytorch -c fastai fastai

try:
    import tensorflow
except ImportError:
    INSTALL_REQUIRES += ['tensorflow']

setup(
    name='sml',
    version='0.1.0',
    packages=['sml'],
    install_requires=INSTALL_REQUIRES
)
