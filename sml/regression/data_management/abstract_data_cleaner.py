import abc


class AbstractDataCleaner(abc.ABC):

    def __init__(self, data_root_dir, verbose=1):
        self.data_root_dir = data_root_dir
        self.verbose = verbose

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def manage_outliers(self):
        pass

    @abc.abstractmethod
    def format_target(self):
        pass

    @abc.abstractmethod
    def features_engineering(self):
        pass
