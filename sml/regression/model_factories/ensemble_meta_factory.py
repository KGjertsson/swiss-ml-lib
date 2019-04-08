from .ensemble_factory import EnsembleRegressionFactory


class MetaEnsembleFactory(EnsembleRegressionFactory):

    def __init__(self, ensemble_cls, child_factory_cls, model_data,
                 meta_model_data):
        super().__init__(ensemble_cls, child_factory_cls, model_data)
        self.meta_model = [child_factory_cls(
            model_type,
            **model_kwargs
        )() for model_type, model_kwargs in meta_model_data.items()][0]

    def __call__(self):
        return self.ensemble_cls(self.models, self.meta_model)
