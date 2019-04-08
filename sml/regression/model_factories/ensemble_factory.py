class EnsembleRegressionFactory:

    def __init__(self, ensemble_cls, child_factory_cls, model_data):
        self.ensemble_cls = ensemble_cls
        self.models = \
            [child_factory_cls(
                model_type,
                **model_kwargs
            )() for model_type, model_kwargs in model_data.items()]

    def __call__(self):
        return self.ensemble_cls(self.models)
