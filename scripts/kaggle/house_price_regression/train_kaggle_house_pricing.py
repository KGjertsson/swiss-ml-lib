# from scripts.kaggle.house_price_regression.configs import \
#     lasso_config as config

# from scripts.kaggle.house_price_regression.configs import \
#     xgboost_config as config

# from scripts.kaggle.house_price_regression.configs import \
#     lgbm_config as config

# from scripts.kaggle.house_price_regression.configs import \
#     kernel_ridge_config as config

from scripts.kaggle.house_price_regression.configs import \
    lasso_enet_krr_gbr_average as config

# from scripts.kaggle.house_price_regression.configs import \
#     lasso_xgboost_enet_krr_gboost_average as config

# from scripts.kaggle.house_price_regression.configs import \
#     lasso_meta_enet_krr_grb_average as config

# from scripts.kaggle.house_price_regression.configs import \
#     lasso_meta_enet_krr_grb_xgboost_lgbm_average as config


def main():
    model_trainer_cls = config.MODEL_TRAINER_CLS
    model_factory_cls = config.MODEL_FACTORY_CLS
    model_kwargs = config.MODEL_KWARGS
    data_cleaner_cls = config.DATA_CLEANER_CLS

    # data params
    train_data_dir = '../../../data/house_price_regression/'

    trainer = model_trainer_cls(
        model_factory_cls=model_factory_cls,
        model_kwargs=model_kwargs,
        data_cleaner_cls=data_cleaner_cls,
        data_root_dir=train_data_dir
    )
    trainer(['cross_eval'])
    # trainer(['fit', 'make_submission'])


if __name__ == '__main__':
    main()
