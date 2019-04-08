# model_trainer_cls = HyeroptTrainer

# model_kwargs = {
#     'train_data': train,
#     'train_labels': y_train,
#     'epochs': 1000,
#     'train_fun': 'mean_rmsle_cv',
#     'child_model_params': {
#         'n_estimators': hp.quniform('n_estimators', 1, 5000, 1),
#         'learning_rate': hp.uniform('learning_rate', 0, 1),
#         'max_depth': hp.quniform('max_depth', 1, 10, 1),
#         'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
#         'min_samples_leaf': 15,
#         'min_samples_split': 10,
#         'loss': hp.choice('loss', ['huber', 'quantile']),
#         'random_state': 5
#     },
#     'child_model_type': 'GBoost',
#     'meta_model_type': 'tpe',
#     'n_folds': 3,
#     'verbose': verbose
# }
