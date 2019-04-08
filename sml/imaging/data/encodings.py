from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class SmlLabelEncoder:

    def __init__(self):
        pass

    @staticmethod
    def str_categories_to_one_hot(str_values):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(str_values)

        one_hot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)

        labels_ids = one_hot_encoded
        return labels_ids, label_encoder
