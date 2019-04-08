from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_root_dir = Path('../../../../data/house_price_regression/')

train_data = pd.read_csv(Path(data_root_dir, 'train.csv'))


plt.plot(train_data["YearBuilt"].values,
         np.log1p(train_data["SalePrice"].values), '.')
# plt.plot(train_data["YearBuilt"].values)
plt.show()

# print(type(train_data["SalePrice"]))
