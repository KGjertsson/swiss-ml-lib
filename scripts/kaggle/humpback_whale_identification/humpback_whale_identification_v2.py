import os
from pathlib import Path

import cv2
from fastai.conv_learner import *
from fastai.dataset import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class HWIDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        bbox = bbox_df.loc[self.fnames[i]]
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
        if not (x0 >= x1 or y0 >= y1):
            img = img[y0:y1, x0:x1, :]
        img = cv2.resize(img, (self.sz, self.sz))
        return img

    def get_y(self, i):
        if (self.path == test_dir): return 0
        return self.train_df.loc[self.fnames[i]]['Id']

    def get_c(self):
        return len(unique_labels)


# define paths
data_root_dir = Path('../../../data/humpback-whale-identification')
train_dir = data_root_dir / 'train'
test_dir = data_root_dir / 'test'
targets_file = data_root_dir / 'train.csv'
sample_sub_file = data_root_dir / 'sample_submission.csv'
bounding_boxes_file = data_root_dir / 'bounding_boxes.csv'

# load data
targets = pd.read_csv(targets_file).set_index('Image')

# only new_whale dataset
new_whale_df = targets[targets.Id == "new_whale"]

# no new_whale dataset, used for training
train_df = targets[~(targets.Id == "new_whale")]
unique_labels = np.unique(train_df.Id.values)

# there are 5004 unique whales in train_df

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
print("Number of classes: {}".format(len(unique_labels)))
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)

test_names = [f for f in os.listdir(test_dir)]

view_class_histogram = False
if view_class_histogram:
    labels_count = train_df.Id.value_counts()

    plt.figure(figsize=(18, 4))
    plt.subplot(121)
    _, _, _ = plt.hist(labels_count.values)
    plt.ylabel("frequency")
    plt.xlabel("class size")

    plt.title('class distribution; log scale')
    labels_count.head()

    plt.subplot(122)
    _ = plt.plot(labels_count[1:].values)
    plt.title('w/o class new_whale; log scale')
    plt.xlabel("class")
    plt.ylabel("log(size)")
    plt.gca().set_yscale('log')
    plt.show()

# load bounding boxes and merge data
train_df['image_name'] = train_df.index
bbox_df = pd.read_csv(bounding_boxes_file).set_index('Image')

# set random seed to be equal to the sense of life
rs = np.random.RandomState(42)
perm = rs.permutation(len(train_df))

# TODO: better train val split
tr_n = train_df['image_name'].values
# Yes, we will validate on the subset of training data
val_n = train_df['image_name'].values[perm][:1000]

print('Train/val:', len(tr_n), len(val_n))
print('Train classes', len(train_df.loc[tr_n].Id.unique()))
print('Val classes', len(train_df.loc[val_n].Id.unique()))
