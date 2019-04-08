import numpy as np
import pandas as pd


def map_per_image(y_true, y_pred):
    """Computes the precision score of one image.

    Parameters
    ----------
    y_true : string
            The true label of the image
    y_pred : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (y_pred[:5].index(y_true) + 1)
    except ValueError:
        return 0.0


def map_per_set(y_true, y_pred):
    """Computes the average over multiple images.

    Parameters
    ----------
    y_true : list
             A list of the true labels. (Only one true label per images allowed!)
    y_pred : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l, p in zip(y_true, y_pred)])


if __name__ == '__main__':
    train_df = pd.read_csv('../../../../humpback-whale/data/train.csv')
    labels = train_df['Id'].values
    sample_pred = ['new_whale', 'w_23a388d', 'w_9b5109b', 'w_9c506f6', 'w_0369a5c']
    predictions = [sample_pred for i in range(len(labels))]

    print(map_per_set(labels, predictions))

