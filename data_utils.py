import numpy as np
import pandas as pd
import scipy.sparse as sp
import csv
import torch.utils.data as data

import config


def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(config.train_rating)
    user_num = train_data['user_id'].max() + 1
    item_num = train_data['click_article_id'].max() + 1

    train_data = train_data.values.tolist()

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for user, item, _ in train_data:
        train_mat[user, item] = 1.0

    test_data = []

    with open(config.test_negative, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            u = int(row[0])  # First column is user ID
            positive_item = int(row[1])  # Second column is positive item

            test_data.append([u, positive_item])

            for item in row[2:]:
                test_data.append([u, int(item)])

    return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features,
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()

        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label
