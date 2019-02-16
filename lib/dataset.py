import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PADDING_VALUE = -1


def get_dataset(transformed_drawings):
    transformed_drawings = sorted(transformed_drawings, key=len, reverse=True)
    n_drawings = len(transformed_drawings)
    n_dims = len(transformed_drawings[0][0])

    lens = list(map(len, transformed_drawings))
    max_len = max(lens)
    lens = torch.IntTensor(lens)

    data = torch.ones((n_drawings, max_len, n_dims)).float() * PADDING_VALUE
    for i, flat_drawing in enumerate(transformed_drawings):
        data[i, :len(flat_drawing), :] = torch.Tensor(flat_drawing)

    labels = data[:, 1:]
    data = data[:, :-1]
    lens = lens - 1

    return TensorDataset(data, labels, lens)


def get_train_val_idxs(n_idxs, train_ratio=0.75, sample_ratio=1.0):
    train_idxs, val_idxs = [], []
    for i in range(n_idxs):
        if np.random.rand() <= sample_ratio:
            if np.random.rand() <= train_ratio:
                train_idxs.append(i)
            else:
                val_idxs.append(i)
    return train_idxs, val_idxs


def get_batches(ds, batch_size):
    dl = DataLoader(ds, batch_size)

    for data_batch, labels_batch, lens_batch in dl:

        data_batch = data_batch.transpose(0, 1)
        labels_batch = labels_batch.transpose(0, 1)

        if torch.cuda.is_available():
            data_batch = data_batch.cuda()
            labels_batch = labels_batch.cuda()

        yield data_batch, labels_batch, lens_batch
