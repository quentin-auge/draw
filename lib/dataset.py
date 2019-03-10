import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def get_dataset(transformed_drawings):
    transformed_drawings = sorted(transformed_drawings, key=len, reverse=True)
    n_drawings = len(transformed_drawings)
    n_dims = len(transformed_drawings[0][0])

    lengths = list(map(len, transformed_drawings))
    max_length = max(lengths)
    lengths = torch.IntTensor(lengths)

    data = torch.zeros((n_drawings, max_length, n_dims)).float()
    for i, flat_drawing in enumerate(transformed_drawings):
        data[i, :len(flat_drawing), :] = torch.Tensor(flat_drawing)

    # Shift labels by -1 element compared to data
    labels = data[:, 1:]

    data = data[:, :-1]
    lengths = lengths - 1

    return TensorDataset(data, labels, lengths)


def get_train_val_idxs(n_idxs, train_ratio=0.75, sample_ratio=1.0):
    train_idxs, val_idxs = [], []
    for i in range(n_idxs):
        if np.random.rand() <= sample_ratio:
            if np.random.rand() <= train_ratio:
                train_idxs.append(i)
            else:
                val_idxs.append(i)
    return train_idxs, val_idxs


def get_batches(ds, means_stds, batch_size):
    dl = DataLoader(ds, batch_size)

    for data_batch, labels_batch, lengths_batch in dl:

        standarize_data(data_batch, lengths_batch, means_stds)

        # Labels sequences are shifted by -1 element compared to data sequences
        standarize_data(labels_batch, lengths_batch - 1, means_stds)

        data_batch = data_batch.transpose(0, 1)
        labels_batch = labels_batch.transpose(0, 1)

        if torch.cuda.is_available():
            data_batch = data_batch.cuda()
            labels_batch = labels_batch.cuda()

        yield data_batch, labels_batch, lengths_batch


def create_length_mask(data, lengths):
    """
    Create lengths mask for data along one dimension.
    """
    n_sequences, max_length, _ = data.shape
    lengths_mask = torch.zeros(n_sequences, max_length)
    for i, length in enumerate(lengths):
        lengths_mask[i, :length + 1] = 1
    return lengths_mask


def get_means_stds(ds):
    means_stds = []

    data, _, lengths = ds.dataset[ds.indices]
    lengths_mask = create_length_mask(data, lengths)

    for dim in (0, 1):
        dim_data = data[:, :, dim][lengths_mask == 1]
        mean = dim_data.mean().item()
        std = dim_data.std().item()
        means_stds.append((mean, std))

    return tuple(means_stds)


def standarize_data(data, lengths, means_stds):
    lengths_mask = create_length_mask(data, lengths)

    for dim, (mean, std) in enumerate(means_stds):
        dim_data = data[:, :, dim]
        dim_data[lengths_mask == 1] -= mean
        dim_data[lengths_mask == 1] /= std


def unstandarize_flat_strokes(flat_strokes, means_stds):
    for dim, (mean, std) in enumerate(means_stds):
        flat_strokes[:, dim] *= std
        flat_strokes[:, dim] += mean

    # Very low softmax temperature ensures all coefficients are roughly 0 except one
    temperature = 1e-5
    flat_strokes[:, 2:5] = F.softmax(flat_strokes[:, 2:5] / temperature, dim=1)

    flat_strokes = flat_strokes.round().int().tolist()

    return flat_strokes
