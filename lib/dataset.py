import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataset(transformed_drawings):
    transformed_drawings = sorted(transformed_drawings, key=len, reverse=True)
    n_drawings = len(transformed_drawings)
    n_dims = len(transformed_drawings[0][0])

    # A point is artificially added at the beginning of each drawing, hence
    # `len(transformed_drawing) + 1`.
    lengths = [len(transformed_drawing) + 1 for transformed_drawing in transformed_drawings]
    max_length = max(lengths)
    lengths = torch.IntTensor(lengths)

    # All drawings have the length of the longest drawing `max_length`, padded with zeros
    # after one artificial point is added at the end of each drawing, hence `max_length + 1`.
    data = torch.zeros((n_drawings, max_length + 1, n_dims)).float()

    for i, flat_drawing in enumerate(transformed_drawings):
        # Set artificial first drawing point
        data[i, 0, :] = torch.Tensor([0, 0, 1, 0, 0]).float()
        # Set drawing points
        data[i, 1:len(flat_drawing) + 1, :] = torch.Tensor(flat_drawing)
        # Set artificial last drawing point
        data[i, len(flat_drawing) + 1, :] = torch.Tensor([0, 0, 0, 0, 1]).float()

    # For labels, discard the null point at the beginning of each drawing. Include the one
    # at the end.
    labels = data[:, 1:]

    # For data, discard the null point at the end of each drawing. Include the one
    # at the beginning.
    data = data[:, :-1]

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


def get_batches(ds, stds, batch_size):
    dl = DataLoader(ds, batch_size)

    for data_batch, labels_batch, lengths_batch in dl:

        standarize_data(data_batch, lengths_batch, stds)
        standarize_data(labels_batch, lengths_batch, stds)

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


def get_stds(ds):
    stds = []

    data, _, lengths = ds.dataset[ds.indices]
    lengths_mask = create_length_mask(data, lengths)

    for dim in (0, 1):
        dim_data = data[:, :, dim][lengths_mask == 1]
        std = dim_data.std().item()
        stds.append(std)

    return tuple(stds)


def standarize_data(data, lengths, stds):
    lengths_mask = create_length_mask(data, lengths)

    for dim, std in enumerate(stds):
        dim_data = data[:, :, dim]
        dim_data[lengths_mask == 1] /= std


def unstandarize_flat_strokes(flat_strokes, stds):
    for dim, std in enumerate(stds):
        flat_strokes[:, dim] *= std

    flat_strokes = flat_strokes.round().int().tolist()

    return flat_strokes
