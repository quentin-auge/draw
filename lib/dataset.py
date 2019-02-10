import struct
from struct import unpack

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def cut_strokes(strokes, n_points):
    result_strokes = []
    current_n_points = 0

    for x, y in strokes:
        stroke_size = len(x)
        n_points_remaining = max(0, n_points - current_n_points)
        result_strokes.append((x[:n_points_remaining], y[:n_points_remaining]))
        current_n_points += stroke_size

    return result_strokes


def get_n_points(strokes):
    n_points = 0
    for x, y in strokes:
        n_points += len(x)
    return n_points


PADDING_VALUE = -1
END_OF_STROKE_VALUE = 1000


def strokes_to_points(strokes):
    points = []
    for xs, ys in strokes:
        for x, y in zip(xs, ys):
            points.append([x, y])
        points.append([END_OF_STROKE_VALUE, END_OF_STROKE_VALUE])
    return points


def points_to_strokes(points):
    strokes = []
    stroke_x, stroke_y = [], []

    for x, y in points:
        if x <= 255 and y <= 255:
            stroke_x.append(x)
            stroke_y.append(y)
        else:
            strokes.append((stroke_x, stroke_y))
            stroke_x, stroke_y = [], []

    if stroke_x:
        strokes.append((stroke_x, stroke_y))

    return strokes


def get_dataset(drawings):
    points_drawings = list(map(strokes_to_points, drawings))
    points_drawings = sorted(points_drawings, key=len, reverse=True)
    n_drawings = len(drawings)

    lens = list(map(len, points_drawings))
    max_len = max(lens)
    lens = torch.IntTensor(lens)

    data = torch.ones((n_drawings, max_len, 2)).float() * PADDING_VALUE
    for i, flat_drawing in enumerate(points_drawings):
        data[i, :len(flat_drawing), :] = torch.Tensor(flat_drawing)

    labels = data[:, 1:]
    data = data[:, :-1]
    lens = lens - 1

    print(data.shape, labels.shape, lens)

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
