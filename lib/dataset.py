import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PADDING_VALUE = -1


def cut_strokes(strokes, n_points):
    """
    Reduce a drawing to its n first points.
    """

    result_strokes = []
    current_n_points = 0

    for xs, ys in strokes:
        stroke_size = len(xs)
        n_points_remaining = max(0, n_points - current_n_points)
        result_strokes.append((xs[:n_points_remaining], ys[:n_points_remaining]))
        current_n_points += stroke_size

    return result_strokes


def get_n_points(strokes):
    """
    Get number of points in a drawing.
    """

    n_points = 0
    for x, y in strokes:
        n_points += len(x)
    return n_points


def _flatten_strokes(strokes):
    """
    Flatten a list of strokes. Add stroke state in the process.

    For each point j, the stroke state is a tuple (pj, qj, rj) where:
      * pj=1 indicates the point is not the end of a stroke.
      * qj=1 indicates the point is the end of a stroke (but not the end of the drawing).
      * rj=1 indicates the point is the end of the drawing.
    By construction, pj + qj + rj = 1

    Input:
        [
            ((x1, x2, ..., xi-1, xi), (y1, y2, ..., yi-1, yi)),
            ((xi+1, ...), (yi+1, ...)),
            ...,
            ((..., xn-1, xn), (..., yn-1, yn))
        ]

    Output:
        [
            [x1,   y1,   1, 0, 0],
            [x2,   y2,   1, 0, 0],
            ...,
            [xi-1, yi-1, 1, 0, 0],
            [xi,   yi,   0, 1, 0]
            [xi+1, yi+1, 1, 0, 0],
            ...,
            [xn-1, yn-1, 1, 0, 0]
            [xn,   yn,   0, 0, 1]
        ]
    """

    flat_strokes = []

    for xs, ys in strokes:

        for x, y in zip(xs, ys):
            # Mark stroke in progress by default
            flat_strokes.append([x, y, 1, 0, 0])

        # Mark end of stroke
        x, y, *_ = flat_strokes[-1]
        flat_strokes[-1] = [x, y, 0, 1, 0]

    # Mark end of drawing
    x, y, *_ = flat_strokes[-1]
    flat_strokes[-1] = [x, y, 0, 0, 1]

    return flat_strokes


def _transform_strokes(flat_strokes):
    """
    Transform the points of (flattened) strokes to vectors from each point to the next.
    Preserve stroke state.
    """

    vector_strokes = []
    for point, next_point in zip(flat_strokes, flat_strokes[1:]):
        x, y, *stroke_state = point
        next_x, next_y, *_ = next_point
        vector_strokes.append([next_x - x, next_y - y] + stroke_state)

    # Mark end of drawing (relevant stroke state is in the next point, not in
    # the current one, for once).
    x, y, *_ = vector_strokes[-1]
    vector_strokes[-1] = [x, y, 0, 0, 1]

    return vector_strokes


def transform_strokes(strokes):
    """
    Flatten and transform a list of strokes.
    """
    flat_strokes = _flatten_strokes(strokes)
    transformed_strokes = _transform_strokes(flat_strokes)
    return transformed_strokes


def reconstruct_strokes(flat_strokes, initial_point=(0, 0)):
    """
    Reconstruct a list of strokes formatted as in the original dataset
    from a list of flattened and transformed strokes.

    The first point may have been lost in the transformation process. It is
    therefore possible to supply it to reconstruct the original strokes
    without loosing information.
    """

    x, y = initial_point

    strokes = []
    stroke_xs, stroke_ys = [], []
    for delta_x, delta_y, *stroke_state in flat_strokes:
        x, y = x + delta_x, y + delta_y

        _, is_end_of_stroke, is_end_of_drawing = stroke_state

        if is_end_of_stroke:
            # Start a new stroke
            strokes.append([stroke_xs, stroke_ys])
            stroke_xs, stroke_ys = [], []

        stroke_xs.append(x)
        stroke_ys.append(y)

        if is_end_of_drawing:
            # Finish drawing
            strokes.append((stroke_xs, stroke_ys))
            stroke_xs, stroke_ys = [], []
            break

    # In case there is no end_of_drawing state in the drawing.
    # Can happen when the model does the drawing.
    if stroke_xs and stroke_ys:
        strokes.append((stroke_xs, stroke_ys))

    return strokes


def get_dataset(drawings):
    transformed_drawings = list(map(transform_strokes, drawings))
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
