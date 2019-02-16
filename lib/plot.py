import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lib.strokes import cut_strokes, get_n_points


def get_canvas(nrows=1, ncols=1, *args, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (3 * ncols, 3 * nrows)

    fig, axs = plt.subplots(nrows, ncols, *args, **kwargs)
    axarr = axs if isinstance(axs, np.ndarray) else np.array([axs])

    for ax in axarr.flatten():
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig, axs


def get_drawing_dims(strokes):
    """
    Get drawing bounding box: [min_x, max_x], [min_y, max_y].
    """

    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    for xs, ys in strokes:
        min_x = min(min(xs), min_x)
        max_x = max(max(xs), max_x)
        min_y = min(min(ys), min_y)
        max_y = max(max(ys), max_y)

    return [min_x, max_x], [min_y, max_y]


def resize_strokes(strokes):
    """
    Resize strokes so that they fit in a [0, 255] x [0,255] canvas.
    """

    (min_x, max_x), (min_y, max_y) = get_drawing_dims(strokes)

    range_x = max_x - min_x
    range_y = max_y - min_y

    resized_strokes = []
    for xs, ys in strokes:
        resized_xs = [int((x - min_x) / range_x * 255) for x in xs]
        resized_ys = [int((y - min_y) / range_y * 255) for y in ys]
        resized_strokes.append([resized_xs, resized_ys])

    return resized_strokes


def plot(strokes, color='b', margin=20, ax=None):
    ax = ax or get_canvas()[1]
    strokes = resize_strokes(strokes)

    x_lim, y_lim = get_drawing_dims(strokes)
    x_lim[0], x_lim[1] = x_lim[0] - margin, x_lim[1] + margin
    # Reverse y axis to match the image coordinates framework,
    # as opposed to the plot coordinates framework
    y_lim[0], y_lim[1] = y_lim[1] + margin, y_lim[0] - margin

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    lines = []
    for xs, ys in strokes:
        line, = ax.plot(xs, ys, color=color)
        lines.append(line)

    return lines


def get_animation(strokes, color='b', margin=20, interval=100, ax=None):
    ax = ax or get_canvas()[1]

    strokes = resize_strokes(strokes)
    lines = plot(strokes, color, margin, ax)

    def update(i, strokes, lines):
        strokes = cut_strokes(strokes, i)
        for (xs, ys), line in zip(strokes, lines):
            line.set_data(xs, ys)
        return lines

    n_frames = get_n_points(strokes) + 10
    animation = FuncAnimation(ax.figure, update, fargs=[strokes, lines],
                              frames=n_frames, interval=interval, blit=True)

    return animation
