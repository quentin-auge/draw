import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lib.dataset import _cut_strokes, get_n_points


def get_canvas(nrows=1, ncols=1, *args, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (3 * ncols, 3 * nrows)

    fig, axs = plt.subplots(nrows, ncols, *args, **kwargs)
    axarr = axs if isinstance(axs, np.ndarray) else np.array([axs])

    for ax in axarr.flatten():
        ax.set_xlim(-20, 275)
        ax.set_ylim(275, -20)  # Reverse y axis
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig, axs


def plot(drawing, color='b', ax=None):
    ax = ax or get_canvas()

    lines = []
    for xs, ys in drawing:
        line, = ax.plot(xs, ys, color=color)
        lines.append(line)

    return lines


def get_animation(drawing, color='b', interval=100, ax=None):
    ax = ax or get_canvas()[1]
    lines = plot(drawing, color, ax)

    def update(i, strokes, lines):
        strokes = _cut_strokes(strokes, i)
        for (xs, ys), line in zip(strokes, lines):
            line.set_data(xs, ys)
        return lines

    n_frames = get_n_points(drawing) + 10
    animation = FuncAnimation(ax.figure, update, fargs=[drawing, lines],
                              frames=n_frames, interval=interval, blit=True)

    return animation
