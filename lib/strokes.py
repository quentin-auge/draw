from lib.unpack import unpack_drawings


def flatten_strokes(strokes):
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


def transform_strokes(strokes):
    """
    First flatten strokes, then transform them from [(x, y)] points to [(dx, dy)] displacements
    between two successive points in cartesian coordinates, while preserving stroke state.
    """

    flat_strokes = flatten_strokes(strokes)

    transformed_strokes = []
    for point, next_point in zip(flat_strokes, flat_strokes[1:]):
        x, y, *stroke_state = point
        x_next, y_next, *_ = next_point
        new_point = [x_next - x, y_next - y] + stroke_state
        transformed_strokes.append(new_point)

    # Mark end of drawing. That state might have been lost in transformation process.
    dx, dy, *_ = transformed_strokes[-1]
    transformed_strokes[-1] = [dx, dy, 0, 0, 1]

    return transformed_strokes


def inverse_transform_strokes(transformed_strokes):
    """
    First untransform transformed strokes, then unflatten them.

    The first point has been lost in the transformation process. It is therefore
    not possible to reconstruct the exact original strokes. The reconstructed strokes
    will thus be a translation of the original strokes.
    """

    strokes = []
    stroke_xs, stroke_ys = [0], [0]
    x, y = 0, 0

    for dx, dy, *stroke_state in transformed_strokes:

        _, is_end_of_stroke, is_end_of_drawing = stroke_state

        if is_end_of_stroke:
            # Start a new stroke
            strokes.append([stroke_xs, stroke_ys])
            stroke_xs, stroke_ys = [], []

        x += dx
        y += dy

        stroke_xs.append(x)
        stroke_ys.append(y)

        if is_end_of_drawing:
            break

    # Flush last stroke
    if stroke_xs and stroke_ys:
        strokes.append((stroke_xs, stroke_ys))

    return strokes


def get_n_points(strokes):
    """
    Get number of points in a drawing.
    """

    n_points = 0
    for x, y in strokes:
        n_points += len(x)
    return n_points


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


if __name__ == '__main__':
    from pprint import pprint

    dataset = unpack_drawings('./data/The Eiffel Tower.bin')
    strokes = next(dataset)['image']

    pprint(strokes)

    transformed_strokes = transform_strokes(strokes)
    pprint(transformed_strokes)

    reconstructed_strokes = inverse_transform_strokes(transformed_strokes)
    pprint(reconstructed_strokes)
