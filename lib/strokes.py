import abc
import math

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


class StrokesTransformer(object):
    """
    Transform a list of strokes to an intermediate representation, and back.
    """

    def transform(self, strokes):
        """
        Flatten and transform a list of strokes.
        """
        flat_strokes = flatten_strokes(strokes)
        transformed_strokes = self._transform_flat_strokes(flat_strokes)
        return transformed_strokes

    def _transform_flat_strokes(self, flat_strokes):
        """
        Transform a list of flattened strokes to its intermediate representation.
        Preserve stroke state.
        """

        transformed_strokes = []
        shifted_flat_strokes = flat_strokes[1:] + [None]
        for point, next_point in zip(flat_strokes, shifted_flat_strokes):
            new_point = self._transform(point, next_point)
            if new_point:
                transformed_strokes.append(new_point)

        # Mark end of drawing. That state might have been lost in transformation process.
        v, w, *_ = transformed_strokes[-1]
        transformed_strokes[-1] = [v, w, 0, 0, 1]

        return transformed_strokes

    @abc.abstractmethod
    def _transform(self, point, next_point):
        raise NotImplementedError

    def inverse_transform(self, transformed_strokes, initial_point=(0, 0)):
        """
        Reconstruct a list of strokes from its intermediate representation using strokes state.

        The first point may have been lost in the transformation process. It is
        therefore possible to supply it to reconstruct the original strokes
        without loosing information.
        """

        strokes = []
        stroke_xs, stroke_ys = [], []

        # Accumulator
        acc = initial_point

        for v, w, *stroke_state in transformed_strokes:

            (x, y), acc = self._inverse_transform((v, w), acc)
            stroke_xs.append(x)
            stroke_ys.append(y)

            _, is_end_of_stroke, is_end_of_drawing = stroke_state

            if is_end_of_stroke:
                # Start a new stroke
                strokes.append([stroke_xs, stroke_ys])
                stroke_xs, stroke_ys = [], []

            if is_end_of_drawing:
                break

        # Flush accumulator
        if acc is not None:
            x, y = acc
            stroke_xs.append(x)
            stroke_ys.append(y)

        # Flush last stroke
        if stroke_xs and stroke_ys:
            strokes.append((stroke_xs, stroke_ys))

        return strokes

    @abc.abstractmethod
    def _inverse_transform(self, point, accumulator):
        raise NotImplementedError


class PointsStrokesTransformer(StrokesTransformer):
    """
    Transformation: identity.
    """

    def _transform(self, point, _):
        return point

    def _inverse_transform(self, point, _):
        return point, None


class VectorsStrokesTransformer(StrokesTransformer):
    def _transform(self, point, next_point):
        """
        Transformation: two-points -> cartesian-coordinates vector.
        """
        if next_point is not None:
            x, y, *stroke_state = point
            x_next, y_next, *_ = next_point
            new_point = [x_next - x, y_next - y] + stroke_state
            return new_point

    def _inverse_transform(self, point, acc_point):
        new_point = acc_point

        delta_x, delta_y = point
        x, y = acc_point
        acc_point = (x + delta_x, y + delta_y)

        return new_point, acc_point


class PolarStrokesTransformer(StrokesTransformer):
    def _transform(self, point, next_point):
        """
        Transformation: two-points -> polar-coordinates vector.
        """
        if next_point is not None:
            x, y, *stroke_state = point
            x_next, y_next, *_ = next_point
            distance = math.sqrt((x_next - x) ** 2 + (y_next - y) ** 2)
            angle = math.atan2(y_next - y, x_next - x)
            new_point = [distance, angle] + stroke_state
            return new_point

    def _inverse_transform(self, point, acc_point):
        new_point = acc_point

        distance, angle = point
        x, y = acc_point
        x = int(x + distance * math.cos(angle))
        y = int(y + distance * math.sin(angle))
        acc_point = (x, y)

        return new_point, acc_point


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


if __name__ == '__main__':
    from pprint import pprint
    from matplotlib import pyplot as plt
    from lib.plot import plot

    dataset = unpack_drawings('./data/The Eiffel Tower.bin')
    strokes = next(dataset)['image']

    pprint(strokes)

    transformer = PointsStrokesTransformer()
    transformer = VectorsStrokesTransformer()
    transformer = PolarStrokesTransformer()

    transformed_strokes = transformer.transform(strokes)
    pprint(transformed_strokes)

    reconstructed_strokes = transformer.inverse_transform(transformed_strokes,
                                                          initial_point=(0, 218))
    pprint(reconstructed_strokes)

    plot(strokes)
    plot(reconstructed_strokes)
    plt.show()
