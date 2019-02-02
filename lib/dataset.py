import struct
from struct import unpack


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


def _cut_strokes(strokes, n_points):
    result_strokes = []
    current_n_points = 0

    for x, y in strokes:
        assert len(x) == len(y)
        stroke_size = len(x)
        n_points_remaining = max(0, n_points - current_n_points)
        result_strokes.append((x[:n_points_remaining], y[:n_points_remaining]))
        current_n_points += stroke_size

    return result_strokes


def get_n_points(drawing):
    n_points = 0
    for x, y in drawing:
        assert len(x) == len(y)
        n_points += len(x)
    return n_points
