import numpy as np

from scipy.spatial.distance import euclidean

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate, resize
from skimage.morphology import skeletonize

THRESHOLD = 150


def is_upside_down(image, line):
    (x1, y1), (x2, y2) = line
    y = int(np.round((y1 + y2) / 2))
    x = int(np.round((x1 + x2) / 2))

    image_with_border = np.pad(image, pad_width=10, mode='constant', constant_values=0.0)

    up = image_with_border[y, x + 10]
    down = image_with_border[y + 20, x + 10]

    up_value = up < THRESHOLD
    down_value = down > THRESHOLD

    return up_value and down_value


def get_rotation_for_vertical(image, line):
    (x1, y1), (x2, y2) = line
    y = int(np.round((y1 + y2) / 2))
    x = int(np.round((x1 + x2) / 2))

    image_with_border = np.pad(image, pad_width=10, mode='constant', constant_values=0.0)

    left = image_with_border[y + 10, x]
    right = image_with_border[y + 10, x + 20]

    left_value = left < THRESHOLD
    right_value = right > THRESHOLD

    if left_value and right_value:
        return 90.0
    else:
        return -90.0


def get_line(image):
    thresh = threshold_otsu(image)
    normalize = image > thresh

    edges = canny(normalize, 0, 1, 1)

    min_line_length = int(image.shape[0] / 2)

    lines = []
    while not lines:
        min_line_length = int(min_line_length * 0.9)
        lines = probabilistic_hough_line(edges, seed=16, line_length=min_line_length, line_gap=3)

    longest_line = None
    longest_line_distance = 0.0

    for line in lines:
        point_a, point_b = line
        distance = euclidean(point_a, point_b)

        if longest_line_distance < distance:
            longest_line = line
            longest_line_distance = distance

    return longest_line


def get_rotation(image):
    line = get_line(image)
    (x1, y1), (x2, y2) = line

    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) else 0

    rad_angle = np.arctan(slope)
    rotation = np.degrees(rad_angle)

    if x1 == x2:
        rotation += get_rotation_for_vertical(image, line)
    elif is_upside_down(image, line):
        rotation += 180.0

    return rotation


def rotate_image(image):
    rotation = get_rotation(image)
    rotated = rotate(image, rotation, resize=True)
    return rotated


def trim_image(image):
    trimmed = image[:, ~np.all(image < 1.0, axis=0)]
    trimmed = trimmed[~np.all(trimmed < 1.0, axis=1)]
    trimmed = trimmed[~np.all(trimmed > 0.0, axis=1)]
    return trimmed


def resize_image(image):
    ratio = 200 / image.shape[1]

    x_size = int(np.round(image.shape[0] * ratio))
    y_size = int(np.round(image.shape[1] * ratio))

    resized = resize(image, (x_size, y_size), anti_aliasing=False)
    return resized


def binarize_image(image):
    binary = image.copy()
    binary[binary > 0.5] = 1.0
    binary[binary <= 0.5] = 0.0
    return binary


def approximate_values(image, bins=5):
    max_value = image.shape[0]
    previous_value = max_value
    values = []

    for i in range(image.shape[1]):
        column = image[:, i]

        if np.any(column):
            current_value = max_value - np.argmax(column)
        else:
            current_value = previous_value

        values.append(current_value)

        previous_value = current_value

    chunks = np.array_split(values, bins)

    approximated_values = []
    inverted_approximated_values = []

    for chunk in chunks:
        median = np.median(chunk)

        approximated_values.append(median)
        inverted_approximated_values.append(max_value - median)

    inverted_approximated_values.reverse()

    return approximated_values, inverted_approximated_values


def get_image_characteristic(set_number, image_number):
    image = imread(f'test_sets/set{set_number}/{image_number}.png')

    rotated = rotate_image(image)
    trimmed = trim_image(rotated)
    resized = resize_image(trimmed)
    binary = binarize_image(resized)
    edges = canny(binary, 0, 1, 1)
    skeleton = skeletonize(edges)

    approximated_values, inverted_approximated_values = approximate_values(skeleton, 8)

    return approximated_values, inverted_approximated_values


def get_images_characteristics(set_number, set_range):
    characteristics = []

    for image_number in range(set_range):
        av, iav = get_image_characteristic(set_number, image_number)
        characteristics.append((image_number, av, iav))

    return characteristics


def compare_characteristics(characteristics):
    for characteristic in characteristics:
        image_number, _, iav = characteristic

        scores = {}

        for comparison_characteristic in characteristics:
            comparison_image_number, av, _ = comparison_characteristic

            if image_number == comparison_image_number:
                continue

            score = 0
            for i in range(len(iav)):
                score += abs(iav[i] - av[i])
            scores[comparison_image_number] = score

        print(image_number, min(scores, key=scores.get))


def test_set(set_number, set_range):
    characteristics = get_images_characteristics(set_number, set_range)
    compare_characteristics(characteristics)


if __name__ == "__main__":
    test_set(1, 20)
