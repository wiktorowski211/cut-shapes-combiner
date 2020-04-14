import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from scipy.spatial.distance import euclidean

from skimage.io import imread, imshow
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


def rotation_for_vertical(image, line):
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


def print_lines(image, edges, lines):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def get_line(image):
    thresh = threshold_otsu(image)
    normalize = image > thresh

    edges = canny(normalize, 0, 1, 1)

    min_line_length = int(image.shape[0] / 2)

    lines = []
    while not lines:
        min_line_length = int(min_line_length * 0.9)
        lines = probabilistic_hough_line(edges, seed=16, line_length=min_line_length, line_gap=3)

    #     print_lines(image, edges, lines)

    longest_line = None
    longest_line_distance = 0.0

    for line in lines:
        point_a, point_b = line
        distance = euclidean(point_a, point_b)

        if longest_line_distance < distance:
            longest_line = line
            longest_line_distance = distance

    return longest_line


def deskew(image):
    line = get_line(image)
    (x1, y1), (x2, y2) = line

    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) else 0

    rad_angle = np.arctan(slope)
    rotation = np.degrees(rad_angle)

    #     print(line)
    #     print(slope)
    #     print(rad_angle)
    #     print(rotation)

    if x1 == x2:
        rotation += rotation_for_vertical(image, line)
    elif is_upside_down(image, line):
        rotation += 180.0
    # print(rotation)

    return rotation


def aproximate_values(image, bins=5):
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

    aproximated_values = []
    inversed_aproximated_values = []

    for chunk in chunks:
        median = np.median(chunk)

        aproximated_values.append(median)
        inversed_aproximated_values.append(max_value - median)

    inversed_aproximated_values.reverse()

    return aproximated_values, inversed_aproximated_values


def test_image(case, number):
    img = imread(f'test_cases/set{case}/{number}.png')
    rotation = deskew(img)
    rotated = rotate(img, rotation, resize=True)

    if number in []:
        plt.figure()
        plt.imshow(rotated, cmap='gray')

    trimmed = rotated[:, ~np.all(rotated < 1.0, axis=0)]
    trimmed = trimmed[~np.all(trimmed < 1.0, axis=1)]
    trimmed = trimmed[~np.all(trimmed > 0.0, axis=1)]

    ratio = 200 / trimmed.shape[1]
    x_size = int(np.round(trimmed.shape[0] * ratio))
    y_size = int(np.round(trimmed.shape[1] * ratio))

    resized = resize(trimmed, (x_size, y_size), anti_aliasing=False)

    binary = resized.copy()
    binary[binary > 0.5] = 1.0
    binary[binary <= 0.5] = 0.0

    edges = canny(binary, 0, 1, 1)

    skeleton = skeletonize(edges)

    approximated_values, inversed_approximated_values = aproximate_values(skeleton, 8)

    return skeleton.std(), approximated_values, inversed_approximated_values


def test_case(case_number, case_range):
    cases = []
    for i in range(case_range):
        std, av, iav = test_image(case_number, i)
        cases.append((i, std, av, iav))

    for case in cases:
        case_number, _, _, iav = case

        scores = {}

        for tested_case in cases:
            tested_case_number, _, av, _ = tested_case

            if case_number == tested_case_number:
                continue

            score = 0
            for x in range(len(iav)):
                score += abs(iav[x] - av[x])
            scores[tested_case_number] = score

        if case_number in []:
            print(scores)

        print(case_number, min(scores, key=scores.get))


if __name__ == "__main__":
    test_case(0, 6)
