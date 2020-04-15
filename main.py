import numpy as np

from scipy.spatial.distance import euclidean

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate, resize
from skimage.morphology import skeletonize

THRESHOLD = 150
SET_RANGES = [6, 20, 20, 20, 20, 200, 200, 20, 100]


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

    edges = canny(normalize, 1.5)

    min_line_length = int(min(image.shape) / 2)

    lines = []
    while not lines:
        lines = probabilistic_hough_line(edges, seed=16, line_length=min_line_length, line_gap=3)
        min_line_length = int(min_line_length * 0.9)

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

    rad_angle2 = np.arctan2(y2-y1, x2-x1)
    
    rotation = np.degrees(rad_angle2)
    return rotation


def rotate_image(image):
    rotation = get_rotation(image)
    rotated = rotate(image, rotation, resize=True)
    return rotated


def trim_image(image):
    trimmed = image[:, ~np.all(image < 0.5, axis=0)]
    trimmed = trimmed[~np.all(trimmed < 0.5, axis=1)]
    trimmed = trimmed[~np.all(trimmed > 0.0, axis=1)]
    return trimmed

def fix_upside_down(image):
    if np.count_nonzero(image[0, :]) > image.shape[1]/2:
        return np.flip(image)
    else:
        return image

def binarize_image(image):
    binary = image.copy()
    binary[binary > 0.5] = 1.0
    binary[binary <= 0.5] = 0.0
    return binary


def approximate_values(image, bins=5):
    max_value = 100
    normalized_ratio = 100./image.shape[0]
    previous_value = 0.
    values = []

    for i in range(image.shape[1]):
        column = image[:, i]

        if np.any(column):
            current_value = max_value - np.argmax(column)*normalized_ratio
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
    fixed = fix_upside_down(trimmed)
    binary = binarize_image(fixed)

    approximated_values, inverted_approximated_values = approximate_values(binary, 10)

    return approximated_values, inverted_approximated_values


def get_images_characteristics(set_number):
    characteristics = []
    set_range = SET_RANGES[set_number]

    for image_number in range(set_range):
        av, iav = get_image_characteristic(set_number, image_number)
        characteristics.append((image_number, av, iav))

    return characteristics

def compare_characteristics(characteristics, ranking_len=15):
    results = []
    for characteristic in characteristics:
        image_number, _, iav = characteristic

        scores = []

        for comparison_characteristic in characteristics:
            comparison_image_number, av, _ = comparison_characteristic

            if image_number == comparison_image_number:
                continue

            score = 0
            scorez = []
            for i in range(len(iav)):
                score += abs(iav[i] - av[i])
                scorez.append((iav[i] - av[i]))
            scores.append((comparison_image_number, np.std(scorez)))
        scores.sort(key=lambda x: x[1])
        scores = scores[:ranking_len]
        scores = list(map(lambda x: x[0], scores))

        results.append(scores)
    return results


def get_correct_results(set_number):
    with open(f'test_sets/set{set_number}/correct.txt', "r") as file:
        lines = file.readlines()

        result = []
        for line in lines:
            result.append(int(line))

        return result


def compare_results(results, correct_results):
    correct = 0
    results_number = len(correct_results)

    incorrect_ids = []

    for i in range(results_number):
        n = 1
        found = False
        for result in results[i]:
            if result == correct_results[i]:
                correct += 1/n
                found = True if n == 1 else False
                break
            n+=1
        if not found:
            incorrect_ids.append(i)
            correct += 1/(results_number-1)

    return correct, results_number, incorrect_ids


def test_set(set_number):
    characteristics = get_images_characteristics(set_number)

    results = compare_characteristics(characteristics, ranking_len=15)
    correct_results = get_correct_results(set_number)

    correct, results_number, incorrect_ids = compare_results(results, correct_results)

    print(f"Set number {set_number}")
    print(f"Score: {correct}/{results_number}")
    print("Incorrect ids:", incorrect_ids)
    print()


def test_sets(sets_range):
    for set_number in range(sets_range):
        test_set(set_number)


if __name__ == "__main__":
    test_sets(9)
