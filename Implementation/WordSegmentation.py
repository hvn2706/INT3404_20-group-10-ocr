from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, convolve2d

import numpy as np
import cv2


def show_img(img, zoom=1.0):
    """
    displaying an image
    :param img : (numpy.ndarray) image to be shown
    :param zoom: (float) zoom ratio
    :return: none
    """
    img = img.astype('uint8')
    img = cv2.resize(img, None, fx=zoom, fy=zoom)

    cv2.imshow('test', img)
    cv2.waitKey(0)


def get_anisotropic_gaussian_kernel(size: int = 5, sigma_y: float = 4, sigma_x: float = 16):
    """Get an anisotropic gaussian kernel

    Args:
        size (str): Kernel size
        sigma_y (float): Sigma value for y-axis
        sigma_x (float): Sigma value for x-axis

    Returns:
        numpy.ndarray: The kernel
    """

    # Check if size is odd, if not, add 1 to size
    if size % 2 == 0:
        size += 1

    # Create an evenly spaced numbers, centered at 0 (which are the coordinates)
    half = (size-1)/2
    tmp = np.linspace(-half, half, size)

    # Create 2 matrices for each square's coordinate in the kernel
    (x, y) = np.meshgrid(tmp, tmp)

    # Calculate the kernel
    kernel = np.exp(
        -0.5 * (
            np.square(y) / np.square(sigma_y) +
            np.square(x) / np.square(sigma_x)
        )
    )

    # Rescaling and return
    return kernel / np.sum(kernel)


def split_lines(image, sigma: float = 2, order: int = 15, vertical: bool = False):
    """Split a document image into lines

    Args:
        image (numpy.ndarray): The image
        sigma (float): Sigma for gaussian smoothing the projection of image
        order (float): Range when comparing for calculating the maximas
        vertical (bool): Lines are vertical or not

    Returns:
        lines (List[(int, int)]): List of start point and end point of lines on the y-axis
        bin_image (numpy.ndarray): The binary representation of input image
    """

    # Check if lines are vertical or horizontal
    if vertical:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Get the gray scale image
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Convert gray scale image to binary image
    _, bin_image = cv2.threshold(gray_image, 255 / 2, 255, cv2.THRESH_BINARY)

    # Get image's shape
    (h, w) = bin_image.shape

    # Initialize the projection array
    tmp = [w] * h

    # Calculate the projection array
    for j in range(h):
        for i in range(w):
            if bin_image[j, i] == 0:
                tmp[j] -= 1

    # Smooth out the projection array to get the maximas (which are spaces between lines)
    tmp = gaussian_filter1d(tmp, sigma)

    # Get the maximas
    maximas = argrelextrema(tmp, np.greater_equal, order=order)[0]

    # Split the image into lines(as coordinates)
    lines = []
    st = 0
    for i in maximas:
        if i > st + 1:
            lines.append((st, i))
        st = i
        # gray_image[i] = [0]*w

    # return the lines and a binary image
    return lines, bin_image


def find_connected(line_img, des_x, des_y, marking, path):
    """
    find all the connected cells, two cells are connected if their value is 0 and share a corner or an edge
    :param line_img:    (numpy.ndarray) a binary line image
    :param des_x:       (int) row index
    :param des_y:       (int) column index
    :param marking:     (numpy.ndarray) a matrix marks visited cells
    :param path:        (List[(int, int)]) to track visited cells
    :return: updated marking matrix and updated path after visit cell[des_x, des_y]
    """
    height, width = line_img.shape

    marking[des_x, des_y] = 1
    path = path + [(des_x, des_y)]

    for i in range(max(0, des_x - 1), min(des_x + 2, height)):
        for j in range(max(0, des_y - 1), min(des_y + 2, width)):
            if line_img[i, j] == 0 and marking[i, j] == 0:
                marking, path = find_connected(line_img, i, j, marking, path)
    return marking, path


def get_blobs(line_img):
    """
    Get destination of each blob in a line
    :param line_img:    (numpy.ndarray) a binary line image
    :return: coordinate of top-left corner and down-right corner, follow by row-column format.
    """
    h, w = line_img.shape
    marking = np.zeros((h, w))
    blobs = []

    for i in range(h):
        for j in range(w):
            path = []
            if line_img[i, j] == 0 and marking[i, j] == 0:
                marking, path = find_connected(line_img, i, j, marking, path)
                blobs = blobs + [path]

    blobs_des = []

    for blob in blobs:
        min_x, min_y, max_x, max_y = 999999999, 999999999, 0, 0
        for des in blob:
            min_x = min(min_x, des[0])
            min_y = min(min_y, des[1])
            max_x = max(max_x, des[0])
            max_y = max(max_y, des[1])
        bh = max_x - min_x
        bw = max_y - min_y
        if bh * bw > 25:
            blobs_des.append(((min_x, min_y), (max_x, max_y)))

    return blobs_des


def join_rectangles(rectangles, ratio: float = 0.5):
    rectangles = list(set(rectangles))

    for i in range(len(rectangles)):
        for j in range(i+1, len(rectangles)):
            ri = rectangles[i]
            rj = rectangles[j]
            cy = min(ri[1][1], rj[1][1]) - max(ri[0][1], rj[0][1])
            cx = min(ri[1][0], rj[1][0]) - max(ri[0][0], rj[0][0])
            # print(cy, cx)
            c = max(0, cx) * max(0, cy)
            si = (ri[1][1] - ri[0][1]) * (ri[1][0] - ri[0][0])
            sj = (rj[1][1] - rj[0][1]) * (rj[1][0] - rj[0][0])
            # print(c, si, sj)
            if c/ratio >= si or c/ratio >= sj:
                min_y = min(ri[0][1], rj[0][1])
                max_y = max(ri[1][1], rj[1][1])
                min_x = min(ri[0][0], rj[0][0])
                max_x = max(ri[1][0], rj[1][0])
                tmp = ((min_x, min_y), (max_x, max_y))
                rectangles[j] = tmp
                rectangles[i] = tmp
                return join_rectangles(rectangles, ratio)

    return rectangles


def word_segment(image, noise=False):
    """
    locate and extract words in a handwritten text image.
    :param noise: (Bool) median blur or not
    :param image: (numpy.ndarray) image of handwritten text
    :return: an image that has boxes surrounding each words, and a list of separated images of those words.
    """

    res_image = image.copy()
    if noise:
        image = cv2.medianBlur(image, ksize=3)
    (lines, bin_image) = split_lines(image)

    print(f'Number of lines: {len(lines)}')

    # Set up sigma values for the filter
    sy = 4
    sx = sy * 4

    # Make image with word blobs
    final = []
    for i, line in enumerate(lines):

        # Get the kernel
        wh = round((lines[i][1] - lines[i][0]) / 8)
        kernel = (get_anisotropic_gaussian_kernel(wh, sx, sx) + get_anisotropic_gaussian_kernel(wh, sy, sy)) / 2

        # Get the line image
        line_image = bin_image[lines[i][0]:lines[i][1]]

        # Convolve the line image with the kernel
        line_image = convolve2d(
            line_image,
            kernel,
            mode='same',
            boundary='fill',
            fillvalue=255
        )

        # Convert resulting image into binary image
        _, line_image = cv2.threshold(line_image, 254, 255, cv2.THRESH_BINARY)

        # if i == 0:
        #     show_img(line_image)

        # Get words coordinate
        words = get_blobs(line_image)
        for word in words:
            final.append((line, word))

    # print(f'Number of words: {len(final)}')
    word_images = []
    boxes = []

    for i, word in enumerate(final):
        top_left = (word[1][0][1], word[0][0] + word[1][0][0] - 5)
        down_right = (word[1][1][1], word[0][0] + word[1][1][0] + 5)

        boxes.append((top_left, down_right))
    # print(boxes)
    boxes = join_rectangles(boxes, ratio=0.7)

    for box in boxes:
        top_left = box[0]
        down_right = box[1]

        color = (0, 0, 255)
        thickness = 1
        res_image = cv2.rectangle(res_image, top_left, down_right, color, thickness)

        word_image = image[top_left[1]:down_right[1], top_left[0]:down_right[0]]
        word_images.append(word_image)

    return res_image, word_images
