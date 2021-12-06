import cv2
import sys
import WordSegmentation as ws


sys.setrecursionlimit(10000000)


def show_img(img, zoom=1.0):
    img = img.astype('uint8')
    img = cv2.resize(img, None, fx=zoom, fy=zoom)

    cv2.imshow('test', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    IMG_DIR = 'images'
    OUT_DIR = 'output'
    test_path = 'test10'

    image = cv2.imread(f'{IMG_DIR}/{test_path}.png')

    # if the picture is too noisy, noise = true
    box_image, words = ws.word_segment(image, noise=True, lightness=12, join_ratio=0.5)
    show_img(box_image, zoom=1)
    cv2.imwrite(f'{OUT_DIR}/{test_path}.png', box_image)
    print(len(words))
    # show_img(words[15], zoom=5)
