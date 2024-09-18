import cv2 as cv
import numpy as np

augment = {0: "Без искажений", 1: "Поворот 60 град", 2: "Поворот -120 град", 3: "Поворот 180 град",
           4: "Яркость +30", 5: "Яркость -30", 6: "Добавление шумов", 7: "Размытость (5, 5)"}


# Методы аугментации изображений
def rotate_image(img, deg):
    height, width = img.shape[:2]
    center_x, center_y = (width / 2, height / 2)
    M = cv.getRotationMatrix2D((center_x, center_y), deg, 1.0)
    out_image = cv.warpAffine(img, M, (width, height))
    return out_image


def brightness(img, value):
    color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        v[v > abs(value)] -= abs(value)
        v[v <= abs(value)] = 0
    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def add_noise(img):
    noise = np.zeros(img.shape, np.uint8)
    cv.randn(noise, 0, 20)
    img_n = cv.add(img, noise)
    return img_n


def add_blur(img):
    img_bl = cv.blur(img, (5, 5))
    return img_bl


def augmentation(img, aug_index):
    out_img = img.copy()
    match aug_index:
        case 1:
            out_img = rotate_image(img, 60)
        case 2:
            out_img = rotate_image(img, -120)
        case 3:
            out_img = rotate_image(img, 180)
        case 4:
            out_img = brightness(img, 30)
        case 5:
            out_img = brightness(img, -30)
        case 6:
            out_img = add_noise(img)
        case 7:
            out_img = add_blur(img)
    return out_img


# Методы предобработки изображений
def clahe_improvement(img):
    clahe = cv.createCLAHE(2, (5, 5))
    # bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    # l, a, b = cv.split(lab)
    # l2 = clahe.apply(l)
    # lab = cv.merge((l2, a, b))
    # img2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    # img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img2 = clahe.apply(img)
    return img2


def gauss_improvement(img):
    # Дилатация (увеличение светлых пятен)
    # img2 = cv.dilate(img, (3, 3), iterations=1)

    # Эрозия (уменьшение светлых пятен)
    # img2 = cv.erode(img, (3, 3), iterations=1)

    img2 = cv.GaussianBlur(img, (5, 5), sigmaX=0, sigmaY=0)

    # Медианное размытие (при "царапинах" на изображении)
    # img2 = cv.medianBlur(img, 5)

    # Повышение резкости изображения
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Фильтр Собеля (обозначает контуры)
    # kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    # Фильтр лапласиан (более качественно обозначает контуры)
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # img2 = cv.filter2D(img, -1, kernel)
    return img2


def resize_img(img, new_width):
    new_height = int(img.shape[0] * (new_width / img.shape[1]))
    # Изменение размера изображения с сохранением пропорций
    resized_image = cv.resize(img, (new_width, new_height))
    return resized_image


def definition_of_blur(height, altitude):
    diff = int(height / altitude)
    if diff <= 5:
        return 5, 5
    elif diff % 2 == 1:
        return diff, diff
    else:
        return diff + 1, diff + 1
