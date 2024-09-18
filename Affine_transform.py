import cv2
import numpy as np


# Функция для применения аффинного преобразования
def affine_transform(img, tilt, rotate):
    # Получаем размеры изображения
    (h, w) = img.shape[:2]

    # Определяем центр изображения
    center = (w // 2, h // 2)

    # Вычисляем матрицу поворота
    M = cv2.getRotationMatrix2D(center, rotate, 1.0)

    # Вычисляем косинус и синус из матрицы поворота
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Вычисляем новые размеры холста после поворота
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Корректируем матрицу поворота с учетом смещения центра
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Применяем поворот к изображению с учетом новых размеров
    img_rot = cv2.warpAffine(img, M, (new_w, new_h))

    # Преобразование перспективы для моделирования наклона
    height, width = img_rot.shape[:2]

    # Параметры изменения перспективы
    d = tilt * 10  # Фактор масштабирования для наклона

    # Определяем исходные и целевые точки для перспективного преобразования
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_points = np.float32([[d, 0], [width - 1 - d, 0], [0, height - 1], [width - 1, height - 1]])

    # Получаем матрицу перспективного преобразования
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_transformed = cv2.warpPerspective(img_rot, matrix, (width, height))

    return img_transformed


def asift_detectAndCompute(img1, sift):
    # Определяем углы наклона и поворота для аффинных преобразований
    tilt_angles = [0, 5]  # Наклоны
    rotate_angles = [0, 90, 180]  # Повороты

    # Обработка изображения
    kp_asift, des_asift = [], []
    for tilt in tilt_angles:
        for rotate in rotate_angles:
            img_transformed = affine_transform(img1, tilt, rotate)
            kp, des = sift.detectAndCompute(img_transformed, None)
            kp_asift.extend(kp)
            if des is not None:
                des_asift.extend(des)

    # Преобразование дескрипторов в numpy массив
    des_asift = np.array(des_asift)

    return kp_asift, des_asift
