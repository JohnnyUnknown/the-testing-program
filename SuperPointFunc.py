import cv2
import torch
import numpy as np
import Preprocessing as Prep


def extract_keypoints(semi, threshold):
    """
    Извлекает ключевые точки из выхода SuperPoint.
    semi: тензор вероятностей ключевых точек (N x 65 x H/8 x W/8)
    threshold: порог для вероятности точки
    Возвращает:
    keypoints: список ключевых точек (cv2.KeyPoint)
    """
    # Убираем последний класс (класс "нет точки")
    heatmap = semi[:-1, :, :]
    # Применение softmax по оси каналов
    heatmap = torch.nn.functional.softmax(heatmap, dim=0).cpu().numpy()

    # Для каждой позиции выбираем максимум по каналам (направлениям)
    heatmap = np.max(heatmap, axis=0)

    # Non-maximum suppression (необязательно, можно для повышения точности)
    # Нахождение точек, которые превышают порог
    keypoints = np.argwhere(heatmap > threshold)

    # Умножаем координаты на 8, так как выходная карта имеет размер H/8 x W/8
    keypoints = keypoints[:, ::-1] * 8  # Меняем порядок на (x, y)
    return keypoints


def extract_descriptors(desc, keypoints, H, W):
    """
    Извлечение дескрипторов для каждой ключевой точки.
    desc: тензор дескрипторов, полученный от SuperPoint, форма (N x 256 x H/8 x W/8)
    keypoints: numpy массив ключевых точек (размерность 2 x N или N x 2)
    H, W: размеры оригинального изображения
    Возвращает:
    descriptors: numpy массив размерности N x 256 (где N — количество ключевых точек)
    """
    desc = desc[0].cpu().numpy()  # Извлечение дескрипторов из тензора
    Hc, Wc = desc.shape[1:3]  # Размер сетки H/8 и W/8

    # Преобразуем координаты ключевых точек обратно в пространство сетки H/8 x W/8
    keypoints_grid = [(int(kp[1] / (H / Hc)), int(kp[0] / (W / Wc))) for kp in keypoints.T]

    # Извлечение дескрипторов для каждой ключевой точки
    descriptors = np.array([desc[:, kp[0], kp[1]] for kp in keypoints_grid])

    return descriptors


# Приведение изображения к нужному формату (оттенки серого и нормализация)
def preprocess_image(img):
    new_size = 2560
    if img.shape[1] > new_size:
        img = Prep.resize_img(img, new_size)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img.astype(np.float32) / 255.0
    img_gray = np.expand_dims(img_gray, axis=0)  # Добавляем размерность канала
    img_gray = torch.from_numpy(img_gray).unsqueeze(0)  # Формат для torch (batch_size, 1, H, W)
    return img_gray


# Преобразование ключевых точек в формат OpenCV KeyPoint
def convert_to_cv_keypoints(keypoints):
    """Преобразование ключевых точек в формат OpenCV KeyPoint"""
    cv_keypoints = []
    for i in range(keypoints.shape[1]):
        x, y = keypoints[0, i], keypoints[1, i]
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=1.0)  # Указываем тип float для x и y
        cv_keypoints.append(kp)
    return cv_keypoints


# Функция для получения ключевых точек и дескрипторов
def get_keypoints_and_descriptors(img, model, threshold=0.015):
    # Предобработка изображения
    img_preprocessed = preprocess_image(img)
    with torch.no_grad():
        # Получение выхода модели
        semi, desc = model(img_preprocessed)
        # Извлечение вероятностной карты ключевых точек
        keypoints = extract_keypoints(semi[0], threshold)
        # Получаем x и y координаты ключевых точек
        keypoints = keypoints.T  # Преобразуем в (2, N), где N - количество точек

        H, W = img.shape[:2]  # Размеры исходного изображения
        descriptors = extract_descriptors(desc, keypoints, H, W)
        descriptors = descriptors.astype(np.float32)
        keypoints = convert_to_cv_keypoints(keypoints)

    return keypoints, descriptors


# Применение порога для отбора ключевых точек и дескрипторов
def filter_keypoints_and_descriptors(keypoints, descriptors, max_num_points=1000):
    num_points = keypoints.shape[1]
    if num_points > max_num_points:
        indices = np.random.choice(num_points, max_num_points, replace=False)
        filtered_keypoints = keypoints[:, indices]
        filtered_descriptors = descriptors[indices]
    else:
        filtered_keypoints = keypoints
        filtered_descriptors = descriptors
    return filtered_keypoints, filtered_descriptors
