import cv2 as cv
import math
from decimal import *
import numpy as np


class Compare():
    good_match = 0
    filter_matches = 0
    center = None
    center_location = None
    key_1 = 0
    key_2 = 0
    dist_kf = 0.48

    def __init__(self, kp, des, height, img_size, img_2, height_2, img):
        self.kp1 = kp
        self.des1 = des
        self.height_map = height
        self.img_size = img_size
        self.gray = img_2
        self.flight_altitude = height_2
        self.img1 = img  # print_map

        # self.comparator()
        # self.get_data()

    # Поиск КТ изображения
    def search_KP(self, img):
        # Инициализация метода SIFT
        sift = cv.SIFT_create()
        # Поиск КТ и их дескрипторов
        kp, des = sift.detectAndCompute(img, None)
        return kp, des

    # Поиск общих КТ двух изображений
    def matcher(self, des1, des2):
        # Инициализация BFMatcher
        bf = cv.BFMatcher()
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=3)  # or pass empty dictionary
        # flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = bf.knnMatch(des1, des2, k=2)

        # Нахождение общих точек
        good = []
        for m, n in matches:
            if m.distance < self.dist_kf * n.distance:
                good.append(m)
        self.good_match = len(good)
        if len(good) >= 3:
            # Возврат списка общих КТ
            return good
        else:
            return None

    # Поиск списка координат общих КТ на главном изображении
    def find_area(self, good_matches, kp1):
        matches = []
        for i in range(len(good_matches)):
            dmatch = good_matches[i]
            # Поиск найденных КТ для обеих изображений в списке КТ главного изображения
            large_image_KP = list(kp1[dmatch.queryIdx].pt)
            large_image_KP[0] = int(large_image_KP[0])
            large_image_KP[1] = int(large_image_KP[1])
            # Добавление в список КТ главного изображения, совпадающих с КТ искомого
            matches.append(large_image_KP)
        return matches

    # Поиск прямоугольника, образующего искомую область на главном изображении
    def search_center(self, matches):
        list_x = []
        list_y = []
        for i in range(len(matches)):
            list_x.append(matches[i][0])
            list_y.append(matches[i][1])
        list_x.sort()
        list_y.sort()
        # Нахождение центральной точки искомого изображения на главном изображении
        center_x = int((list_x[0] + list_x[-1]) / 2)
        center_y = int((list_y[0] + list_y[-1]) / 2)
        return [center_x, center_y]

    # Отображение местоположения дрона на главном изображении
    def print_map(self, center):
        start_point = center
        end_point = center
        color = (0, 0, 255)
        thickness = 25
        img3 = cv.rectangle(self.img1, start_point, end_point, color, thickness)
        cv.imwrite(f"main.jpg", img3)

    # Маска проверки найденных КТ на карте
    def pixel_mask(self, matches):  # принимаются координаты КТ главного изображения
        correct_matches = []
        mask_correction = 2
        match_x = sorted(matches, key=lambda i: i[1])
        match_y = sorted(matches)

        if len(matches) % 2 == 0:
            indx1 = int(len(matches) / 2 - 1)
            indx2 = int(len(matches) / 2)
            median_x = (match_x[indx1][1] + match_x[indx2][1]) / 2
            median_y = (match_y[indx1][0] + match_y[indx2][0]) / 2
        else:
            indx = int((len(matches) - 1) / 2)
            median_x = match_x[indx][1]
            median_y = match_y[indx][0]

        # Нахождение коэффициента разницы высот полета и главного снимка для маски
        height_coefficient = int(self.height_map / self.flight_altitude)

        for i in range(len(matches)):
            if ((matches[i][0] >= median_y - self.img_size[1] / (height_coefficient * mask_correction))
                    and (matches[i][0] < median_y + self.img_size[1] / (height_coefficient * mask_correction))):
                if ((matches[i][1] >= median_x - self.img_size[0] / (height_coefficient * mask_correction))
                        and (matches[i][1] < median_x + self.img_size[0] / (height_coefficient * mask_correction))):
                    correct_matches.append(matches[i])
        return correct_matches

    def comparator(self):
        kp2, des2 = self.search_KP(self.gray)
        self.key_1 = len(self.kp1)
        self.key_2 = len(kp2)

        if len(kp2) > 2:
            good_matches = self.matcher(self.des1, des2)
            if good_matches != None:
                # поиск общих КТ на главном изображении
                main_matches = self.find_area(good_matches, self.kp1)
                # Сравнение найденных общих КТ с маской проверки
                main_matches = self.pixel_mask(main_matches)
                self.filter_matches = len(main_matches)
                if len(main_matches) > 2:
                    self.center = self.search_center(main_matches)
                    self.print_map(self.center)

    def get_data(self):
        find = True if self.center != None else False
        # key_1 - кол-во КТ на опорном изображении; key_2 - кол-во КТ на области видимости;
        # good_match - кол-во общих КТ; filter_matches - кол-во общих КТ после фильтра;
        # find - найдено ли местоположение; center - координаты
        return [self.key_1, self.key_2, self.good_match, self.filter_matches, find, self.center]

    def set_dist_kf(self, num):
        self.dist_kf = num