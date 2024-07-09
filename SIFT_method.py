import cv2 as cv
from decimal import *
import numpy as np

class Compare():
    good_match = 0
    filter_matches = 0
    center = None
    center_location = None
    key_1 = 0
    key_2 = 0

    def __init__(self, kp, des, height, img_size, img_2, height_2, img, iter):
        self.kp1 = kp
        self.des1 = des
        self.flight_altitude = height_2
        self.height_map = height
        self.img_size = img_size
        self.img1 = img
        self.iter = iter

        # self.img1 = cv.imread(self.path_main, cv.IMREAD_GRAYSCALE)
        self.gray = img_2

        self.comparator()
        self.get_data()

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
        matches = bf.knnMatch(des1, des2, k=2)

        # Нахождение общих точек
        good = []
        for m, n in matches:
            if m.distance < 0.43 * n.distance:
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
        thickness = 30
        img3 = cv.rectangle(self.img1, start_point, end_point, color, thickness)
        # name = self.path_main.split('\\')[-1][:-4]
        cv.imwrite(f"main.jpg", img3)

    # Определение географических координат местоположения дрона
    def determ_coordinates(self, img, center):
        y, x = img.shape
        # MapFree cam5
        first_x = Decimal(46.159330)
        first_y = Decimal(48.238724)
        second_x = Decimal(46.164232)
        second_y = Decimal(48.245512)

        pixel_price_x = Decimal((second_x - first_x) / x)
        pixel_price_y = Decimal((first_y - second_y) / y)
        center_coord = [round((pixel_price_x*center[0]+first_x), 6), round(first_y-(pixel_price_y*center[1]), 6)]
        self.center = center_coord

    # Маска проверки найденных КТ на карте
    def pixel_mask(self, matches):    # принимаются координаты КТ главного изображения
        # print(matches[0], self.img_size)
        correct_matches = []
        mask_correction = 2
        sum_x, sum_y = 0, 0
        for i in range(len(matches)):
            sum_x += matches[i][0]
            sum_y += matches[i][1]
        medium_x = int(sum_x / len(matches))
        medium_y = int(sum_y / len(matches))
        # Нахождение коэффициента разницы высот полета и главного снимка для маски
        height_coefficient = int(self.height_map / self.flight_altitude)

        for i in range(len(matches)):
            if ((matches[i][1] >= medium_y - self.img_size[0] / (height_coefficient * mask_correction))
                    and (matches[i][1] < medium_y + self.img_size[0] / (height_coefficient * mask_correction))):
                if ((matches[i][0] >= medium_x - self.img_size[1] / (height_coefficient * mask_correction))
                        and (matches[i][0] < medium_x + self.img_size[1] / (height_coefficient * mask_correction))):
                    correct_matches.append(matches[i])
        return correct_matches

    def comparator(self):
        # kp1, des1 = self.search_KP(self.img1)
        kp2, des2 = self.search_KP(self.gray)
        self.key_1 = len(self.kp1)
        self.key_2 = len(kp2)

        if len(kp2) > 1:
            good_matches = self.matcher(self.des1, des2)
            if good_matches != None:
                # поиск общих КТ на главном изображении
                main_matches = self.find_area(good_matches, self.kp1)
                # Сравнение найденных общих КТ с маской проверки
                main_matches = self.pixel_mask(main_matches)
                self.filter_matches = len(main_matches)
                if len(main_matches) > 1:
                    center = self.search_center(main_matches)
                    self.center = center
                    # self.print_map(center)

                # self.determ_coordinates(self.img1, center)

    def get_data(self):
        find = True if self.center != None else False
        # print(self.center_location)
        # key_1 - кол-во КТ на опорном изображении; key_2 - кол-во КТ на области видимости;
        # good_match - кол-во общих КТ; filter_matches - кол-во общих КТ после фильтра;
        # find - найдено ли местоположение; center - координаты
        return [self.key_1, self.key_2, self.good_match, self.filter_matches, find, self.center]