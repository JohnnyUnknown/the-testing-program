import cv2 as cv
import os
import numpy as np


class Compare():
    good_match = 0
    filter_matches = 0
    center = None
    center_location = None
    key_1 = 0  # Кол-во контрольных точек основного изображения
    key_2 = 0  # Кол-во контрольных точек области видимости
    method = None  # Объект класса Method

    def __init__(self, img, kp, des, height, img_2, altitude, method):
        self.img1 = img
        self.kp1 = kp
        self.des1 = des
        self.height_map = height
        self.img_size = img.shape
        self.gray = img_2
        self.flight_altitude = altitude
        self.method = method

    # Поиск списка координат общих КТ на главном изображении
    def find_area(self, good_matches, kp1):
        matches = []
        for i in range(len(good_matches)):
            dmatch = good_matches[i]
            large_image_KP = list(kp1[dmatch.queryIdx].pt)
            large_image_KP[0] = int(large_image_KP[0])
            large_image_KP[1] = int(large_image_KP[1])
            matches.append(large_image_KP)
        return matches

    # Поиск списка координат общих КТ на кадре после pixel_mask
    def location_images_2(self, good_matches, kp, matches_index):
        matches = []
        for i in range(len(good_matches)):
            if i in matches_index:
                large_image_KP = list(kp[good_matches[i].trainIdx].pt)
                large_image_KP[0] = int(large_image_KP[0])
                large_image_KP[1] = int(large_image_KP[1])
                matches.append(large_image_KP)
        return matches

    # Отображение местоположения дрона на главном изображении
    def print_map(self, center2):
        color = (0, 0, 0)
        temp_main_img = self.img1.copy()
        radius = 10 if temp_main_img.shape[0] > 1024 else 5
        main_img = cv.circle(
            temp_main_img,
            self.center,
            radius=radius,
            color=color,
            thickness=radius*2
        )
        radius = 8 if self.gray.shape[0] > 1024 else 3
        crop_img = cv.circle(
            self.gray,
            center2,
            radius=radius,
            color=color,
            thickness=radius*2
        )
        return main_img, crop_img

    # Маска проверки найденных КТ на карте
    def pixel_mask(self, matches):  # принимаются координаты КТ главного изображения
        correct_matches = []
        correct_matches_index = []
        mask_correction = 1
        match_x = sorted(matches)
        match_y = sorted(matches, key=lambda i: i[1])

        if len(matches) % 2 == 0:
            indx1 = int(len(matches) / 2 - 1)
            indx2 = int(len(matches) / 2)
            median_y = (match_y[indx1][1] + match_y[indx2][1]) / 2
            median_x = (match_x[indx1][0] + match_x[indx2][0]) / 2
        else:
            indx = int((len(matches) - 1) / 2)
            median_y = match_y[indx][1]
            median_x = match_x[indx][0]

        # Нахождение коэффициента разницы высот полета и главного снимка для маски
        height_coefficient = round(self.height_map / self.flight_altitude, 2)

        for i in range(len(matches)):
            if ((matches[i][0] >= median_x - self.img1.shape[1] / height_coefficient * mask_correction)
                    and (matches[i][0] <= median_x + self.img1.shape[1] / height_coefficient * mask_correction)):
                if ((matches[i][1] >= median_y - self.img1.shape[1] / height_coefficient * mask_correction)
                        and (matches[i][1] <= median_y + self.img1.shape[1] / height_coefficient * mask_correction)):
                    correct_matches.append(matches[i])
                    correct_matches_index.append(i)
        return correct_matches, correct_matches_index

    # Вычисление матрицы преобразования координат
    def transformation_matrix(self, main_matches, matches_2):
        # Массивы с точками соответствия
        pts1 = np.float32([m for m in matches_2]).reshape(-1, 1, 2)
        pts2 = np.float32([m for m in main_matches]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
        return H

    # Определение положения на опорном кадре с помощью матрицы преобразования
    def true_center(self, img, main_matches, matches):
        crop_center = np.array([[img.shape[1] / 2, img.shape[0] / 2]], dtype='float32').reshape(-1, 1, 2)
        H = self.transformation_matrix(main_matches, matches)
        try:
            find_center = cv.perspectiveTransform(crop_center, H)
            true_center = []
            true_center.append(round(find_center[0][0][0]))
            true_center.append(round(find_center[0][0][1]))
            # Отсеивание выбросов
            true_center = self.filtering_emissions(true_center, main_matches)
            return true_center

        except cv.error:
            print("Ошибка матрицы гомографии.\n")
            return None

    # Функция отсеивания выбросов
    def filtering_emissions(self, center, matches):
        # Нахождение коэффициента разницы высот полета и главного снимка для определения области возможного нахождения
        height_coefficient = round(self.height_map / self.flight_altitude, 2)

        mask_correction = 1
        match_x = sorted(matches)
        match_y = sorted(matches, key=lambda i: i[1])

        # Среднее значение центра по крайним точкам
        median_x = int((match_x[0][0] + match_x[-1][0]) / 2)
        median_y = int((match_y[0][1] + match_y[-1][1]) / 2)

        k = 1
        if (((center[0] < median_x - self.img1.shape[k] / height_coefficient * mask_correction)
             or (center[0] > median_x + self.img1.shape[k] / height_coefficient * mask_correction))
                or ((center[1] < median_y - self.img1.shape[k] / height_coefficient * mask_correction)
                    or (center[1] > median_y + self.img1.shape[k] / height_coefficient * mask_correction))):
            print(f"Emission found: {center=}")
            return None
        return center

    # Удаление одинаковых точек
    def deleting_identical_points(self, main_matches, crop_matches):
        matches_1, matches_2 = [], []
        for i in range(len(main_matches)):
            flag = True
            for j in range(len(matches_1)):
                if main_matches[i] == matches_1[j] and crop_matches[i] == matches_2[j]:
                    flag = False
                    break
            if flag:
                matches_1.append(main_matches[i])
                matches_2.append(crop_matches[i])

        if len(matches_1) > 3:
            return matches_1, matches_2
        else:
            return [], []

    def comparator(self):
        kp2, des2 = self.method.get_kp_and_des(self.gray)

        if kp2 == None or len(kp2) > 3:
            if kp2 == None:
                self.kp1, kp2, good_matches = self.method.find_and_get_matches(img1=self.img1, img2=self.gray)
            else:
                _, _, good_matches = self.method.find_and_get_matches(des1=self.des1, des2=des2)
            self.key_1 = len(self.kp1)
            self.key_2 = len(kp2)

            if good_matches != None:
                self.good_match = len(good_matches)
                main_matches = self.find_area(good_matches, self.kp1)
                main_matches, matches_index = self.pixel_mask(main_matches)
                matches_2 = self.location_images_2(good_matches, kp2, matches_index)
                main_matches_filter, matches_2_filter = self.deleting_identical_points(main_matches, matches_2)
                self.filter_matches = len(main_matches)
                if len(main_matches_filter) > 3:
                    self.center = self.true_center(self.gray, main_matches_filter, matches_2_filter)
                    if self.center:
                        if os.path.exists("main_with_points.jpg"):
                            radius = 10 if self.img1.shape[0] > 1024 else 3
                            main_img_with_points = cv.circle(
                                cv.imread("main_with_points.jpg"),
                                self.center,
                                radius=radius,
                                color=(0, 0, 0),
                                thickness=radius*2
                            )
                            cv.imwrite("main_with_points.jpg", main_img_with_points)
                        else:
                            cv.imwrite("main_with_points.jpg", self.img1)

    def get_data(self):
        # key_1 - кол-во КТ на опорном изображении; key_2 - кол-во КТ на области видимости;
        # good_match - кол-во общих КТ; filter_matches - кол-во общих КТ после фильтра;
        # find - найдено ли местоположение; center - координаты
        return [self.key_1, self.key_2, self.good_match, self.filter_matches, bool(self.center), self.center]
