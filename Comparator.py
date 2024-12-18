import cv2 as cv
import os
import numpy as np
import SearchMethods
from typing import List, Tuple, Optional

""" В этом модуле определён класс Compare для нахождения центра одного изображения на другом. """


class Compare:
    """ Класс Compare сравнивает два изображения подстилающей поверхности, фильтрует найденные общие КТ этих
        изображений, находит центр одного изображения на другом и отсеивает выбросы найденных центров. """
    good_match = 0
    filter_matches = 0
    center = None
    center_location = None
    key_1 = 0  # Кол-во контрольных точек основного изображения
    key_2 = 0  # Кол-во контрольных точек области видимости
    method = None  # Объект класса Method

    def __init__(self,
                 *,
                 img_1: np.ndarray,
                 kp_1: Tuple[cv.KeyPoint] | None,
                 des_1: np.ndarray | None,
                 height_main: int,
                 img_2: np.ndarray,
                 altitude: int,
                 method: SearchMethods.Method
                 ):
        """ Инициализация объекта класса Compare.
            :param img_1: Опорное изображение (numpy массив).
            :param kp_1: Список контрольных точек опорного изображения.
            :param des_1: Описание контрольных точек опорного изображения.
            :param height_main: Высота опорного снимка
            :param img_2: Изображение для сравнения (кадр после обработки).
            :param altitude: Высота снимка области видимости
            :param method: Объект метода для извлечения и сопоставления ключевых точек. """
        self.img1 = img_1
        self.kp1 = kp_1
        self.des1 = des_1
        self.height_map = height_main
        self.img_size = img_1.shape
        self.gray = img_2
        self.flight_altitude = altitude
        self.method = method

    @staticmethod
    def __find_area(good_matches: List[cv.DMatch], kp1: List[cv.KeyPoint]) -> List[List[int]]:
        """ Поиск списка координат общих контрольных точек на главном изображении.
            :param good_matches: Список хороших совпадений.
            :param kp1: Список контрольных точек основного изображения.
            :return: Список координат общих контрольных точек. """
        matches = []
        for i in range(len(good_matches)):
            dmatch = good_matches[i]
            large_image_kp = list(kp1[dmatch.queryIdx].pt)
            large_image_kp[0] = int(large_image_kp[0])
            large_image_kp[1] = int(large_image_kp[1])
            matches.append(large_image_kp)
        return matches

    @staticmethod
    def __location_images_2(good_matches: List[cv.DMatch], kp: List[cv.KeyPoint], matches_index: List[int]
                            ) -> List[List[int]]:
        """ Поиск списка координат общих контрольных точек на кадре после pixel_mask.
            :param good_matches: Список хороших совпадений.
            :param kp: Список контрольных точек на кадре.
            :param matches_index: Индексы совпадений для фильтрации.
            :return: Список координат общих контрольных точек. """
        matches = []
        for i in range(len(good_matches)):
            if i in matches_index:
                large_image_kp = list(kp[good_matches[i].trainIdx].pt)
                large_image_kp[0] = int(large_image_kp[0])
                large_image_kp[1] = int(large_image_kp[1])
                matches.append(large_image_kp)
        return matches

    @staticmethod
    def __transformation_matrix(main_matches: List[List[int]], matches_2: List[List[int]]) -> Optional[np.ndarray]:
        """ Вычисление матрицы преобразования координат.
             :param main_matches: Основные совпадения (координаты).
             :param matches_2: Совпадения на втором изображении (координаты).
             :return: Матрица гомографии или None в случае ошибки. """
        # Массивы с точками соответствия
        pts1 = np.float32([m for m in matches_2]).reshape(-1, 1, 2)
        pts2 = np.float32([m for m in main_matches]).reshape(-1, 1, 2)
        homo, _ = cv.findHomography(pts1, pts2, cv.RANSAC)
        return homo

    @staticmethod
    def __deleting_identical_points(main_matches: List[List[int]], crop_matches: List[List[int]]
                                    ) -> Tuple[List[List[int]], List[List[int]]] | Tuple[List]:
        """ Удаление одинаковых точек.
            :param main_matches: Основные совпадения (координаты).
            :param crop_matches: Совпадения на втором изображении (координаты).
            :return: Кортеж из уникальных основных и обрезанных совпадений. """
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

    def print_map(self, center2: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """ Отображение центра изображения области видимости и найденного положения центра области видимости на
            опорном изображении.
            :param center2: Координаты центра на изображении области видимости.
            :return: Изображение с отмеченным центром и изображение области видимости с центром."""
        color = (0, 0, 0)
        temp_main_img = self.img1.copy()
        radius = 10 if temp_main_img.shape[0] >= 1024 or temp_main_img.shape[1] >= 1024 else 5
        main_img = cv.circle(temp_main_img, self.center, radius=radius, color=color, thickness=radius * 2)
        radius = 8 if self.gray.shape[0] >= 1024 or self.gray.shape[1] >= 1024 else 3
        crop_img = cv.circle(self.gray, center2, radius=radius, color=color, thickness=radius * 2)
        return main_img, crop_img

    def __pixel_mask(self, matches: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """ Метод отфильтровывает КТ, которые удалены от медианного значения более чем на
            ширину изображения области видимости с учётом разницы высот.
            :param matches: Координаты контрольных точек главного изображения.
            :return: Кортеж из корректных совпадений и их индексов. """
        correct_matches = []
        correct_matches_index = []
        mask_correction = 1
        match_x = sorted(matches)
        match_y = sorted(matches, key=lambda y: y[1])

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

    def __true_center(self, img: np.ndarray, main_matches: List[List[int]], matches: List[List[int]]
                      ) -> Optional[List[int]]:
        """ Определяется положение центра области видимости на опорном кадре с помощью матрицы преобразования.
            Выбросы отсеиваются методом 'filtering_emissions()'.
            :param img: Изображение области видимости.
            :param main_matches: Основные совпадения (координаты).
            :param matches: Совпадения на втором изображении (координаты).
            :return: Координаты истинного центра области видимости на опорном изображении или None в случае ошибки."""
        crop_center = np.array([[img.shape[1] / 2, img.shape[0] / 2]], dtype='float32').reshape(-1, 1, 2)
        homo = self.__transformation_matrix(main_matches, matches)
        try:
            find_center = cv.perspectiveTransform(crop_center, homo)
            true_center = [round(float(find_center[0][0][0])), round(float(find_center[0][0][1]))]
            # Отсеивание выбросов
            true_center = self.__filtering_emissions(true_center, main_matches)
            return true_center
        except cv.error:
            print("Ошибка матрицы гомографии.\n")
            return None

    def __filtering_emissions(self, center: List[int], matches: List[List[int]]) -> List[int] | None:
        """ Метод отфильтровывает координаты, если
            они удалены от среднего значения более чем на ширину опорного изображения с учётом
            разницы высот.
            :param center: Координаты центра области видимости на опорном изображении.
            :param matches: Список совпадений для анализа выбросов.
            :return: Координаты центра или None при обнаружении выброса."""

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

    def comparator(self):
        """ Метод сравнивает два изображения подстилающей поверхности, фильтрует найденные общие КТ этих
            изображений, находит центр одного изображения на другом. Если найденный центр является выбросом,
            то метод сохранит в поле 'center' None, иначе относительные координаты в пикселях.
            Если координаты были найдены, то они будут выделены на изображении 'main_with_points.jpg'
            а само изображение сохранено в текущей директории. """
        kp2, des2 = self.method.get_kp_and_des(self.gray)

        if kp2 is None or len(kp2) > 3:
            if kp2 is None:
                self.kp1, kp2, good_matches = self.method.find_and_get_matches(img1=self.img1, img2=self.gray)
            else:
                _, _, good_matches = self.method.find_and_get_matches(des1=self.des1, des2=des2)
            self.key_1 = len(self.kp1)
            self.key_2 = len(kp2)

            if good_matches is not None:
                self.good_match = len(good_matches)
                main_matches = self.__find_area(good_matches, self.kp1)
                main_matches, matches_index = self.__pixel_mask(main_matches)
                matches_2 = self.__location_images_2(good_matches, kp2, matches_index)
                main_matches_filter, matches_2_filter = self.__deleting_identical_points(main_matches, matches_2)
                self.filter_matches = len(main_matches)
                if len(main_matches_filter) > 3:
                    self.center = self.__true_center(self.gray, main_matches_filter, matches_2_filter)
                    if self.center:
                        if os.path.exists("main_with_points.jpg"):
                            radius = 10 if self.img1.shape[0] > 1024 else 3
                            main_img_with_points = cv.circle(
                                cv.imread("main_with_points.jpg"),
                                self.center,
                                radius=radius,
                                color=(0, 0, 0),
                                thickness=radius * 2
                            )
                            cv.imwrite("main_with_points.jpg", main_img_with_points)
                        else:
                            cv.imwrite("main_with_points.jpg", self.img1)

    def get_data(self) -> List[int | bool | List[int] | None]:
        """ Получение данных о результатах сравнения.
            - key_1 - кол-во КТ на опорном изображении;
            - key_2 - кол-во КТ на области видимости;
            - good_match - кол-во общих КТ;
            - filter_matches - кол-во общих КТ после фильтров;
            - center - координаты найденного положения в пикселях. """
        return [self.key_1, self.key_2, self.good_match, self.filter_matches, self.center]
