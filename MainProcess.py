import cv2 as cv
import Comparator as Cmp
import Determ_coord as DC
import Preprocessing as Prep
import SearchMethods as SM
import math
from time import perf_counter


class MainProcess():
    stop_flag = False
    determ_main = None
    determ_vision = None
    flight_altitude = None
    width = [7, 16, 16, 10, 17, 15, 16]  # Ширина столбцов протокола
    width_statistic_table = [12, 13, 13, 13, 13, 13, 13, 13, 13]  # Ширина столбцов итоговой таблицы
    augmentation_index = 0  # Индекс типа искажения изображения
    all_found_points = 0  # Всего найдено координат
    found_points = 0  # Найдено координат на каждой высоте
    height_difference = 0  # Разность высот снимков для изменения размеров области видимости
    iteration = 0  # Количество сравнений на каждой высоте
    all_iter = 0  # Общее количество сравнений на всех высотах
    methods = {1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"}

    def __init__(self, *, path1, height1, path2, height2, dist_kf, height_diff, cycles, step, method, show,
                 coord1, coord2):
        self.main_path = path1
        self.crop_img_path = path2
        self.big_map = cv.imread(self.main_path, cv.IMREAD_GRAYSCALE)
        self.main_crop_img = cv.imread(self.crop_img_path, cv.IMREAD_GRAYSCALE)
        self.height = height1
        self.height_crop_img = height2
        self.dist_kf = dist_kf
        self.height_difference_change = height_diff
        self.cycles = cycles
        self.step = step
        self.method_index = method
        self.main_coordinates = coord1
        self.crop_img_coordinates = coord2
        self.show_image_flag = show

        # Определение названия метода поиска контрольных точек и изображений. Формирование имени файла протокола
        self.name = self.main_path.split('\\')[-1][:-4]
        self.name_crop = self.crop_img_path.split("\\")[-1][:-4]
        self.name_protocol = (f"Protocols\\{self.methods[self.method_index]}_m-'{self.name[-5:]}'"
                              f"_v-'{self.name_crop[-5:]}'_kf={self.dist_kf}_"
                              f"{self.cycles}x{self.height_difference_change}.txt")
        if self.method_index == 5:
            self.big_map = Prep.resize_img(self.big_map, 1024)
            self.main_crop_img = Prep.resize_img(self.main_crop_img, 1024)
        # Обновление изображения для отображения всех точек
        cv.imwrite("main_with_points.jpg", self.big_map)


    # Создание файла и заголовка протокола
    def protocol_head(self, access):
        data = [
            " Номер|   Кол-во КТ | Кол-во КТ обл. |  Кол-во | Кол-во общих КТ |  Вариант | Отклонение от",
            "  п/п | опорного изобр.|   видимости | общих КТ |  после фильтра | искажения | истины (метров)"
        ]

        out_str = "-" * (sum(self.width) + 8) + "\n"

        for row in data:
            first_str, second_str, third_str, four_str, five_str, six_str, seven_str = row.split("|")
            out_str += ("|" + first_str.ljust(self.width[0]) + "|" + second_str.ljust(self.width[1]) + "|" +
                        third_str.ljust(self.width[2]) + "|" + four_str.ljust(self.width[3]) +
                        "|" + five_str.ljust(self.width[4]) + "|  " + six_str.ljust(self.width[5] - 2) + "|" +
                        seven_str.ljust(self.width[6]) + "|" + "\n")

        out_str += "-" * (sum(self.width) + 8) + "\n"

        with open(self.name_protocol, access) as out_file:
            if access == "w":
                out_file.write(
                    f"\t\t\t\tП Р О Т О К О Л\n\t\tпроход по изображению {self.name_crop} и сравнение с {self.name} "
                    f"методом {self.methods[self.method_index]}.\n\n")
            out_file.write(
                f"\nИмитация высоты полета на {self.flight_altitude} м. Высота опорного изображения: {self.height} м.\n")
            out_file.write(out_str)

    def print_data(self, all_data, width_table):
        cnt = 0
        out_data = "|"
        for data in all_data:
            space = int((width_table[cnt] - len(str(data))) / 2)  # расчет центрального положения данных в протоколе
            temp_str = (" " * space + str(data))
            out_data += (temp_str.ljust(width_table[cnt]) + "|")
            cnt += 1
        return out_data

    def print_line(self, width):
        out_data = "-" * (sum(width) + len(width) + 1) + "\n"
        return out_data

    # Вычисление СКО предсказания от истинного значения
    def fluctuation(self, center, center2):
        coord_find_point = self.determ_main.calculate(center)
        coord_center = self.determ_vision.calculate(center2)
        fluct = [round(coord_center[0] - coord_find_point[0], 6), round(coord_center[1] - coord_find_point[1], 6)]
        meters_dolong = round((6378137 * math.cos(float(coord_find_point[0]) * math.pi / 180) * 2 * math.pi) / 360, 3)
        meters_dolat = 111100
        fluct_meters_y = fluct[0] * meters_dolat
        fluct_meters_x = fluct[1] * meters_dolong
        fluct_meters = round(math.sqrt(fluct_meters_x * fluct_meters_x + fluct_meters_y * fluct_meters_y), 1)
        return fluct_meters

    # Функция предобработки опорного изображения
    def preprocess_main_image(self, big_map):
        big_map = Prep.gauss_improvement(big_map)
        return big_map

    # Функция предобработки изображения области видимости
    def preprocess_crop_image(self, crop_img, augmentation_index):
        # crop_img = Prep.resize_img(main_crop_img, 1024)
        kernel = Prep.definition_of_blur(self.height, self.height_crop_img)
        crop_img = cv.GaussianBlur(crop_img, kernel, sigmaX=0, sigmaY=0)
        crop_img = Prep.augmentation(crop_img, augmentation_index)
        return crop_img

    # Формирование и распечатка итоговой статистики проверки
    def print_main_statistic(self, minutes, seconds, general_percent_statistics, general_fluctuation_statistics):
        with open(self.name_protocol, "a") as out_file:
            out_file.write(f"\n\nВремя выполнения программы: {minutes} мин. {seconds} сек.\n")
            out_file.write(
                f"Из {self.all_iter} сравнений найдено координат: {self.all_found_points}. Средний процент нахождения - "
                f"{round(self.all_found_points / self.all_iter * 100, 1)} %\n")

            total_percent = [0, 0, 0, 0, 0, 0, 0, 0]
            total_fluct = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            working_cycles = [0, 0, 0, 0, 0, 0, 0, 0]

            self.height_difference -= self.cycles * self.height_difference_change

            # Вычисление и печать итоговых таблиц проверки
            for i in range(self.cycles + 1):
                # Заголовок таблицы
                if i < self.cycles:
                    self.height_difference += self.height_difference_change  # (10 / (height / height_crop_img))
                    flight_altitude = round((self.height / self.height_difference), 2)  # Примерная высота полета
                    out_file.write(
                        f"\n\nИмитация высоты полета на {flight_altitude} м. Разница высот с опорным изображением в "
                        f"{round(self.height / flight_altitude, 1)} раз.\n")
                elif i == self.cycles:
                    out_file.write(f"\n\nОБЩАЯ СТАТИСТИКА ПО ВСЕМ ВЫСОТАМ.\n")
                out_file.write(self.print_line(self.width_statistic_table))
                cnt = 0

                # Шапка таблицы
                heading = "|" + " " * self.width_statistic_table[cnt] + "|"
                for j in range(8):
                    # расчет центрального положения данных в протоколе
                    space = int((self.width_statistic_table[j + 1] - len(Prep.augment[j])) / 2)
                    temp_str = (" " * space + Prep.augment[j])
                    heading += temp_str.ljust(self.width_statistic_table[j + 1]) + "|"
                out_file.write(heading + "\n")
                out_file.write(self.print_line(self.width_statistic_table))

                # Внесение данных в итоговую таблицу по высотам
                if i < self.cycles:
                    out_file.write(
                        self.print_data(general_percent_statistics[i], self.width_statistic_table) + "\n")
                    out_file.write(self.print_line(self.width_statistic_table))

                    out_file.write(
                        self.print_data(general_fluctuation_statistics[i], self.width_statistic_table) + "\n")
                    out_file.write(self.print_line(self.width_statistic_table))

                    # Суммирование данных для общей таблицы
                    for j in range(1, 9):
                        try:
                            percent = int(general_percent_statistics[i][j].split(" %")[0])
                            total_percent[j - 1] += percent
                            working_cycles[j - 1] += 1 if type(percent) is int else 0
                        except ValueError:
                            print("Value Error")
                        if general_fluctuation_statistics[i][j] != "Не найдено":
                            total_fluct[j - 1][0] += float(general_fluctuation_statistics[i][j].split(" м")[0])
                            total_fluct[j - 1][1] += 1

                # Внесение данных в итоговую общую таблицу
                elif i == self.cycles:
                    for j in range(8):
                        try:
                            total_percent[j] = str(round(total_percent[j] / working_cycles[j])) + " %"
                            total_fluct[j] = str(round(total_fluct[j][0] / total_fluct[j][1], 1)) + " м"
                        except ZeroDivisionError:
                            total_percent[j] = "Нет данных"
                            total_fluct[j] = "Не Найдено"

                    total_percent.insert(0, "Найдено %")
                    total_fluct.insert(0, "Отклонение")

                    out_file.write(self.print_data(total_percent, self.width_statistic_table) + "\n")
                    out_file.write(self.print_line(self.width_statistic_table))

                    out_file.write(self.print_data(total_fluct, self.width_statistic_table) + "\n")
                    out_file.write(self.print_line(self.width_statistic_table))

    def start_cycle(self):
        general_percent_statistics = []
        local_percent_statistic = ["Найдено %"]  # Процент найденных точек на каждой высоте
        general_fluctuation_statistics = []
        local_fluctuation_statistic = ["Отклонение"]  # Среднее СКО найденных точек на каждой высоте

        # Предобработка опорного изображения
        self.big_map = self.preprocess_main_image(self.big_map)

        # Создание карт по изображениям и угловым координатам
        self.determ_main = DC.Determ_coord(self.main_coordinates[0], self.main_coordinates[1],
                                           self.main_coordinates[2], self.big_map.shape)
        self.determ_vision = DC.Determ_coord(self.crop_img_coordinates[0], self.crop_img_coordinates[1],
                                             self.crop_img_coordinates[2], self.main_crop_img.shape)

        start_program = perf_counter()
        paused_time = 0

        # Выбор метода и определение особых точек на опорном изображении
        method = SM.Method(self.method_index, self.dist_kf)
        kp, des = method.get_kp_and_des(self.big_map)

        # Цикл по изменению размера области видимости
        for i in range(self.cycles):
            if self.stop_flag:
                break
            self.height_difference += self.height_difference_change
            # Примерная высота полета
            if self.height_difference <= 1:
                self.height_difference = round(self.height / self.height_crop_img, 2)
            self.flight_altitude = int(self.height / self.height_difference)
            height_coefficient = round((self.flight_altitude / self.height_crop_img), 2)

            local_iter = 0  # Количество сравнений на текущей высоте
            local_found_points = 0  # Количество найденных точек на текущей высоте

            self.protocol_head("a" if i > 0 else "w")

            # Цикл по типу искажения области видимости
            while self.augmentation_index < len(Prep.augment):
                if self.stop_flag:
                    break
                x1, y1 = 0, 0
                x2 = int(self.main_crop_img.shape[1] * height_coefficient)
                y2 = int(self.main_crop_img.shape[0] * height_coefficient)
                sum_local_fluct = 0

                # Проход по опорному изображению
                while y2 <= self.main_crop_img.shape[0]:
                    if self.stop_flag or x2 < 200 or y2 < 100:
                        break
                    local_iter += 1
                    self.iteration += 1
                    self.all_iter += 1
                    crop_img = self.main_crop_img[y1:y2, x1:x2]

                    # Предобработка изображения области видимости
                    crop_img = self.preprocess_crop_image(crop_img, self.augmentation_index)

                    # Сравнение изображений
                    test = Cmp.Compare(self.big_map, kp, des, self.height, crop_img, self.flight_altitude, method)
                    test.comparator()

                    # Формирование списка с данными для внесения в протокол
                    data_compare = test.get_data()

                    self.all_found_points += 1 if data_compare[4] else 0
                    self.found_points += 1 if data_compare[4] else 0
                    local_found_points += 1 if data_compare[4] else 0

                    data_compare.pop(4)

                    # Вычисление отклонения найденного местоположения от реального
                    data_compare.insert(0, self.iteration)  # добавление номера итерации
                    data_compare.insert(5, Prep.augment[self.augmentation_index])  # добавление индекса искажения
                    center_vision = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
                    data_compare[6] = "не найдено" if data_compare[6] == None else self.fluctuation(data_compare[6],
                                                                                                    center_vision)
                    # Прибавление найденного СКО от истины для вычисления среднего
                    sum_local_fluct = sum_local_fluct + data_compare[6] if data_compare[6] != "не найдено" \
                        else sum_local_fluct

                    # Запись полученных данных в протокол
                    with open(self.name_protocol, "a") as out_file:
                        out_file.write(self.print_data(data_compare, self.width) + "\n")

                    # Вывод изображений сравнения
                    if data_compare[6] != "не найдено" and self.show_image_flag and not self.stop_flag:
                        start_paused = perf_counter()
                        center_crop_image = [center_vision[0] - x1, center_vision[1] - y1]
                        main_image_for_show, crop_image_for_show = test.print_map(center_crop_image)
                        new_width = 1024
                        main_image_for_show = main_image_for_show if self.method_index == 5 \
                            else Prep.resize_img(main_image_for_show, new_width)
                        cv.imshow("Main image", main_image_for_show)

                        if crop_image_for_show.shape[1] < new_width:
                            new_width = crop_image_for_show.shape[1]
                        crop_image_for_show = crop_image_for_show if self.method_index == 5 \
                            else Prep.resize_img(crop_image_for_show, new_width)
                        cv.imshow("Crop image", crop_image_for_show)
                        cv.waitKey(0)
                        cv.destroyAllWindows()
                        end_paused = perf_counter()
                        paused_time += end_paused - start_paused

                    # Проход по изображению
                    x1 += crop_img.shape[1] + self.step
                    x2 += crop_img.shape[1] + self.step
                    if x2 > self.main_crop_img.shape[1]:
                        x1 = 0
                        x2 = int(self.main_crop_img.shape[1] * height_coefficient)
                        y1 += crop_img.shape[0] + self.step
                        y2 += crop_img.shape[0] + self.step

                if local_iter > 0:
                    with open(self.name_protocol, "a") as out_file:
                        out_file.write(self.print_line(self.width))

                try:
                    local_percent_statistic.append(
                        f"{int(local_found_points / local_iter * 100)} % ({local_found_points}/{local_iter})")
                except ZeroDivisionError:
                    local_percent_statistic.append("Нет данных")

                try:
                    local_fluctuation_statistic.append(f"{round(sum_local_fluct / local_found_points, 1)} м")
                except ZeroDivisionError:
                    local_fluctuation_statistic.append("Не найдено")

                local_found_points = 0
                local_iter = 0
                self.augmentation_index += 1

            with (open(self.name_protocol, "a") as out_file):
                try:
                    out_str = (
                        f"Разница высот опорного изображения и области видимости: {round(self.height / self.flight_altitude, 1)} "
                        f"раз.\nИз {self.iteration} сравнений найдено координат: {self.found_points}. "
                        f"Процент нахождения - {round(self.found_points / self.iteration * 100, 1)} %\n\n")
                except ZeroDivisionError:
                    out_str = (
                        f"Разница высот опорного изображения и области видимости: {round(self.height / self.flight_altitude, 1)} "
                        f"раз.\nОшибка размера изображения! Сравнений не производилось.\n\n")
                out_file.write(out_str)
                # self.temp_protocol += out_str
                out_file.write(self.print_line(self.width))

            self.augmentation_index = 0
            self.iteration = 0
            self.found_points = 0
            general_percent_statistics.append(local_percent_statistic)
            local_percent_statistic = ["Найдено %"]
            general_fluctuation_statistics.append(local_fluctuation_statistic)
            local_fluctuation_statistic = ["Отклонение"]

        finish = perf_counter()
        minutes = round((finish - start_program - paused_time) // 60)
        seconds = round((finish - start_program - paused_time) % 60)

        # Формирование и распечатка итоговой статистики проверки
        if not self.stop_flag:
            self.print_main_statistic(minutes, seconds, general_percent_statistics, general_fluctuation_statistics)


# Для запуска программы без GUI
# main_path = 'C:\\My\\Projects\\images\\main\\WK_00005-1.jpg'  # Опорное изображение
# big_map = None
# crop_img_path = 'C:\\My\\Projects\\images\\main\\WK_00004-1.jpg'  # Изображение для взятия областей видимости
# main_crop_img = None
#
# # Параметры настройки
# height = 500  # Высота снимка опорного изображения
# height_crop_img = 400  # Высота снимка взятия областей видимости
# dist_kf = 0.5  # Коэффициент точности сравнения опорных точек
# height_difference_change = 3  # Коэффициент для изменения размеров области видимости
# step = 1533  # Шаг смещения области видимости по опорному изображению
# cycles = 3  # Кол-во циклов программы
# method_index = 4  # 1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"
#
# # Координаты углов опорного изображения
# point_main1 = (48.245954, 46.164273)  # Левый верхний угол
# point_main2 = (48.238956, 46.166415)  # Правый верхний угол
# point_main3 = (48.238237, 46.160394)  # Нижний верхний угол
# # Координаты углов изображения области видимости
# point_view1 = (48.245465, 46.163374)  # Левый верхний угол
# point_view2 = (48.239811, 46.165553)  # Правый верхний угол
# point_view3 = (48.239240, 46.160377)  # Нижний верхний угол
# obj2 = MainProcess(
#     path1=main_path,
#     height1=height,
#     path2=crop_img_path,
#     height2=height_crop_img,
#     dist_kf=dist_kf,
#     height_diff=height_difference_change,
#     step=step,
#     cycles=cycles,
#     method=method_index,
#     show=False,
#     coord1=[point_main1, point_main2, point_main3],
#     coord2=[point_view1, point_view2, point_view3])
# obj2.start_cycle()
