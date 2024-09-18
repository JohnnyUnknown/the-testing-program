import cv2 as cv
import Comparator as Cmp
import Determ_coord as DC
import Preprocessing as Prep
import SearchMethods as SM
# import SuperPointMain as SP
# import demo_superpoint as DSP
import time
import math
# import torch


# Создание файла и заголовка протокола
def protocol_head(access):
    data = [
        " номер | Кол-во КТ опорного | Кол-во КТ обл. | Кол-во | Кол-во общих КТ | Вариант | Отклонение от",
        " п/п | изображения | видимости | общих КТ | после фильтра | искажения | истины (метров)"
    ]

    out_str = "-" * (sum(width) + 8) + "\n"

    for row in data:
        first_str, second_str, third_str, four_str, five_str, six_str, seven_str = row.split("|")
        out_str += ("|" + first_str.ljust(width[0]) + "|" + second_str.ljust(width[1]) + "|" + third_str.ljust(
            width[2]) + "|" + four_str.ljust(width[3]) +
                    "|" + five_str.ljust(width[4]) + "|    " + six_str.ljust(width[5] - 4) + "|" + seven_str.ljust(
                    width[6]) + "|" + "\n")

    out_str += "-" * (sum(width) + 8) + "\n"

    with open(name_protocol, access) as out_file:
        if access == "w":
            out_file.write(
                f"\t\tМетод {methods[method_index]}. Протокол прохода {name_crop} над изображением {name} с шагом {step} пикселей.\n\n")
        out_file.write(f"\nИмитация высоты полета на {flight_altitude} м. Высота опорного изображения: {height} м.\n")
        out_file.write(out_str)


def print_data(all_data, width_table):
    cnt = 0
    out_data = "|"
    for data in all_data:
        space = int((width_table[cnt] - len(str(data))) / 2)  # расчет центрального положения данных в протоколе
        temp_str = (" " * space + str(data))
        out_data += (temp_str.ljust(width_table[cnt]) + "|")
        cnt += 1
    return out_data


def print_line(width):
    out_data = "-" * (sum(width) + len(width) + 1) + "\n"
    return out_data


# Вычисление СКО предсказания от истинного значения
def fluctuation(center, center2):
    coord_find_point = determ_main.calculate(center)
    coord_center = determ_vision.calculate(center2)
    fluct = [round(coord_center[0] - coord_find_point[0], 6), round(coord_center[1] - coord_find_point[1], 6)]
    meters_dolong = round((6378137 * math.cos(float(coord_find_point[0]) * math.pi / 180) * 2 * math.pi) / 360, 3)
    meters_dolat = 111100
    fluct_meters_y = fluct[0] * meters_dolat
    fluct_meters_x = fluct[1] * meters_dolong
    fluct_meters = round(math.sqrt(fluct_meters_x * fluct_meters_x + fluct_meters_y * fluct_meters_y), 1)
    return fluct_meters


# Функция предобработки опорного изображения
def preprocess_main_image(big_map):
    # big_map = Prep.resize_img(big_map, 1920)
    big_map = Prep.gauss_improvement(big_map)
    # main_crop_img = resize_img(main_crop_img, 1024)
    return big_map


# Функция предобработки изображения области видимости
def preprocess_crop_image(crop_img, augmentation_index):
    # crop_img = Prep.resize_img(main_crop_img, 1024)
    # crop_img = Prep.gauss_improvement(crop_img)
    kernel = Prep.definition_of_blur(height, height_crop_img)
    crop_img = cv.GaussianBlur(crop_img, kernel, sigmaX=0, sigmaY=0)
    crop_img = Prep.augmentation(crop_img, augmentation_index)
    return crop_img


main_path = 'C:\\My\\Projects\\images\\main\\WK_00005-1.jpg'  # Опорное изображение
big_map = cv.imread(main_path, cv.IMREAD_GRAYSCALE)
crop_img_path = 'C:\\My\\Projects\\images\\main\\WK_00004-1.jpg'  # Изображение для взятия областей видимости
main_crop_img = cv.imread(crop_img_path, cv.IMREAD_GRAYSCALE)

# Параметры настройки
height = 500  # Высота снимка опорного изображения
height_crop_img = 400  # Высота снимка взятия областей видимости
dist_kf = 0.5  # Коэффициент точности сравнения опорных точек
height_difference_change = 5  # Коэффициент для изменения размеров области видимости
step = 2000  # Шаг смещения области видимости по опорному изображению
cycles = 2  # Кол-во циклов программы
method_index = 1  # 1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"

# Координаты углов опорного изображения
point_main1 = (48.245954, 46.164273)  # Левый верхний угол
point_main2 = (48.238956, 46.166415)  # Правый верхний угол
point_main3 = (48.238237, 46.160394)  # Нижний верхний угол
# Координаты углов изображения области видимости
point_view1 = (48.245465, 46.163374)  # Левый верхний угол
point_view2 = (48.239811, 46.165553)  # Правый верхний угол
point_view3 = (48.239240, 46.160377)  # Нижний верхний угол

# ---------------------------------------------------------------------------------------------------------

augmentation_index = 0  # Индекс типа искажения изображения
all_found_points = 0  # Всего найдено координат
found_points = 0  # Найдено координат на каждой высоте
height_difference = 0  # Разность высот снимков для изменения размеров области видимости
width = [7, 20, 16, 10, 17, 20, 16]  # Ширина столбцов протокола
width_statistic_table = [12, 18, 18, 18, 18, 18, 18, 18, 18]  # Ширина столбцов итоговой таблицы
iteration = 0  # Количество сравнений на каждой высоте
all_iter = 0  # Общее количество сравнений на всех высотах
general_percent_statistics = []
local_percent_statistic = ["Найдено %"]  # Процент найденных точек на каждой высоте
general_fluctuation_statistics = []
local_fluctuation_statistic = ["Отклонение"]  # Среднее СКО найденных точек на каждой высоте
methods = {1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"}

# Предобработка опорного изображения
big_map = preprocess_main_image(big_map)

determ_main = DC.Determ_coord(point_main1, point_main2, point_main3, big_map.shape)
determ_vision = DC.Determ_coord(point_view1, point_view2, point_view3, main_crop_img.shape)

start_program = time.perf_counter()

method = SM.Method(method_index, dist_kf)
kp, des = method.get_kp_and_des(big_map)

# Определение названия метода поиска контрольных точек и изображений. Формирование имени файла протокола
name = main_path.split('\\')[-1][:-4]
name_crop = crop_img_path.split("\\")[-1][:-4]
name_protocol = f"Protocols\\{methods[method_index]}_m-'{name}'_v-'{name_crop}'_step{step}_protocol.txt"

# Цикл по изменению размера области видимости
for i in range(cycles):
    height_difference += height_difference_change
    # Примерная высота полета
    if height_difference <= 1:
        height_difference = round(height / height_crop_img, 2)
    flight_altitude = int(height / height_difference)
    height_coefficient = round((flight_altitude / height_crop_img), 2)

    local_iter = 0  # Количество сравнений на текущей высоте
    local_found_points = 0  # Количество найденных точек на текущей высоте

    protocol_head("a" if i > 0 else "w")

    # Цикл по типу искажения области видимости
    while augmentation_index < len(Prep.augment):
        x1, y1 = 0, 0
        x2 = int(main_crop_img.shape[1] * height_coefficient)
        y2 = int(main_crop_img.shape[0] * height_coefficient)
        sum_local_fluct = 0

        # Проход по опорному изображению
        while y2 <= main_crop_img.shape[0]:
            local_iter += 1
            iteration += 1
            all_iter += 1
            crop_img = main_crop_img[y1:y2, x1:x2]

            # Предобработка изображения области видимости
            crop_img = preprocess_crop_image(crop_img, augmentation_index)

            # Сравнение изображений
            test = Cmp.Compare(big_map, kp, des, height, crop_img, flight_altitude, method_index, dist_kf)
            test.comparator()

            # Формирование списка с данными для внесения в протокол
            data_compare = test.get_data()
            # data_compare = SP.get_data(big_map, kp, des, crop_img, model)

            all_found_points += 1 if data_compare[4] else 0
            found_points += 1 if data_compare[4] else 0
            local_found_points += 1 if data_compare[4] else 0

            data_compare.pop(4)

            # Вычисление отклонения найденного местоположения от реального
            data_compare.insert(0, iteration)  # добавление номера итерации
            data_compare.insert(5, Prep.augment[augmentation_index])  # добавление индекса искажения
            center_vision = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
            data_compare[6] = "не найдено" if data_compare[6] == None else fluctuation(data_compare[6], center_vision)
            # Прибавление найденного СКО от истины для вычисления среднего
            sum_local_fluct = sum_local_fluct + data_compare[6] if data_compare[6] != "не найдено" else sum_local_fluct

            # Показ текущей области видимости с отмеченным центром
            if data_compare[6] != "не найдено":
                start_point = [center_vision[0] - x1, center_vision[1] - y1]
                end_point = [center_vision[0] - x1, center_vision[1] - y1]
                color = (0, 0, 255)
                thickness = 15
                img3 = cv.rectangle(crop_img, start_point, end_point, color, thickness)
                cv.imshow(f"crop image", img3)
                cv.waitKey(0)
                cv.destroyAllWindows()

            # Проход по изображению
            x1 += step
            x2 += step
            if x2 > main_crop_img.shape[1]:
                x1 = 0
                x2 = int(main_crop_img.shape[1] * height_coefficient)
                y1 += step
                y2 += step

            # Запись полученных данных в протокол
            with open(name_protocol, "a") as out_file:
                out_file.write(print_data(data_compare, width) + "\n")
        if local_iter > 0:
            with open(name_protocol, "a") as out_file:
                out_file.write(print_line(width))

        try:
            local_percent_statistic.append(
                f"{int(local_found_points / local_iter * 100)} % ({local_found_points} из {local_iter})")
        except ZeroDivisionError:
            local_percent_statistic.append("Сравнений не было!")

        try:
            local_fluctuation_statistic.append(f"{round(sum_local_fluct / local_found_points, 1)} м")
        except ZeroDivisionError:
            local_fluctuation_statistic.append("Не найдено")

        local_found_points = 0
        local_iter = 0
        augmentation_index += 1

    with (open(name_protocol, "a") as out_file):
        try:
            out_str = (f"Разница высот опорного изображения и области видимости: {round(height / flight_altitude, 1)} "
                       f"раз.\nИз {iteration} сравнений найдено координат: {found_points}. "
                       f"Процент нахождения - {round(found_points / iteration * 100, 1)} %\n\n")
        except ZeroDivisionError:
            out_str = (f"Разница высот опорного изображения и области видимости: {round(height / flight_altitude, 1)} "
                       f"раз.\nСравнений не было!\n\n")
        out_file.write(out_str)
        out_file.write(print_line(width))

    augmentation_index = 0
    iteration = 0
    found_points = 0
    general_percent_statistics.append(local_percent_statistic)
    local_percent_statistic = ["Найдено %"]
    general_fluctuation_statistics.append(local_fluctuation_statistic)
    local_fluctuation_statistic = ["Отклонение"]

finish = time.perf_counter()
minutes = round((finish - start_program) // 60)
seconds = round((finish - start_program) % 60)

# --------------------------------------------------------------------------------------------------------

# Формирование итогов проверки
with open(name_protocol, "a") as out_file:
    out_file.write(f"\n\nВремя выполнения программы: {minutes} мин. {seconds} сек.\n")
    out_file.write(
        f"Из {all_iter} сравнений найдено координат: {all_found_points}. Средний процент нахождения - "
        f"{round(all_found_points / all_iter * 100, 1)} %\n")

    average_percent = [0, 0, 0, 0, 0, 0, 0, 0]
    average_fluct = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    height_difference -= cycles * height_difference_change

    # Вычисление и печать итоговых таблиц проверки
    for i in range(cycles+1):
        # Заголовок таблицы
        if i < cycles:
            height_difference += height_difference_change  # (10 / (height / height_crop_img))
            flight_altitude = round((height / height_difference), 2)  # Примерная высота полета
            out_file.write(
                f"\n\nИмитация высоты полета на {flight_altitude} м. Разница высот с опорным изображением в "
                f"{round(height / flight_altitude, 1)} раз.\n")
        elif i == cycles:
            out_file.write(f"\n\nОБЩАЯ СТАТИСТИКА ПО ВСЕМ ВЫСОТАМ.\n")
        out_file.write(print_line(width_statistic_table))
        cnt = 0

        # Шапка таблицы
        heading = "|" + " " * width_statistic_table[cnt] + "|"
        for j in range(8):
            # расчет центрального положения данных в протоколе
            space = int((width_statistic_table[j + 1] - len(Prep.augment[j])) / 2)
            temp_str = (" " * space + Prep.augment[j])
            heading += temp_str.ljust(width_statistic_table[j + 1]) + "|"
        out_file.write(heading + "\n")
        out_file.write(print_line(width_statistic_table))

        # Внесение данных в итоговую таблицу по высотам
        if i < cycles:
            out_file.write(print_data(general_percent_statistics[i], width_statistic_table) + "\n")
            out_file.write(print_line(width_statistic_table))

            out_file.write(print_data(general_fluctuation_statistics[i], width_statistic_table) + "\n")
            out_file.write(print_line(width_statistic_table))

            # Суммирование данных для общей таблицы
            for j in range(1, 9):
                try:
                    average_percent[j - 1] += int(general_percent_statistics[i][j].split(" %")[0])
                except ValueError:
                    average_percent[j - 1] = 100
                if general_fluctuation_statistics[i][j] != "Не найдено":
                    average_fluct[j - 1][0] += float(general_fluctuation_statistics[i][j].split(" м")[0])
                    average_fluct[j - 1][1] += 1

        # Внесение данных в итоговую общую таблицу
        elif i == cycles:
            for j in range(8):
                average_percent[j] = str(round(average_percent[j] / cycles)) + " %"
                try:
                    average_fluct[j] = str(round(average_fluct[j][0] / average_fluct[j][1], 1)) + " м"
                except ZeroDivisionError:
                    average_fluct[j] = "Не Найдено"

            average_percent.insert(0, "Найдено %")
            average_fluct.insert(0, "Отклонение")

            out_file.write(print_data(average_percent, width_statistic_table) + "\n")
            out_file.write(print_line(width_statistic_table))

            out_file.write(print_data(average_fluct, width_statistic_table) + "\n")
            out_file.write(print_line(width_statistic_table))
