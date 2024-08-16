import cv2 as cv
import SIFT_method as cmp
import determ_coord as DC
import numpy as np
import time
import math


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
                    "|" + five_str.ljust(width[4]) + "|    " + six_str.ljust(width[5]-4) + "|" + seven_str.ljust(
                    width[6]) + "|" + "\n")

    out_str += "-" * (sum(width) + 8) + "\n"

    with open(name_protocol, access) as out_file:
        if access == "w":
            out_file.write(f"\t\t\tПротокол прохода {name_crop} над изображением {name} с шагом {step} пикселей.\n\n")
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


main_path = 'C:\\My\\Projects\\images\\main\\WK_00005-1.jpg'  # Опорное изображение
big_map = cv.imread(main_path, cv.IMREAD_GRAYSCALE)
crop_img_path = 'C:\\My\\Projects\\images\\main\\google.jpg'  # Изображение для взятия областей видимости
main_crop_img = cv.imread(crop_img_path, cv.IMREAD_GRAYSCALE)

# Параметры настройки
height = 500  # Высота снимка опорного изображения
height_crop_img = 1000  # Высота снимка взятия областей видимости
dist_kf = 0.48  # Коэффициент точности сравнения опорных точек
height_difference = (height / height_crop_img)  # Разность высот снимков для изменения размеров области видимости
height_difference_change = 5  # Коэффициент для изменения размеров области видимости
step = 300  # Шаг смещения области видимости по опорному изображению

augmentation_index = 0  # Индекс типа искажения изображения
all_found_points = 0  # Всего найдено координат
found_points = 0  # Найдено координат на каждой высоте
width = [7, 20, 16, 10, 17, 20, 16]  # Ширина столбцов протокола
iteration = 0   # Количество сравнений на каждой высоте
all_iter = 0    # Общее количество сравнений на всех высотах
general_percent_statistics = []
local_percent_statistic = ["Найдено %"]
general_fluctuation_statistics = []
local_fluctuation_statistic = ["Отклонение"]

augment = {
    0: "Без искажений", 1: "Поворот 60 град", 2: "Поворот -120 град", 3: "Поворот 180 град",
    4: "Яркость +30", 5: "Яркость -30", 6: "Добавление шумов", 7: "Размытость (5, 5)"
}

# Координаты углов опорного изображения
point_main1 = (48.245954, 46.164273)  # Левый верхний угол
point_main2 = (48.238956, 46.166415)  # Правый верхний угол
point_main3 = (48.238237, 46.160394)  # Нижний верхний угол

# Координаты углов изображения области видимости
point_view1 = (48.243964, 46.162271)    # Левый верхний угол
point_view2 = (48.241188, 46.163686)    # Правый верхний угол
point_view3 = (48.240549, 46.160974)    # Нижний верхний угол

determ_main = DC.Determ_coord(point_main1, point_main2, point_main3, big_map.shape)
determ_vision = DC.Determ_coord(point_view1, point_view2, point_view3, main_crop_img.shape)

start_program = time.perf_counter()

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(big_map, None)
big_map = gauss_improvement(big_map)


name = main_path.split('\\')[-1][:-4]
name_crop = crop_img_path.split("\\")[-1][:-4]
name_protocol = f"SIFT_m-'{name}'_v-'{name_crop}'_step{step}_protocol.txt"

for i in range(4):
    height_difference += height_difference_change
    flight_altitude = round((height / height_difference), 2)  # Примерная высота полета
    height_coefficient = round((flight_altitude / height_crop_img), 2)

    local_iter = 0  # Количество сравнений на каждой высоте при каждом искажении
    local_found_points = 0  # Количество найденных координат на каждой высоте при каждом искажении

    protocol_head("a" if i > 0 else "w")

    while augmentation_index < len(augment):

        x1, y1 = 0, 0
        x2 = int(main_crop_img.shape[1] * height_coefficient)
        y2 = int(main_crop_img.shape[0] * height_coefficient)
        local_fluct = 0

        # Проход по опорному изображению
        while y2 <= main_crop_img.shape[0]:
            local_iter += 1
            iteration += 1
            all_iter += 1
            crop_img = main_crop_img[y1:y2, x1:x2]
            crop_img = gauss_improvement(crop_img)
            crop_img = augmentation(crop_img, augmentation_index)
            # crop_img = cv.resize(crop_img, (300, 150), cv.INTER_NEAREST)

            # Сравнение изображений
            test = cmp.Compare(kp, des, height, big_map.shape, crop_img, flight_altitude, big_map)
            test.set_dist_kf(dist_kf)
            test.comparator()

            # Формирование списка с данными для внесения в протокол
            data_compare = test.get_data()

            all_found_points += 1 if data_compare[4] else 0
            found_points += 1 if data_compare[4] else 0
            local_found_points += 1 if data_compare[4] else 0

            data_compare.pop(4)

            # Вычисление отклонения найденного местоположения от реального
            data_compare.insert(0, iteration)  # добавление номера итерации
            data_compare.insert(5, augment[augmentation_index])  # добавление индекса искажения
            center_vision = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
            data_compare[6] = "не найдено" if data_compare[6] == None else fluctuation(data_compare[6], center_vision)
            local_fluct = local_fluct + data_compare[6] if data_compare[6] != "не найдено" else local_fluct

            # Показ текущей области видимости с отмеченным центром
            # if data_compare[6] != "не найдено":
            #     start_point = [center_vision[0]-x1, center_vision[1]-y1]
            #     end_point = [center_vision[0]-x1, center_vision[1]-y1]
            #     color = (0, 0, 255)
            #     thickness = 15
            #     img3 = cv.rectangle(crop_img, start_point, end_point, color, thickness)
            #     cv.imshow(f"crop image", img3)
            #     cv.waitKey(0)
            #     cv.destroyAllWindows()

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
            local_percent_statistic.append("Сравнений не произошло.")

        try:
            local_fluctuation_statistic.append(f"{round(local_fluct / local_found_points, 1)} м")
        except ZeroDivisionError:
            local_fluctuation_statistic.append("Не найдено")
        local_found_points = 0
        local_iter = 0
        augmentation_index += 1

    with (open(name_protocol, "a") as out_file):
        out_str = (f"Разница высот опорного изображения и области видимости: {round(height / flight_altitude, 1)} раз."
                   f"\nИз {iteration} сравнений найдено координат: {found_points}. "
                   f"Процент нахождения - {round(found_points / iteration * 100, 1)} %\n\n")
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
min = round((finish - start_program) // 60)
sec = round((finish - start_program) % 60)

width_statistic_table = [12, 18, 18, 18, 18, 18, 18, 18, 18]

# Формирование итогов проверки
with open(name_protocol, "a") as out_file:
    out_file.write(f"\n\nВремя выполнения программы: {min} мин. {sec} сек.\n")
    out_file.write(
        f"Из {all_iter} сравнений найдено координат: {all_found_points}. Средний процент нахождения - {round(all_found_points / all_iter * 100, 1)} %\n")

    average_percent = [0, 0, 0, 0, 0, 0, 0, 0]
    average_fluct = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    height_difference -= 4 * height_difference_change

    for i in range(5):

        # Заголовок таблицы
        if i < 4:
            height_difference += height_difference_change  # (10 / (height / height_crop_img))
            flight_altitude = round((height / height_difference), 2)  # Примерная высота полета
            out_file.write(
                f"\n\nИмитация высоты полета на {flight_altitude} м. Разница высот с опорным изображением в {round(height / flight_altitude, 1)} раз.\n")
        elif i == 4:
            out_file.write(f"\n\nОбщая статистика по всем высотам.\n")
        out_file.write(print_line(width_statistic_table))
        cnt = 0

        # Шапка таблицы
        heading = "|" + " " * width_statistic_table[cnt] + "|"
        for j in range(8):
            space = int((width_statistic_table[j + 1] - len(
                augment[j])) / 2)  # расчет центрального положения данных в протоколе
            temp_str = (" " * space + augment[j])
            heading += temp_str.ljust(width_statistic_table[j + 1]) + "|"
        out_file.write(heading + "\n")
        out_file.write(print_line(width_statistic_table))

        # Внесение данных в итоговую таблицу по высотам
        if i < 4:
            out_file.write(print_data(general_percent_statistics[i], width_statistic_table) + "\n")
            out_file.write(print_line(width_statistic_table))

            out_file.write(print_data(general_fluctuation_statistics[i], width_statistic_table) + "\n")
            out_file.write(print_line(width_statistic_table))

            # Суммирование данных для общей таблицы
            for j in range(1, 9):
                average_percent[j - 1] += int(general_percent_statistics[i][j].split(" %")[0])
                if general_fluctuation_statistics[i][j] != "Не найдено":
                    average_fluct[j - 1][0] += float(general_fluctuation_statistics[i][j].split(" м")[0])
                    average_fluct[j - 1][1] += 1

        # Внесение данных в итоговую общую таблицу
        elif i == 4:
            for j in range(8):
                average_percent[j] = str(round(average_percent[j] / 4)) + " %"
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
