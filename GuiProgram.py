from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import sys
import os
import MainProcess as MP


# Определение потока для выполнения вычислений
class Worker(QThread):
    finished = pyqtSignal()  # Сигнал для завершения работы

    def __init__(self, main_process):
        super().__init__()
        self.main_process = main_process

    def run(self):
        # Выполнение вычислений цикла
        self.main_process.start_cycle()
        self.finished.emit()  # Отправляем сигнал о завершении

    def stop(self):
        self.main_process.stop_flag = True


class Program(QWidget):
    main_process = None
    metka_stop = False  # Для вывода информации об остановке процесса
    coord1 = []
    coord2 = []

    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Поля для заполнения правой стороны
        layouts_for_right = []
        for i in range(9):
            if i == 0:
                layout = QGridLayout()
                layouts_for_right.append(layout)
            else:
                layout = QHBoxLayout()
                layouts_for_right.append(layout)
            right_layout.addLayout(layouts_for_right[i])

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Устанавливаем основной макет для виджета
        self.setLayout(main_layout)

        # Настройки окна
        self.setWindowTitle('Программа тестирования алгоритмов')
        self.setGeometry(0, 0, 1280, 720)

        label_left = QLabel("Поле вывода информации")
        label_left.setStyleSheet("font-size: 14px; font-weight: bold;")
        left_layout.addWidget(label_left)
        left_layout.setAlignment(label_left, Qt.AlignCenter)

        # Текстовое поле для вывода информации
        self.text_output = QTextEdit(self)
        self.text_output.setMinimumWidth(930)
        font = QFont("Lucida Console", 9, 400)
        self.text_output.setCurrentFont(font)
        self.text_output.setReadOnly(True)
        left_layout.addWidget(self.text_output)

        # ---------------------------Имена изображений и высоты--------------------------------

        # Правая часть программы
        label_path = QLabel("Путь к основному изображению", self)
        label_path.setFixedWidth(300)
        label_path.setMaximumHeight(30)
        layouts_for_right[0].addWidget(label_path, 0, 0, 1, 6)
        self.main_path = QLineEdit(self)
        self.main_path.setMinimumWidth(300)
        self.main_path.setMinimumHeight(25)
        self.main_path.textChanged.connect(self.clear_main_height)
        self.main_path.setStyleSheet("font-size: 14px; font-weight: 450;")
        layouts_for_right[0].addWidget(self.main_path, 1, 0, 1, 6)

        label_path2 = QLabel("Путь к изображению области видимости", self)
        label_path2.setMinimumWidth(300)
        label_path2.setMaximumHeight(30)
        layouts_for_right[0].addWidget(label_path2, 3, 0, 1, 6)
        self.second_path = QLineEdit(self)
        self.second_path.setMinimumWidth(300)
        self.second_path.setMinimumHeight(25)
        self.second_path.textChanged.connect(self.clear_crop_height)
        self.second_path.setStyleSheet("font-size: 14px; font-weight: 450;")
        layouts_for_right[0].addWidget(self.second_path, 4, 0, 1, 6)

        # "Промежуток"
        lab = QLabel()
        lab.setMaximumHeight(50)
        layouts_for_right[0].addWidget(lab, 2, 0, 1, 10)

        label_height1 = QLabel("Высота снимка 1, м", self)
        label_height1.setFixedWidth(100)
        label_height1.setMaximumHeight(30)
        layouts_for_right[0].addWidget(label_height1, 0, 7, 1, 2)
        self.height = QLineEdit(self)
        self.height.setFixedWidth(100)
        self.height.setMinimumHeight(25)
        self.height.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.height.setAlignment(Qt.AlignCenter)
        layouts_for_right[0].addWidget(self.height, 1, 7, 1, 2)

        label_height2 = QLabel("Высота снимка 2, м", self)
        label_height2.setFixedWidth(100)
        label_height2.setMaximumHeight(30)
        layouts_for_right[0].addWidget(label_height2, 3, 7, 1, 2)
        self.height_crop = QLineEdit(self)
        self.height_crop.setFixedWidth(100)
        self.height_crop.setMaximumHeight(25)
        self.height_crop.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.height_crop.setAlignment(Qt.AlignCenter)
        layouts_for_right[0].addWidget(self.height_crop, 4, 7, 1, 2)

        # -----------------------------Координаты углов изображений--------------------------------

        coord_info_label = QLabel("Три угловые координаты изображений")
        coord_info_label.setFixedHeight(20)
        coord_info_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        layouts_for_right[0].addWidget(coord_info_label, 5, 2, 1, 5)
        coord_left_label = QLabel("Основное изображение:")
        coord_left_label.setFixedHeight(20)
        layouts_for_right[0].addWidget(coord_left_label, 6, 0, 1, 4)
        coord_right_label = QLabel("Изображение области видимости:")
        coord_right_label.setFixedHeight(20)
        layouts_for_right[0].addWidget(coord_right_label, 6, 5, 1, 4)

        self.main_coord_1 = QLineEdit(self)
        self.main_coord_1.setFixedWidth(220)
        self.main_coord_1.setStyleSheet("font-size: 14px; font-weight: normal;")
        self.main_coord_1.setAlignment(Qt.AlignCenter)
        layouts_for_right[0].addWidget(self.main_coord_1, 7, 0, 1, 4)

        self.main_coord_2 = QLineEdit(self)
        self.main_coord_2.setAlignment(Qt.AlignCenter)
        self.main_coord_2.setStyleSheet("font-size: 14px; font-weight: normal;")
        layouts_for_right[0].addWidget(self.main_coord_2, 8, 0, 1, 4)

        self.main_coord_3 = QLineEdit(self)
        self.main_coord_3.setAlignment(Qt.AlignCenter)
        self.main_coord_3.setStyleSheet("font-size: 14px; font-weight: normal;")
        layouts_for_right[0].addWidget(self.main_coord_3, 9, 0, 1, 4)

        self.crop_coord_1 = QLineEdit(self)
        self.crop_coord_1.setFixedWidth(220)
        self.crop_coord_1.setStyleSheet("font-size: 14px; font-weight: normal;")
        self.crop_coord_1.setAlignment(Qt.AlignCenter)
        layouts_for_right[0].addWidget(self.crop_coord_1, 7, 5, 1, 4)

        self.crop_coord_2 = QLineEdit(self)
        self.crop_coord_2.setAlignment(Qt.AlignCenter)
        self.crop_coord_2.setStyleSheet("font-size: 14px; font-weight: normal;")
        layouts_for_right[0].addWidget(self.crop_coord_2, 8, 5, 1, 4)

        self.crop_coord_3 = QLineEdit(self)
        self.crop_coord_3.setAlignment(Qt.AlignCenter)
        self.crop_coord_3.setStyleSheet("font-size: 14px; font-weight: normal;")
        layouts_for_right[0].addWidget(self.crop_coord_3, 9, 5, 1, 4)

        # ---------------------------Параметры проверки первая линия-------------------------------

        label5 = QLabel("Параметры проверки:")
        label5.setMinimumHeight(50)
        label5.setMaximumHeight(100)
        label5.setStyleSheet("font-size: 14px; font-weight: bold;")
        layouts_for_right[1].addWidget(label5)
        layouts_for_right[1].setAlignment(label5, Qt.AlignHCenter)

        # Настройки программы первая линия
        label_height_diff = QLabel("Увеличение. раз/цикл")
        label_height_diff.setMinimumWidth(110)
        label_height_diff.setMaximumHeight(30)
        label_step = QLabel("Смещение (пиксели)")
        label_step.setMinimumWidth(110)
        label_step.setMaximumHeight(30)
        label_cycles = QLabel("Кол-во циклов")
        label_cycles.setMinimumWidth(110)
        label_cycles.setMaximumHeight(30)
        label_imshow = QLabel("Точность сравнения")
        label_imshow.setMinimumWidth(110)
        label_imshow.setMaximumHeight(30)
        layouts_for_right[2].addWidget(label_height_diff)
        layouts_for_right[2].addWidget(label_step)
        layouts_for_right[2].addWidget(label_cycles)
        layouts_for_right[2].addWidget(label_imshow)

        self.height_diff = QComboBox(self)
        self.height_diff.addItems(["1", "2", "3", "4", "5"])
        self.height_diff.setMinimumWidth(110)
        self.height_diff.setMaximumWidth(120)
        self.height_diff.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.step = QLineEdit(self)
        self.step.setMinimumWidth(110)
        self.step.setMaximumWidth(120)
        self.step.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.step.setAlignment(Qt.AlignCenter)
        self.cycles = QComboBox(self)
        self.cycles.setMinimumWidth(110)
        self.cycles.setMaximumWidth(120)
        self.cycles.addItems(["1", "2", "3", "4", "5"])
        self.cycles.setStyleSheet("font-size: 14px; font-weight: 500;")
        # self.cycles.setAlignment(Qt.AlignCenter)
        self.dist = QLineEdit(self)
        self.dist.setMinimumWidth(100)
        self.dist.setMaximumWidth(120)
        self.dist.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.dist.setAlignment(Qt.AlignCenter)
        layouts_for_right[3].addWidget(self.height_diff)
        layouts_for_right[3].addWidget(self.step)
        layouts_for_right[3].addWidget(self.cycles)
        layouts_for_right[3].addWidget(self.dist)

        # "Промежуток"
        lab2 = QLabel()
        lab2.setMaximumHeight(50)
        layouts_for_right[4].addWidget(lab2)

        # ------------------------------Параметры проверки вторая линия------------------------------

        # Настройки программы вторая линия
        label_step = QLabel("Выбор метода")
        label_step.setMinimumWidth(110)
        label_step.setMaximumHeight(30)
        layouts_for_right[5].addWidget(label_step)

        self.method = QComboBox(self)
        self.method.addItems(["SIFT", "AKAZE", "ORB", "ASIFT", "SuperPoint"])
        self.method.setMinimumWidth(100)
        self.method.setMaximumWidth(120)
        self.method.setStyleSheet("font-size: 14px; font-weight: 450;")
        self.start = QPushButton("START", self)
        self.start.setMinimumWidth(100)
        self.start.setMaximumWidth(120)
        self.start.setMinimumHeight(50)
        self.start.clicked.connect(self.start_program)
        self.start.setStyleSheet("background-color: green; color: white; font-size: 14px; font-weight: 500;")
        self.stop = QPushButton("STOP", self)
        self.stop.setMinimumWidth(100)
        self.stop.setMaximumWidth(120)
        self.stop.setMinimumHeight(50)
        self.stop.clicked.connect(self.stop_program)
        self.stop.setStyleSheet("background-color: red; color: white; font-size: 14px; font-weight: 500;")

        layouts_for_right[6].addWidget(self.method)
        layout_plug = QHBoxLayout()
        self.imshow = QRadioButton("Вывод \nизображений", self)
        self.imshow.toggled.connect(self.show_images)
        plug3 = QLabel()
        plug3.setFixedWidth(15)
        layout_plug.addWidget(plug3)
        layout_plug.addWidget(self.imshow)
        layouts_for_right[6].addLayout(layout_plug)
        layouts_for_right[6].addWidget(self.start)
        layouts_for_right[6].addWidget(self.stop)

        self.label_process = QLabel("Текущее состояние...")
        self.label_process.setFixedHeight(70)
        self.label_process.setStyleSheet("font-size: 12px; font-weight: 400;")
        layouts_for_right[7].addWidget(self.label_process)
        layouts_for_right[7].setAlignment(self.label_process, Qt.AlignHCenter)

        # "Промежуток"
        layouts_for_right[8].addWidget(QLabel())

        self.load_program_state()

    def show_images(self):
        if self.main_process:
            self.main_process.show_image_flag = self.imshow.isChecked()

    def update_text_edit(self):
        self.text_output.clear()
        # Обновляем содержимое QTextEdit
        name_protocol = self.main_process.name_protocol
        try:
            with open(name_protocol, "r") as out_file:
                states = out_file.readlines()
        except FileNotFoundError:
            pass
        except IndexError:
            os.remove("Program state.txt")
        for state in states:
            self.text_output.append(state)

    def clear_main_height(self):
        self.height.clear()

    def clear_crop_height(self):
        self.height_crop.clear()

    def start_program(self):
        if not self.checking_paths():
            if not self.checking_empty_fields():
                self.metka_stop = False
                if self.worker == None or self.worker.isFinished():
                    self.save_program_state()

                    self.main_process = MP.MainProcess(self.main_path.text(), int(float(self.height.text())),
                                                       self.second_path.text(),
                                                       int(float(self.height_crop.text())), float(self.dist.text()),
                                                       self.height_diff.currentIndex() + 1,
                                                       int(float(self.step.text())), self.cycles.currentIndex() + 1,
                                                       self.method.currentIndex() + 1,
                                                       self.imshow.isChecked(), self.coord1, self.coord2)

                    # Создаем процесс Worker и подключаем сигналы
                    self.worker = Worker(self.main_process)
                    self.worker.finished.connect(self.on_finished)

                if self.worker.isRunning() == False:
                    self.worker.start()
                    self.label_process.setText("Выполняется...")
                else:
                    self.show_message("Программа уже запущена")
            else:
                self.show_message("Не все поля правильно заполнены.")
        else:
            self.show_message("Указан неверный путь к файлу!")

    def stop_program(self):
        self.metka_stop = True
        if self.worker == None or not self.worker.isRunning():
            self.show_message("Нет запущенных вычислений.")
        else:
            if self.worker:
                self.worker.stop()  # Останавливаем поток
                self.worker.quit()
                self.worker.wait()  # Ждем завершения потока
                self.update_text_edit()

    def on_finished(self):
        self.worker.quit()
        self.worker.wait()
        self.update_text_edit()
        finish_text = "Выполнено." if not self.metka_stop else "Выполнение остановлено."
        self.label_process.setText(finish_text)

    def save_program_state(self):
        state_str = (f"{self.main_path.text()}\n{int(float(self.height.text()))}\n{self.second_path.text()}\n"
                     f"{int(float(self.height_crop.text()))}\n{self.dist.text()}\n{self.height_diff.currentText()}\n"
                     f"{int(float(self.step.text()))}\n{self.cycles.currentText()}\n"
                     f"{[self.method.currentIndex(), self.method.currentText()]}\n{self.imshow.isChecked()}\n"
                     f"{self.coord1[0][0]}, {self.coord1[0][1]}\n{self.coord1[1][0]}, {self.coord1[1][1]}\n"
                     f"{self.coord1[2][0]}, {self.coord1[2][1]}\n"
                     f"{self.coord2[0][0]}, {self.coord2[0][1]}\n{self.coord2[1][0]}, {self.coord2[1][1]}\n"
                     f"{self.coord2[2][0]}, {self.coord2[2][1]}")
        with open("Program state.txt", "w") as out_file:
            out_file.write(state_str)

    def load_program_state(self):
        def format(string):
            coord1 = string.split(",")[0]
            coord2 = string[len(coord1) + 2:]
            coord1 = coord1 if len(coord1) == 9 else coord1 + "0" * (9 - len(coord1))
            coord2 = coord2 if len(coord2) == 9 else coord2 + "0" * (9 - len(coord2))
            string = str(coord1) + ", " + str(coord2)
            return string

        try:
            with open("Program state.txt", "r") as out_file:
                states = out_file.readlines()

            for state in states:
                if len(state) < 2:
                    raise IndexError

            self.main_path.setText(states[0][:-1])
            self.height.setText(states[1][:-1])
            self.second_path.setText(states[2][:-1])
            self.height_crop.setText(states[3][:-1])
            self.dist.setText(states[4][:-1])
            self.height_diff.setCurrentText(states[5][:-1])
            self.step.setText(states[6][:-1])
            self.cycles.setCurrentText(states[7][:-1])
            self.method.setCurrentIndex(int(states[8][1]))
            self.method.setCurrentText(states[8][5:-1].split("'")[0])
            self.imshow.setChecked(True if states[9][:-1] == "True" else False)

            # Запись координат из файла
            self.main_coord_1.setText(format(states[10][:-1]))
            self.main_coord_2.setText(format(states[11][:-1]))
            self.main_coord_3.setText(format(states[12][:-1]))

            self.crop_coord_1.setText(format(states[13][:-1]))
            self.crop_coord_2.setText(format(states[14][:-1]))
            self.crop_coord_3.setText(format(states[15]))


        except FileNotFoundError:
            pass
        except IndexError:
            os.remove("Program state.txt")

    def entering_coordinates(self):
        nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

        def char_for_split(string):
            for char in string:
                if char not in nums:
                    return char

        def num_for_split(string):
            index = 0
            for char in string:
                if char not in nums:
                    index += 1
                else:
                    break
            return index

        def coordinates(string):
            if len(string) < 16:
                return 0, 0
            coord1 = string.split(char_for_split(string))[0]
            index = len(coord1)
            coord1 = coord1 if len(coord1) == 9 else coord1 + "0" * (9 - len(coord1))
            index += num_for_split(string.split(char_for_split(string))[1])
            coord2 = string[index:].strip()
            coord2 = coord2.split(char_for_split(coord2))[0]
            coord2 = coord2 if len(coord2) == 9 else coord2 + "0" * (9 - len(coord2))
            coord1 = float(coord1)
            coord2 = float(coord2)
            return coord1, coord2

        try:
            point_main1 = coordinates(self.main_coord_1.text())
            point_main2 = coordinates(self.main_coord_2.text())
            point_main3 = coordinates(self.main_coord_3.text())
            point_view1 = coordinates(self.crop_coord_1.text())
            point_view2 = coordinates(self.crop_coord_2.text())
            point_view3 = coordinates(self.crop_coord_3.text())

            fields_list = [point_main1[0], point_main1[1], point_main2[0], point_main2[1], point_main3[0],
                           point_main3[1], point_view1[0], point_view1[1], point_view2[0], point_view2[1],
                           point_view3[0], point_view3[1]]
            for field in fields_list:
                if len(str(field)) < 7:
                    return True

            self.main_coord_1.setText(str(point_main1[0]) + ", " + str(point_main1[1]))
            self.main_coord_2.setText(str(point_main2[0]) + ", " + str(point_main2[1]))
            self.main_coord_3.setText(str(point_main3[0]) + ", " + str(point_main3[1]))
            self.crop_coord_1.setText(str(point_view1[0]) + ", " + str(point_view1[1]))
            self.crop_coord_2.setText(str(point_view2[0]) + ", " + str(point_view2[1]))
            self.crop_coord_3.setText(str(point_view3[0]) + ", " + str(point_view3[1]))

            self.coord1 = [point_main1, point_main2, point_main3]
            self.coord2 = [point_view1, point_view2, point_view3]

            return False
        except ValueError:
            return True

    def checking_paths(self):
        if not os.path.exists(self.main_path.text()) or not os.path.exists(self.second_path.text()):
            return True

    def checking_empty_fields(self):
        if self.entering_coordinates():
            return True

        fields_list = [self.height.text(), self.height_crop.text(), self.dist.text(), self.step.text()]
        for field in fields_list:
            try:
                float(field)
            except ValueError:
                return True
        if float(self.height.text()) < 20 or float(self.height_crop.text()) < 20 or float(self.height.text()) < float(
                self.height_crop.text()):
            return True
        if float(self.dist.text()) < 0.01 or float(self.dist.text()) > 1 or float(self.step.text()) < 50:
            return True

        return False

    def show_message(self, txt):
        # Создаем диалоговое окно
        msg = QMessageBox()
        msg.setWindowTitle("Ошибка")
        msg.setText(txt)
        msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Program()
    ex.show()
    sys.exit(app.exec_())
