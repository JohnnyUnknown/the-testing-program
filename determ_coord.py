import math

class Determ_coord():
    point1, point2, point3, point4 = None, None, None, None
    point_pixel1, point_pixel2, point_pixel3, point_pixel4 = None, None, None, None
    target_point = [0, 0]
    target_pixel_point = [0, 0]
    angle = 0
    count_rotate = 0
    old_begin = [0, 0]

    def __init__(self, point1, point2, point3, image_size):
        point_pixel1 = [0, 0]
        point_pixel2 = [0, image_size[1]]
        point_pixel3 = [image_size[0], image_size[1]]
        point_pixel4 = [image_size[0], 0]

        point4 = self.find_p4(point1, point2, point3)
        self.point1, self.point2, self.point3, self.point4 = self.new_points(point1, point2, point3, point4)
        self.point_pixel1, self.point_pixel2, self.point_pixel3, self.point_pixel4 = (
            self.new_pixel_points(point_pixel1, point_pixel2, point_pixel3, point_pixel4))

        meters_dolong = round((6378137 * math.cos(float((self.point1[0] + self.point4[0]) / 2) *
                                                  math.pi / 180) * 2 * math.pi) / 360, 3)
        meters_dolat = 111100
        self.coef = meters_dolong / meters_dolat

        self.angle = self.find_angle()
        # print(math.degrees(self.angle))

    def calculate(self, pixel_center):
        self.target_pixel_point[0] = pixel_center[1]
        self.target_pixel_point[1] = pixel_center[0]
        if self.angle != 0:
            self.new_center()
            self.target_point[0] = self.find_latitude()
            self.target_point[1] = self.find_longitude()
        else:
            pixel_price_x = round((self.point2[1] - self.point1[1]) / self.point_pixel2[1], 10)
            pixel_price_y = round((self.point2[0] - self.point3[0]) / self.point_pixel3[0], 10)
            self.target_point[0] = round(pixel_center[0] * abs(pixel_price_y) + self.point1[0], 6)
            self.target_point[1] = round(pixel_center[1] * abs(pixel_price_x) + self.point1[1], 6)
        target = self.target_point.copy()
        return target

    def inversion(self, point_pixel1, point_pixel2, point_pixel3, point_pixel4):
        point_pixel1[0], point_pixel1[1] = point_pixel1[1], point_pixel1[0]
        point_pixel2[0], point_pixel2[1] = point_pixel2[1], point_pixel2[0]
        point_pixel3[0], point_pixel3[1] = point_pixel3[1], point_pixel3[0]
        point_pixel4[0], point_pixel4[1] = point_pixel4[1], point_pixel4[0]
        return point_pixel1, point_pixel2, point_pixel3, point_pixel4

    def find_p4(self, point1, point2, point3):
        x_point = round(point1[1] - (point2[1] - point3[1]), 6)
        y_point = round(point1[0] - (point2[0] - point3[0]), 6)
        return (y_point, x_point)

    def new_points(self, point1, point2, point3, point4):
        array = [point1, point2, point3, point4]
        array_out = []
        temp = min(array, key=lambda x: x[1])
        for i in range(4):
            if temp == array[i] and i != 0:
                self.count_rotate = i
                for j in range(4):
                    array_out.append(array[i])
                    i = i + 1 if i < 3 else 0
                break
        # print(array)
        # print(array_out)
        # print(self.count_rotate)
        return array_out

    def new_pixel_points(self, point1, point2, point3, point4):
        array = [point1, point2, point3, point4]
        array_out = []
        # print(array)
        # self.count_rotate = 3
        for i in range(self.count_rotate):
            k = 0
            array[0], array[1] = array[1], array[0]
            array[2], array[3] = array[3], array[2]
            array[0], array[1], array[2], array[3] = self.inversion(array[0], array[1], array[2], array[3])
            for j in range(4):
                k = (j + 1) if k < 3 else 0
                array_out.append(array[k])
            array = array_out.copy()
            array_out.clear()
        self.old_begin = array[4-self.count_rotate] if self.count_rotate > 0 else array[0]
        # print(array)
        # print(self.count_rotate, self.old_begin)
        return array

    def new_center(self):
        if self.count_rotate % 2 == 0:
            self.target_pixel_point[0] = abs(self.old_begin[0] - self.target_pixel_point[0])
            self.target_pixel_point[1] = abs(self.old_begin[1] - self.target_pixel_point[1])
        else:
            self.target_pixel_point[0] = abs(self.old_begin[1] - self.target_pixel_point[0])
            self.target_pixel_point[1] = abs(self.old_begin[0] - self.target_pixel_point[1])
            self.target_pixel_point[0], self.target_pixel_point[1] = self.target_pixel_point[1], self.target_pixel_point[0]
        # print(self.target_pixel_point)

    def find_angle(self):
        size1 = round((self.point1[1] - self.point4[1]) * self.coef, 6)
        size2 = round(self.point1[0] - self.point4[0], 6)
        hypotenuse = math.sqrt(size1 * size1 + size2 * size2)
        sin_angle = abs(size1 / hypotenuse)
        angle_rad = round(math.asin(sin_angle), 6)
        # angle = math.degrees(angle_rad)
        # print(round(angle_rad, 3))
        return round(angle_rad, 3)

    def find_latitude(self):
        size_1 = abs((self.point_pixel2[1] - self.target_pixel_point[1]) * math.tan(self.angle))
        size_2 = size_1 + self.target_pixel_point[0]
        size_3 = round((self.point_pixel3[0] - size_2), 2)
        height = size_3 * math.cos(self.angle)
        height_pixel_price = abs((self.point2[0] - self.point3[0]) / (self.point_pixel3[0] * math.cos(self.angle)))
        height_orig = height_pixel_price * height
        if size_2 < self.point_pixel3[0]:
            return round(self.point3[0] + height_orig, 6)
        if size_2 >= self.point_pixel3[0]:
            return round(self.point3[0] + height_orig, 6)

    def find_longitude(self):
        size_1 = abs(self.target_pixel_point[1] + self.target_pixel_point[0] * math.tan(self.angle))
        size_2 = round((self.point_pixel2[1] - size_1), 2)
        width = size_2 * math.cos(self.angle)
        width_pixel_price = abs((self.point2[1] - self.point1[1]) / (self.point_pixel2[1] * math.cos(self.angle)))
        width_orig = (self.point_pixel2[1] * math.cos(self.angle) - width) * width_pixel_price
        if size_1 < self.point_pixel2[1]:
            return round(self.point1[1] + width_orig, 6)
        elif size_1 >= self.point_pixel2[1]:
            return round(self.point1[1] + width_orig, 6)