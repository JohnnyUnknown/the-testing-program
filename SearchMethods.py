import cv2 as cv
import torch
import SuperPointFunc as SPF
import demo_superpoint as DSP
import Affine_transform as Affine


class Method():
    search_model = None
    matcher = None
    method_index = 1
    dist_kf = 0.5

    def __init__(self, method_index, dist_kf):
        self.method_index = method_index
        self.set_method()
        self.dist_kf = dist_kf

    # Выбор метода поиска КТ
    def set_method(self):
        match self.method_index:
            case 1:
                self.search_model = cv.SIFT_create()
            case 2:
                self.search_model = cv.AKAZE_create()
            case 3:
                self.search_model = cv.ORB_create(nfeatures=60000)
            case 4:  # ASIFT
                self.search_model = cv.SIFT_create()
            case 5:
                self.search_model = DSP.SuperPointNet()
                self.search_model.load_state_dict(torch.load('C:\\My\\Projects\\SuperPoint\\superpoint_v1.pth',
                                                             weights_only=True))
            case _:
                self.search_model = cv.SIFT_create()


    # Поиск КТ изображения
    def get_kp_and_des(self, img):
        if self.method_index == 4:
            kp, des = Affine.asift_detectAndCompute(img, self.search_model)
        elif self.method_index == 5:
            kp, des = SPF.get_keypoints_and_descriptors(img, self.search_model)
        else:
            kp, des = self.search_model.detectAndCompute(img, None)
        return kp, des

    def set_distance(self, new_dist):
        self.dist_kf = new_dist

    # Инициализация метода поиска совпадений
    def set_matcher(self):
        if self.method_index == 3:
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        elif self.method_index == 5:
            # self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

            # Используем FlannBasedMatcher для сопоставления дескрипторов
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv.BFMatcher()

    # Поиск общих КТ двух изображений
    def find_and_get_matches(self, des1, des2):
        self.set_matcher()
        if self.method_index == 3:
            matches = self.matcher.match(des1, des2)
            good = [m for m in matches if m.distance < self.dist_kf * matches[-1].distance]
        elif self.method_index == 5:
            # При использовании Flann
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.dist_kf * n.distance]
        else:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.dist_kf * n.distance]

        return good if len(good) >= 3 else None



    # bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=3)  # or pass empty dictionary
    # flann = cv.FlannBasedMatcher(index_params, search_params)
