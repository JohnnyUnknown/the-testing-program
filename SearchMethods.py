import cv2 as cv
import torch
import demo_superpoint as DSP
import Affine_transform as Affine
from superpoint_superglue_deployment import Matcher


class Method():
    search_model = None
    matcher = None

    def __init__(self, method_index=1, dist_kf=0.5, nf=60000):
        self.method_index = method_index
        self.nf = nf
        self.set_method()
        self.dist_kf = dist_kf

    # Выбор метода поиска КТ
    def set_method(self):
        match self.method_index:
            case 2:
                self.search_model = cv.AKAZE_create()
            case 3:
                self.search_model = cv.ORB_create(nfeatures=self.nf)
            case 5:
                self.search_model = DSP.SuperPointNet()
                self.search_model.load_state_dict(torch.load('C:\\My\\Projects\\SuperPoint\\superpoint_v1.pth',
                                                             weights_only=True))
            case _:
                self.search_model = cv.SIFT_create()  # nOctaveLayers=3, contrastThreshold=0.03, edgeThreshold=10


    # Поиск КТ изображения
    def get_kp_and_des(self, img):
        if self.method_index == 4:
            kp, des = Affine.asift_detectAndCompute(img, self.search_model)
        elif self.method_index == 5:
            return None, None
        else:
            kp, des = self.search_model.detectAndCompute(img, None)
        return kp, des

    def set_distance(self, new_dist):
        self.dist_kf = new_dist

    # Инициализация метода поиска совпадений
    def set_matcher(self):
        if self.method_index == 3:
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            # FLANN_INDEX_LSH = 6
            # index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
            # search_params = dict(checks=50)
            # self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        elif self.method_index == 4:
            # self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        elif self.method_index == 5:
            self.matcher = Matcher(
                {
                    "superpoint": {
                        "input_shape": (-1, -1),
                        "keypoint_threshold": 0.003,
                    },
                    "superglue": {
                        "match_threshold": self.dist_kf,
                    },
                    "use_gpu": False,
                }
            )
        else:
            self.matcher = cv.BFMatcher()

    # Поиск общих КТ двух изображений
    def find_and_get_matches(self, *, img1=None, img2=None,  des1=None, des2=None):
        self.set_matcher()
        keypoints1, keypoints2 = None, None
        if self.method_index == 3:
            matches = self.matcher.match(des1, des2)
            good = [m for m in matches if m.distance < self.dist_kf * matches[-1].distance]
        elif self.method_index == 4:
            # При использовании Flann
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.dist_kf * n.distance]
        elif self.method_index == 5:
            keypoints1, keypoints2, _, _, good = self.matcher.match(img1, img2)
        else:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.dist_kf * n.distance]

        return keypoints1, keypoints2, good if len(good) >= 3 else None
