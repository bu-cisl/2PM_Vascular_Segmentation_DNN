import time
from typing import List, Iterable, Dict

import numpy as np

from const import PATCH_SIZE

import scipy.io as sio

# consts
AUGMENTATION_N = 1
PATCH_STRIDE_TEST = 128
PATCH_STRIDE_TRAIN = 64


class Mouse:
    """
    A list of mouse image data in numpy array
    """
    # consts
    PREPROCESS_THRESHOLD = 184
    # preprocess change rule table
    _CHANGE_TABLE = np.arange(256, dtype=float)
    _CHANGE_TABLE[PREPROCESS_THRESHOLD:256] = PREPROCESS_THRESHOLD
    _CHANGE_TABLE = ((_CHANGE_TABLE / PREPROCESS_THRESHOLD) ** 2 * 255).astype('uint8')

    # type hint
    _is_test = ...  # type: bool
    index_list = ...  # type: List[List[slice]]
    shape = ...  # type: tuple
    mousenum = ...

    def __init__(self, mouse_n: str, *, img_num: int, is_test: bool, preprocess: bool = False,
                 data_dir: str = './test_data/', prefix: List[str] = None):
        """
        Read the mouse image data (.tiff format) for test and train. One mouse object
        may include several images according to img_num.

        Example: if image names are 'o0.tiff, o1.tiff; g0.tiff, g1.tiff',
        then img_num=2, prefix=['o', 'g']

        :param mouse_i: Index of mouse (used to locate directory)
        :param img_num: Images in 1 mouse folder
        :param is_test: Whether it is for test (or not = for training)
        :param preprocess: Whether the original image is intensity projected
        :param data_dir: Actual directory for data
        :param prefix: Prefix of .tiff file
        """
        import cv2
        self.mousename = mouse_n
        self._is_test = is_test
        self._img_num = img_num
        self.mousenum = 0

        if prefix is None:
            prefix = ['o', 'g']

        # Read data
        self._o_images = []

        for aug_i in range(img_num):

            # read data
            mat_contents = sio.loadmat(mouse_n)
            temp_o = mat_contents['cube'] #arrangement is (x,y,z) in order


            self._set_or_check_shape(temp_o)
            self._o_images.append(temp_o)

        # init index list
        self.index_list = []
        nx, ny, nz = self.shape
        pnx, pny, pnz = PATCH_SIZE

        if self._is_test:
            ps = PATCH_STRIDE_TEST
            offset = 128
        else:
            ps = PATCH_STRIDE_TRAIN
            offset = 0

        for i in range(0, nx - ps + offset, ps):
            for j in range(0, ny - ps + offset, ps):
                for k in range(0, nz - ps + offset, ps):
                    idx = []
                    # nx
                    if i < nx - pnx:
                        idx.append(slice(i, i + 128))
                    else:
                        idx.append(slice(nx - 128, nx))
                    # ny
                    if j < ny - pny:
                        idx.append(slice(j, j + 128))
                    else:
                        idx.append(slice(ny - 128, ny))
                    # nz
                    if k < nz - pnz:
                        idx.append(slice(k, k + 128))
                    else:
                        idx.append(slice(nz - 128, nz))

                    self.index_list.append(idx)

    def _set_or_check_shape(self, img: np.ndarray):
        if not isinstance(self.shape, tuple):
            self.shape = img.shape
        elif self.shape != img.shape:
            raise ValueError("An image have different shape " + str(img.shape) +
                             " with shape of other images " + str(self.shape) + " !")

    def get_original_image(self, img_n: int) -> np.ndarray:
        return self._o_images[img_n]

    def get_ground_truth(self, img_n: int) -> np.ndarray:
        return self._g_images[img_n]

    def __len__(self):
        return self._img_num

    def __getitem__(self, item):
        return {'original_image': self.get_original_image(item),
                'ground_truth': self.get_ground_truth(item)}


class Patch:
    """
    An image patch. Including a raw image and a label.

    Default shape: 128x128x128
    """
    def __init__(self, mouse: Mouse, img_n: int, index: List[slice]):
        self._mouse = mouse
        self._index = index
        self._img_n = img_n

    def get_index(self):
        return self._index

    def get_mouse(self):
        return self._mouse

    def get_original_image(self):
        return self._mouse.get_original_image(self._img_n)[self._index]

    def get_ground_truth(self):
        return self._mouse.get_ground_truth(self._img_n)[self._index]


class Batch:
    """
    A batch of data in numpy.array. Axis order is 'NDHWC'
    """
    def __init__(self, patch_list: List[Patch]):
        """
        Stack at batch axis and then, add "channels" axis for original image and
        (Changed:stack 2 channels) for ground truth.
        Axis order "NDHWC": [batch, depth, height, width, channels]

        :param patch_list: list of patches in this batch
        """
        self._o = np.stack([patch.get_original_image() for patch in patch_list])[..., np.newaxis]
        self._p = patch_list

    def get_original_images_in_batch(self):
        return self._o

    def get_ground_truths_in_batch(self):
        return self._g

    def get_mouse_list(self):
        return [patch.get_mouse() for patch in self._p]

    def get_index_list(self):
        return [patch.get_index() for patch in self._p]


def batch_generator(mouse_list: List[Mouse], batch_size: int) -> Iterable[Batch]:
    """
    A generator of image patches. Get image from disk (.tiff files) and
    return small patches to input into TensorFlow
    (Written by Jiabei Zhu on Jul. 30 2018)

    :param mouse_list: data source of mouses
    :param batch_size: the number of patches in a batch
    :return: dictionary of {"originalImage":'NDHWC' data in numpy array,
        "groundTruth":(same format)}
    """

    # get random list
    yield_sequence = [
        Patch(i, j, k)
        for i in mouse_list for k in i.index_list
        for j in range(len(i))  # test only original now
    ]

    np.random.shuffle(yield_sequence)

    # do yield
    patch_list = []
    for patch in yield_sequence:  # type: Patch
        # img_i = Patch(patch_i['mouse'], )
        patch_list.append(patch)
        if len(patch_list) == batch_size:
            yield Batch(patch_list)
            patch_list = []

def save_3d_tiff(path_prefix: str, img_dict: Dict[str, np.ndarray]):
    from libtiff import TIFF
    if path_prefix[-1] != '/':
        path_prefix = path_prefix + '/'
    for name in img_dict:
        img = img_dict[name]
        if img.min() >= 0 and img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            raise ValueError('The img data is not in 0~1!')
        if len(img.shape) != 3:
            raise ValueError('Images should be 3D numpy.ndarray')
        img = img.transpose((2, 0, 1))
        if name[-5:] != '.tiff' or name[-4:] != '.tif':
            tmp = ''.join(name)
            name = tmp + '_seg.tiff'
        tif = TIFF.open(path_prefix + name, mode='w')
        for img2d in img:
            tif.write_image(img2d, compression="lzw")
        tif.close()
