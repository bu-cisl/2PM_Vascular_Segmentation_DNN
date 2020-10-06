import getopt
import json
import os
import sys
from typing import List, Union

import loss_function as lf
import tensorflow as tf
from VNetOriginal3 import v_net as vnet3
from const import RESULT_DIR


def _error():
    print("main.py: missing or invalid options\n"
          "Try 'python main.py [-h --help]' for help or 'python main.py [-d --default] to\n"
          "run as default'")
    sys.exit(1)


def _help():
    print("usage: python main.py [OPTION] [PARAMETER]\n"
          "OPTIONS:\n"
          "(if not set or run as default, PARAMETER is set to the first one)\n"
          "  -h, --help                   display this help and exit\n"
          "  -d, --default                use default settings\n"
          "  -l, --loss LOSS_TYPE         set loss function, including\n"
          "                                 ce          Simple cross entropy\n"
          "                                 bce         Balanced cross entropy\n"
          "                                 bce_fpc     Balanced cross entropy with false\n"
          "                                               positive corrected)\n"
          "                                 dice        Dice loss according to V-Net paper\n"
          "                                 uns         Dice with TV"
          "                                 tv_bce      Blanced cross entropy with TV"
          "  -n, --net NETWORK            set neural network type, including\n"
          "                                 vnet, vnet5 Original V-Net\n"
          "                                 unet        3D U-Net\n"
          "                                 vnet3       Change V-Net filter to 3x3x3\n"
          "  -i, --img IMAGE_NUMBER       set how many images in a mouse, default '1'\n"
          "  -p, --prefix NAME_PREFIX     set name prefix of data file, must be 2\n"
          "                                 characters for raw img and ground truth,\n"
          "                                 for example 'og'\n"
          "  -b, --batch BATCH_SIZE       set batch size, default 2\n"
          "  -o, --opt OPTIMIZER_TYPE     set optimizer type, including\n"
          "                                 adam        Adam optimizer\n"
          "  -r, --rate LEARNING_RATE     set learning rate for optimizer")
    sys.exit(0)


class Settings:
    def __init__(self):
        self._config_dict = {}
        try:
            opts, unexpected_args = getopt.getopt(
                sys.argv[1:],
                shortopts="hdl:n:i:p:b:o:r:",
                longopts=["help", "default", "loss=", "net=", "img=", "prefix=",
                          "batch=", "opt=", "rate="]
            )
            assert opts  # at least one opt, or raise error
            assert not unexpected_args
            for opt, arg in opts:
                if opt in ['-h', '--help']:
                    _help()
                elif opt in ['-d', '--default']:
                    break
                elif opt in ['-l', '--loss']:
                    self._set_loss(arg)
                elif opt in ['-n', '--net']:
                    self._set_net(arg)
                elif opt in ['-i', '--img']:
                    self._set_img_num(arg)
                elif opt in ['-p', '--prefix']:
                    self._set_prefix(arg)
                elif opt in ['-b', '--batch']:
                    self._set_batch(arg)
                elif opt in ['-o', '--opt']:
                    self._set_opt(arg)
                elif opt in ['-r', '--rate']:
                    self._set_rate(arg)
                else:
                    print("Check the option '" + opt + "' in setting.py!")
                    sys.exit(-1)

            # if not set, set to default value
            try:
                self._config_dict['loss_function']
            except KeyError:
                self._set_loss('tv_bce')
            try:
                self._config_dict['network']
            except KeyError:
                self._set_net('vnet3')
            try:
                self._config_dict['image_filename']['range']
            except KeyError:
                self._set_img_num(1)
            try:
                self._config_dict['image_filename']['prefixes']
            except KeyError:
                self._set_prefix('fg')
            try:
                self._config_dict['batch_size']
            except KeyError:
                self._set_batch(4)
            try:
                self._config_dict['train_options']['optimizer']
            except KeyError:
                self._set_opt("adam")
            try:
                self._config_dict['train_options']['learning_rate']
            except KeyError:
                self._set_rate(1e-4)

        except (getopt.GetoptError, AssertionError, ValueError):
            print("Hello")
            _error()
        try:
            os.makedirs(RESULT_DIR)
        except FileExistsError:
            pass
        with open(RESULT_DIR + "config.json", "w") as config_file:
            json.dump(self._config_dict, config_file, indent=4)

    def _set_loss(self, loss: str):
        assert loss in ['dice', 'ce', 'bce', 'bce_fpc', 'uns', 'tv_bce']
        self._config_dict['loss_function'] = loss

    def _set_net(self, net: str):
        assert net in ['vnet', 'vnet5', 'unet', 'vnet3']
        self._config_dict['network'] = net

    def _set_img_num(self, img_n: Union[str, int]):
        try:
            self._config_dict['image_filename']
        except KeyError:
            self._config_dict['image_filename'] = {}
        self._config_dict['image_filename']['range'] = int(img_n)  # ValueError is expected

    def _set_prefix(self, pre: str):
        assert len(pre) == 2
        try:
            self._config_dict['image_filename']
        except KeyError:
            self._config_dict['image_filename'] = {}
        self._config_dict['image_filename']['prefixes'] = {'data': pre[0], 'label': pre[1]}

    def _set_batch(self, bat: Union[str, int]):
        self._config_dict['batch_size'] = int(bat)
        # raise NotImplementedError

    def _set_opt(self, opt: str):
        assert opt in ['adam']
        try:
            self._config_dict['train_options']
        except KeyError:
            self._config_dict['train_options'] = {}
        self._config_dict['train_options']['optimizer'] = opt
        # raise NotImplementedError

    def _set_rate(self, rate: Union[str, float]):
        self._rate = float(rate)
        try:
            self._config_dict['train_options']
        except KeyError:
            self._config_dict['train_options'] = {}
        self._config_dict['train_options']['learning_rate'] = float(rate)
        # raise NotImplementedError

    def get_loss(self) -> lf.LossFunction:
        loss = self._config_dict['loss_function']
        if loss == 'dice':
            return lf.sigmoid_dice
        elif loss == 'ce':
            return lf.sigmoid_cross_entropy
        elif loss == 'bce':
            return lf.sigmoid_cross_entropy_balanced
        elif loss == 'uns':
            return lf.derivative_mean
        elif loss == 'tv_bce':
            return lf.total_variation_balanced_cross_entropy
        else:
            raise NotImplementedError

    def get_net(self) -> callable:
        net = self._config_dict['network']
        if net in ['vnet', 'vnet5']:
            return vnet
        elif net == 'vnet3':
            return vnet3
        elif net == 'unet':
            return unet
        else:
            raise NotImplementedError

    def get_image_number(self) -> int:
        return self._config_dict['image_filename']['range']

    def get_prefix(self) -> List[str]:
        pre = self._config_dict['image_filename']['prefixes']
        return [pre['data'], pre['label']]

    def get_batch(self) -> int:
        return self._config_dict['batch_size']

    def get_optimizer(self) -> tf.train.Optimizer:
        tr_o = self._config_dict['train_options']
        opt, rate = tr_o['optimizer'], tr_o['learning_rate']
        if opt == 'adam':
            return tf.train.AdamOptimizer(rate)
        else:
            raise NotImplementedError
