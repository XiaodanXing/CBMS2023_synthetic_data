import argparse
import os
from cal_fidelity import cal_fidelity
from cal_avg import  cal_avg
import torch
from torchvision import transforms

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # data directories
        self.parser.add_argument('-i',"--input_dir", type=str,
                            help="The directory of synthetic data to be evaluated")
        self.parser.add_argument('-r',"--ref_dir", type=str,
                                 default='none',
                            help="The directory of real data to be compared")
        self.parser.add_argument('-o',"--output_dir", type=str,

                            help="The directory for outputs if there are any")



        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(self.opt.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        print('The calculation of FID,precision and recall requires imageNet features. '
              'Thus the feature extraction could take a while.')
        return self.opt



if __name__ == '__main__':
    opt = BaseOptions().parse()
    cal_fidelity(opt)
    cal_avg(opt)
