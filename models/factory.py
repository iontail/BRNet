# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from .DSFD_BRNet import build_net_DSFD


def build_net(phase, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    return build_net_DSFD(phase, num_classes)


def basenet_factory():
	basenet = 'BRNet_base.pth'
	return basenet

