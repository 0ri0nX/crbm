#!/usr/bin/env python

import ctypes
import pkg_resources




lib = ctypes.cdll.LoadLibrary(pkg_resources.resource_filename(__name__, 'librbmcomputer.so'))

lib_RBMStack_new = lib.RBMStack_new
lib_RBMStack_delete = lib.RBMStack_delete
lib_RBMStack_Transform = lib.RBMStack_Transform
lib_RBMStack_GetOutputSize = lib.RBMStack_GetOutputSize
lib_RBMStack_GetOutputSize.restype = ctypes.c_int

class RBMComputer(object):
    def __init__(self, weights, gpu = 0):
        TWeights = ctypes.c_char_p*len(weights)

        self.rbm = lib_RBMStack_new(ctypes.c_int(len(weights)), TWeights(*weights), ctypes.c_int(gpu))
        self.outputNum = lib_RBMStack_GetOutputSize(self.rbm)

    def __del__(self):
        lib_RBMStack_delete(self.rbm)
        self.rbm = None

    def transform(self, inVector):
        TVIn = ctypes.c_float*len(inVector)
        TVOut = ctypes.c_float*self.outputNum
        #outVector = TVOut(*([0.]*10))
        outVector = TVOut()
        lib_RBMStack_Transform(self.rbm, ctypes.c_int(len(inVector)), TVIn(*inVector), ctypes.c_int(self.outputNum), outVector)

        return outVector


