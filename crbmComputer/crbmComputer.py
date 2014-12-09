#!/usr/bin/env python

import ctypes
import pkg_resources

lib = ctypes.cdll.LoadLibrary(pkg_resources.resource_filename(__name__, 'libcrbmcomputer.so'))

lib_CRBMStack_new = lib.CRBMStack_new
lib_CRBMStack_delete = lib.CRBMStack_delete
lib_CRBMStack_Transform = lib.CRBMStack_Transform
lib_CRBMStack_GetOutputSize = lib.CRBMStack_GetOutputSize
lib_CRBMStack_GetOutputSize.restype = ctypes.c_int

class CRBMComputer(object):
    def __init__(self, crbmsFiles, gpu = 0):
        TStringList = ctypes.c_char_p*len(crbmsFiles)

        self.crbm = lib_CRBMStack_new(ctypes.c_int(len(crbmsFiles)), TStringList(*crbmsFiles), ctypes.c_int(gpu))
        self.outputNum = lib_CRBMStack_GetOutputSize(self.crbm)

    def __del__(self):
        lib_CRBMStack_delete(self.crbm)
        self.crbm = None

    def transform(self, inVector):
        TVIn = ctypes.c_float*len(inVector)
        TVOut = ctypes.c_float*self.outputNum
        #outVector = TVOut(*([0.]*10))
        outVector = TVOut()
        lib_CRBMStack_Transform(self.crbm, ctypes.c_int(len(inVector)), TVIn(*inVector), ctypes.c_int(self.outputNum), outVector)

        return outVector


