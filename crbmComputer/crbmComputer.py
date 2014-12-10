import ctypes
import pkg_resources
lib = ctypes.cdll.LoadLibrary(pkg_resources.resource_filename(__name__, 'libcrbmcomputer.so'))
lib_CRBMStack_new = lib.CRBMStack_new
lib_CRBMStack_delete = lib.CRBMStack_delete
lib_CRBMStack_Transform = lib.CRBMStack_Transform
lib_CRBMStack_TransformBatch = lib.CRBMStack_TransformBatch
lib_CRBMStack_GetOutputSize = lib.CRBMStack_GetOutputSize
lib_CRBMStack_GetOutputSize.restype = ctypes.c_int
lib_CRBMStack_GetInputSize = lib.CRBMStack_GetInputSize
lib_CRBMStack_GetInputSize.restype = ctypes.c_int

class CRBMComputer(object):

    def __init__(self, crbmsFiles, gpu = 0):
        TStringList = ctypes.c_char_p * len(crbmsFiles)
        self.crbm = lib_CRBMStack_new(ctypes.c_int(len(crbmsFiles)), TStringList(*crbmsFiles), ctypes.c_int(gpu))
        self.outputNum = lib_CRBMStack_GetOutputSize(self.crbm)
        self.inputNum = lib_CRBMStack_GetInputSize(self.crbm)

    def __del__(self):
        lib_CRBMStack_delete(self.crbm)
        self.crbm = None

    def transform(self, inVector):
        assert len(inVector) == self.inputNum
        TVIn = ctypes.c_float * self.inputNum
        TVOut = ctypes.c_float * self.outputNum
        outVector = TVOut()
        lib_CRBMStack_Transform(self.crbm, ctypes.c_int(self.inputNum), TVIn(*inVector), ctypes.c_int(self.outputNum), outVector)
        return outVector

    def transformBatch(self, inVector):
        assert len(inVector) % self.inputNum == 0
        batchNum = len(inVector) / self.inputNum

        TVIn = ctypes.c_float * (batchNum * self.inputNum)
        TVOut = ctypes.c_float * (batchNum * self.outputNum)

        outVector = TVOut()

        lib_CRBMStack_TransformBatch(self.crbm, ctypes.c_int(batchNum * self.inputNum), TVIn(*inVector), ctypes.c_int(batchNum * self.outputNum), outVector)
        return outVector

    def reconstruct(self, inVector):
        assert len(inVector) == self.outputNum
        TVIn = ctypes.c_float * self.outputNum
        TVOut = ctypes.c_float * self.inputNum
        outVector = TVOut()
        lib_CRBMStack_Reconstruct(self.crbm, ctypes.c_int(self.outputNum), TVIn(*inVector), ctypes.c_int(self.inputNum), outVector)
        return outVector
