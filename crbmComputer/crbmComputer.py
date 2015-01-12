import ctypes
import pkg_resources

#lib = ctypes.cdll.LoadLibrary(pkg_resources.resource_filename(__name__, 'libcrbmcomputercpu.so'))
lib = ctypes.cdll.LoadLibrary('libcrbmcomputercpu.so')

lib_CRBMStack_new = lib.CRBMStack_new
lib.CRBMStack_new.restype = ctypes.c_void_p

lib_CRBMStack_delete = lib.CRBMStack_delete
lib_CRBMStack_delete.argtypes = [ctypes.c_void_p]

lib_CRBMStack_Transform = lib.CRBMStack_Transform
lib_CRBMStack_Transform.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

lib_CRBMStack_TransformBatch = lib.CRBMStack_TransformBatch
lib_CRBMStack_TransformBatch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

lib_CRBMStack_Reconstruct = lib.CRBMStack_Reconstruct
lib_CRBMStack_Reconstruct.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

lib_CRBMStack_ReconstructBatch = lib.CRBMStack_ReconstructBatch
lib_CRBMStack_ReconstructBatch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

lib_CRBMStack_GetOutputSize = lib.CRBMStack_GetOutputSize
lib_CRBMStack_GetOutputSize.restype = ctypes.c_int
lib_CRBMStack_GetOutputSize.argtypes = [ctypes.c_void_p]

lib_CRBMStack_GetInputSize = lib.CRBMStack_GetInputSize
lib_CRBMStack_GetInputSize.restype = ctypes.c_int
lib_CRBMStack_GetInputSize.argtypes = [ctypes.c_void_p]

class CRBMComputer(object):

    def __init__(self, crbmsFiles, device = 0):
        TStringList = ctypes.c_char_p * len(crbmsFiles)
        self.crbm = lib_CRBMStack_new(ctypes.c_int(len(crbmsFiles)), TStringList(*crbmsFiles), ctypes.c_int(device))
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

    def reconstructBatch(self, inVector):
        assert len(inVector) % self.outputNum == 0
        batchNum = len(inVector) / self.outputNum

        TVIn = ctypes.c_float * (batchNum * self.outputNum)
        TVOut = ctypes.c_float * (batchNum * self.inputNum)

        outVector = TVOut()

        lib_CRBMStack_ReconstructBatch(self.crbm, ctypes.c_int(self.outputNum * batchNum), TVIn(*inVector), ctypes.c_int(self.inputNum * batchNum), outVector)
        return outVector
