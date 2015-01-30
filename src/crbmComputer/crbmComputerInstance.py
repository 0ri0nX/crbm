import crbmComputer as r

from remote import RemoteFile

REMOTESERVER = "orion.dev"

WEIGHTS = ["crbm/data5-5k-200x200x3.txt.rbm"
         , "crbm/data5-5k-200x200x3.txt.transformed.rbm"
         , "crbm/data5-5k-200x200x3.txt.transformed.transformed.rbm"
         , "crbm/data5-5k-200x200x3.txt.transformed.transformed.transformed.rbm"
         , "crbm/data5-5k-200x200x3.txt.transformed.transformed.transformed.transformed.rbm"
         , "crbm/data5-5k-200x200x3.txt.transformed.transformed.transformed.transformed.transformed.rbm"]



localWeights = [RemoteFile(i, REMOTESERVER).file_name for i in WEIGHTS]
GPUID = 0

RBMInstance = r.CRBMComputer(localWeights, GPUID)

def transform(val):
    global RBMInstance

    res = RBMInstance.transform(val)
    res = list(res)

    return res

def transformBatch(val):
    global RBMInstance

    res = RBMInstance.transformBatch(val)
    res = list(res)

    return res


