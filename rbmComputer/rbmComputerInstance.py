import rbmComputer as r

from remote import RemoteFile

REMOTESERVER = "orion.dev"

WEIGHTS = ['weights/vectorsRBMGeorge.2473.2000.weights'
         , 'weights/vectorsRBMGeorge.2473.2000.1000.weights'
         , 'weights/vectorsRBMGeorge.2473.2000.1000.500.weights'
         , 'weights/vectorsRBMGeorge.2473.2000.1000.500.200.weights'
         , 'weights/vectorsRBMGeorge.2473.2000.1000.500.200.100.weights'
         , 'weights/vectorsRBMGeorge.2473.2000.1000.500.200.100.50.weights']


localWeights = [RemoteFile(i, REMOTESERVER).file_name for i in WEIGHTS]

RBMInstance = r.RBMComputer(localWeights, 0) 

def transform(val):
    global RBMInstance

    res = RBMInstance.transform(val)
    res = list(res)

    return res

