#ifndef CRBMCOMPUTERBINDING_H
#define CRBMCOMPUTERBINDING_H

#ifdef __cplusplus
extern "C"
{
#endif
    /**
     * Forward declaration of a stack of crbm layers
     *
     */
    struct CRBMStack;

    /**
     * Creates new crbm-stack.
     * @param inLength is number of input weight-filenames
     * @param inWeights are weight-filenames
     * @param inDeviceID is device we want to use (only appropriate for GPU version)
     * @return pointer to a new crbm-stack instance
     */
    CRBMStack* CRBMStack_new(int inLength, const char** inWeights, int inDeviceID);

    /**
     * Returns output size of crbm-stack
     * @param inCRBMStack is pointer to crbm-stack instance
     */
    int CRBMStack_GetOutputSize(CRBMStack* inCRBMStack);

    /**
     * Returns input size of crbm-stack
     * @param inCRBMStack is pointer to crbm-stack instance
     */
    int CRBMStack_GetInputSize(CRBMStack* inCRBMStack);

    /**
     * Transforms one image into metasignals
     * @param inCRBMStack is pointer to crbm-stack instance
     * @param inLenInData is the size of input data. It must be equal to the crbm-stack input-size.
     * @param inData are image data (rgb pixels divided by 255)
     * @param inLenOutData is the size of output data. It must be equal to the crbm-stack output-size.
     * @param outData is representation of the image computed by crbm-stack.
     */
    void CRBMStack_Transform(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData);

    /**
     * Transforms a batch of images into a batch of metasignals
     * @param inCRBMStack is pointer to crbm-stack instance
     * @param inLenInData is the size of input data. It must be equal to the crbm-stack input-size multiplied by batch-size.
     * @param inData are image data (rgb pixels divided by 255)
     * @param inLenOutData is the size of output data. It must be equal to the crbm-stack output-size multiplied by batch-size.
     * @param outData is representation of the images computed by crbm-stack.
     */
    void CRBMStack_TransformBatch(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData);

    /**
     * Reconstructs one image-metasignals into its image representation.
     * @param inCRBMStack is pointer to crbm-stack instance
     * @param inLenInData is the size of input data. It must be equal to the crbm-stack output-size.
     * @param inData are image metasignal data.
     * @param inLenOutData is the size of output data. It must be equal to the crbm-stack input-size.
     * @param outData is the image reconstructed from the metasignals (values must be multiplied by 255 to get rgb pixels).
     */
    void CRBMStack_Reconstruct(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData);

    /**
     * Reconstructs batch of image-metasignals into their image representation.
     * @param inCRBMStack is pointer to crbm-stack instance
     * @param inLenInData is the size of input data. It must be equal to the crbm-stack output-size multiplied by batch-size.
     * @param inData are image metasignal data.
     * @param inLenOutData is the size of output data. It must be equal to the crbm-stack input-size  multiplied by batch-size.
     * @param outData is the image-batch reconstructed from the metasignals (values must be multiplied by 255 to get rgb pixels).
     */
    void CRBMStack_ReconstructBatch(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData);

    /**
     * destroys crbm-stack instance
     * @param inCRBMStack is pointer to crbm-stack instance
     */
    void CRBMStack_delete(CRBMStack* inCRBMStack);
#ifdef __cplusplus
}
#endif

#endif // CRBMCOMPUTERBINDING_H

