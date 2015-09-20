/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.opencl;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import java.nio.FloatBuffer;

/**
 *
 * @author naoki
 */
public class ConvolutionForwardCL {
    public static ConvolutionForwardCL INSTANCE = new ConvolutionForwardCL();
    CLProgram prog;
    private ConvolutionForwardCL() {
    }

    public float[] forward(float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight, float[] result,
            int filterSize, int stride, float[] bias){
        if(prog == null){
            prog = OpenCL.compile("convolution_forward.cl");
        }

        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufFilter = OpenCL.createReadBuffer(filter);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadWriteBuffer(result);
        CLBuffer<FloatBuffer> bufBias = OpenCL.createReadBuffer(bias);

        OpenCL.getQueue()
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufFilter, false)
                .putWriteBuffer(bufBias, false);

        CLKernel forwardKernel = prog.createCLKernel("forward");
        forwardKernel
                .putArg(outputHeight)
                .putArg(outputWidth)
                .putArg(inputChannels)
                .putArg(filterSize)
                .putArg(stride)
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArgs(
                        bufInput,
                        bufFilter,
                        bufResult,
                        bufBias);
        OpenCL.execute(forwardKernel,
                outputChannels * outputWidth * outputHeight);
        forwardKernel.release();

        CLKernel normalizeKernel = prog.createCLKernel("localNormalize");
        normalizeKernel
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArg(outputChannels)
                .putArg(bufResult);
        OpenCL.execute(normalizeKernel,
                outputChannels * outputWidth * outputHeight);
        normalizeKernel.release();

        OpenCL.getQueue()
                .putReadBuffer(bufResult, true);
        bufResult.getBuffer().get(result);

        bufBias.release();
        bufResult.release();
        bufInput.release();
        bufFilter.release();

        return result;
    }

}
