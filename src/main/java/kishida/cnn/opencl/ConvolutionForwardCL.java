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
    CLKernel forwardKernel;
    CLKernel normalizeKernel;

    private ConvolutionForwardCL() {
    }

    /**
     * バッファを外部にもたない
     */
    public void forward(float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight, float[] result,
            int filterSize, int stride, float[] bias){

        CLBuffer<FloatBuffer> bufFilter = OpenCL.createReadBuffer(filter);
        CLBuffer<FloatBuffer> bufBias = OpenCL.createReadBuffer(bias);

        OpenCL.getQueue()
                .putWriteBuffer(bufFilter, false)
                .putWriteBuffer(bufBias, false);

        forward(input, inputChannels, inputWidth, inputHeight,
                bufFilter, outputChannels, outputWidth, outputHeight, result,
                filterSize, stride, bufBias);

        bufBias.release();
        bufFilter.release();
    }

    /**
     * filterとbiasは外部管理
     */
    public void forward(float[] input,
        int inputChannels, int inputWidth, int inputHeight,
        CLBuffer<FloatBuffer> bufFilter, int outputChannels, int outputWidth, int outputHeight,
        float[] result,
        int filterSize, int stride, CLBuffer<FloatBuffer> bufBias){

        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadWriteBuffer(result);
        OpenCL.getQueue()
                .putWriteBuffer(bufInput, false);

        forward(bufInput, inputChannels, inputWidth, inputHeight,
                bufFilter, outputChannels, outputWidth, outputHeight, bufResult,
                filterSize, stride, bufBias);

        OpenCL.getQueue()
                .putReadBuffer(bufResult, true);
        bufResult.getBuffer().get(result);

        bufResult.release();
        bufInput.release();

    }
    public void forward(CLBuffer<FloatBuffer> bufInput,
            int inputChannels, int inputWidth, int inputHeight,
            CLBuffer<FloatBuffer> bufFilter, int outputChannels, int outputWidth, int outputHeight,
            CLBuffer<FloatBuffer> bufResult,
            int filterSize, int stride, CLBuffer<FloatBuffer> bufBias){
        if(prog == null){
            prog = OpenCL.compile("convolution_forward.cl");
            forwardKernel = prog.createCLKernel("forward");
        }

        forwardKernel
                .rewind()
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

        normalizeKernel = prog.createCLKernel("localNormalize");
        normalizeKernel
                .rewind()
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArg(outputChannels)
                .putArg(bufResult);
        OpenCL.execute(normalizeKernel,
                outputChannels * outputWidth * outputHeight);

    }

}
