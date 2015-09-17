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
public class ConvolutionBackwordCL {
    public static ConvolutionBackwordCL INSTANCE = new ConvolutionBackwordCL();
    CLProgram prog;

    private ConvolutionBackwordCL() {
    }

    public float[] backward(float[] delta, float[] result,
            float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight,
            float[] filterDelta, float[] biasDelta,
            int filterSize, int stride, float[] newDelta, float learningRate) {
        if(prog == null){
            prog = OpenCL.compile("convolution_backword.cl");
        }

        CLBuffer<FloatBuffer> bufDelta = OpenCL.createReadBuffer(delta);
        CLBuffer<FloatBuffer> bufFilter = OpenCL.createReadBuffer(filter);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadBuffer(result);
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufFilterDelta = OpenCL.createReadWriteBuffer(filterDelta);
        CLBuffer<FloatBuffer> bufTempBias = OpenCL.createReadWriteBuffer(result.length);
        CLBuffer<FloatBuffer> bufBiasDelta = OpenCL.createReadWriteBuffer(biasDelta);
        CLBuffer<FloatBuffer> bufNewDelta = OpenCL.createWriteBuffer(newDelta.length);

        OpenCL.getQueue()
                .putWriteBuffer(bufDelta, false)
                .putWriteBuffer(bufFilter, false)
                .putWriteBuffer(bufResult, false)
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufFilterDelta, false)
                .putWriteBuffer(bufBiasDelta, false);

        CLKernel deltaKernel = prog.createCLKernel("delta");
        deltaKernel
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArg(filterSize)
                .putArg(outputChannels)
                .putArg(stride)
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArgs(
                        bufResult,
                        bufDelta,
                        bufFilter)
                .putArg(inputChannels)
                .putArg(bufNewDelta);
        OpenCL.getQueue()
                .put1DRangeKernel(deltaKernel, 0,
                        inputChannels * inputWidth * inputHeight, 256);
        deltaKernel.release();

        CLKernel filterKernel = prog.createCLKernel("filter");
        filterKernel
                .putArg(inputChannels)
                .putArg(filterSize)
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArgs(
                        bufResult,
                        bufDelta)
                .putArg(stride)
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArg(learningRate)
                .putArgs(
                        bufInput,
                        bufFilter);
        OpenCL.getQueue()
                .put1DRangeKernel(filterKernel, 0,
                        outputChannels * inputChannels * filterSize * filterSize, 128);
        filterKernel.release();

        CLKernel biasKernel = prog.createCLKernel("bias");
        biasKernel
                .putArgs(
                        bufResult,
                        bufDelta,
                        bufTempBias)
                .putArg(learningRate);
        OpenCL.getQueue()
                .put1DRangeKernel(biasKernel, 0,
                        outputChannels * outputWidth * outputHeight, 128);
        biasKernel.release();

        CLKernel biasAfterKernel = prog.createCLKernel("biasAfter");
        biasAfterKernel
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArgs(
                        bufTempBias,
                        bufBiasDelta);
        OpenCL.getQueue()
                .put1DRangeKernel(biasAfterKernel, 0, outputChannels, 16)
                .putReadBuffer(bufBiasDelta, false)
                .putReadBuffer(bufFilterDelta, false)
                .putReadBuffer(bufNewDelta, true);
        bufNewDelta.getBuffer().get(newDelta);
        biasAfterKernel.release();
        return newDelta;
    }
}
