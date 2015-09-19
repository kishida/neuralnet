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
        OpenCL.getQueue().putBarrier()
                .putWriteBuffer(bufDelta, false)
                .putWriteBuffer(bufFilter, false)
                .putWriteBuffer(bufResult, false)
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufFilterDelta, false)
                .putWriteBuffer(bufTempBias, false)
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
        OpenCL.execute(deltaKernel,
                inputChannels * inputWidth * inputHeight);
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
                        bufFilterDelta);
        OpenCL.execute(filterKernel,
                outputChannels * inputChannels * filterSize * filterSize);
        filterKernel.release();

        CLKernel biasKernel = prog.createCLKernel("bias");
        biasKernel
                .putArgs(
                        bufResult,
                        bufDelta,
                        bufTempBias)
                .putArg(learningRate);
        OpenCL.execute(biasKernel,
                outputChannels * outputWidth * outputHeight);
        biasKernel.release();

        /*
        OpenCL.getQueue().putReadBuffer(bufTempBias, true);
        float[] tempBias = new float[result.length];
        bufTempBias.getBuffer().get(tempBias).rewind();

        float[] compTempBias = new float[tempBias.length];
        for(int i = 0; i < compTempBias.length; ++i){
            compTempBias[i] = result[i] >= 0 ? delta[i] * learningRate : 0;
        }*/

        CLKernel biasAfterKernel = prog.createCLKernel("biasAfter");
        biasAfterKernel
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArgs(
                        bufTempBias,
                        bufBiasDelta);
        OpenCL.execute(biasAfterKernel, outputChannels);
        biasAfterKernel.release();
        OpenCL.getQueue()
                .putReadBuffer(bufBiasDelta, true)
                .putReadBuffer(bufFilterDelta, true)
                .putReadBuffer(bufNewDelta, true);
        bufNewDelta.getBuffer().get(newDelta);
        bufFilterDelta.getBuffer().get(filterDelta);
        bufBiasDelta.getBuffer().get(biasDelta);

        bufDelta.release();
        bufFilter.release();
        bufResult.release();
        bufInput.release();
        bufFilterDelta.release();
        bufTempBias.release();
        bufBiasDelta.release();
        bufNewDelta.release();

        return newDelta;
    }

    public static void main(String[] args) {
        int inputChannels = 3;
        int inputWidth = 200;
        int inputHeight = 200;
        int stride = 3;
        int filterSize = 11;
        int outputChannels = 24;
        int outputWidth = inputWidth / stride;
        int outputHeight = inputHeight / stride;
        float[] input = new float[inputChannels * inputWidth * inputHeight];
        float[] newDelta = new float[input.length];
        float[] filter = new float[inputChannels * outputChannels * filterSize * filterSize];
        float[] filterDelta = new float[filter.length];
        float[] biasDelta = new float[outputChannels];
        float[] result = new float[outputChannels * outputWidth * outputHeight];
        float[] delta = new float[result.length];
        float learningRate = 0.001f;

        for(int i = 0; i < 3; ++i){
            System.out.println(i + 1);
        ConvolutionBackwordCL.INSTANCE.backward(delta, result,
                input, inputChannels, inputWidth, inputHeight,
                filter, outputChannels, outputWidth, outputHeight, filterDelta, biasDelta, filterSize, stride, newDelta, learningRate);
        }
    }
}
