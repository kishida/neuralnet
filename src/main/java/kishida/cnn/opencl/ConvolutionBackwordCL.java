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
import java.util.Map;

/**
 *
 * @author naoki
 */
public class ConvolutionBackwordCL {
    public static ConvolutionBackwordCL INSTANCE = new ConvolutionBackwordCL();
    CLProgram prog;
    Map<String, CLKernel> kernels;

    private ConvolutionBackwordCL() {
    }

    public void backward(float[] delta, float[] result,
            float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight,
            float[] filterDelta, float[] biasDelta,
            int filterSize, int stride, float[] newDelta, float learningRate) {
        CLBuffer<FloatBuffer> bufFilter = OpenCL.createReadBuffer(filter);
        CLBuffer<FloatBuffer> bufFilterDelta = OpenCL.createReadWriteBuffer(filterDelta);
        CLBuffer<FloatBuffer> bufBiasDelta = OpenCL.createReadWriteBuffer(biasDelta);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadBuffer(result);
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        OpenCL.getQueue()
                .putWriteBuffer(bufFilter, false)
                .putWriteBuffer(bufFilterDelta, false)
                .putWriteBuffer(bufBiasDelta, false)
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufResult, false);

        backward(delta, bufResult,
                bufInput, inputChannels, inputWidth, inputHeight,
                bufFilter, outputChannels, outputWidth, outputHeight,
                bufFilterDelta, bufBiasDelta,
                filterSize, stride, newDelta, learningRate);

        OpenCL.getQueue()
                .putReadBuffer(bufBiasDelta, true)
                .putReadBuffer(bufFilterDelta, true);
        bufFilterDelta.getBuffer().get(filterDelta);
        bufBiasDelta.getBuffer().get(biasDelta);

        bufFilter.release();
        bufFilterDelta.release();
        bufBiasDelta.release();
        bufInput.release();
        bufResult.release();
    }
    public void backward(float[] delta, CLBuffer<FloatBuffer> bufResult,
            CLBuffer<FloatBuffer> bufInput, int inputChannels, int inputWidth, int inputHeight,
            CLBuffer<FloatBuffer> bufFilter, int outputChannels, int outputWidth, int outputHeight,
            CLBuffer<FloatBuffer> bufFilterDelta, CLBuffer<FloatBuffer> bufBiasDelta,
            int filterSize, int stride, float[] newDelta, float learningRate) {
        CLBuffer<FloatBuffer> bufDelta = OpenCL.createReadBuffer(delta);
        CLBuffer<FloatBuffer> bufNewDelta = OpenCL.createWriteBuffer(newDelta.length);
        CLBuffer<FloatBuffer> bufTempBias = OpenCL.createReadWriteBuffer(outputChannels * outputWidth * outputHeight);
        OpenCL.getQueue()
                .putWriteBuffer(bufDelta, false);

        backward(bufDelta, bufResult,
                bufInput, inputChannels, inputWidth, inputHeight,
                bufFilter, outputChannels, outputWidth, outputHeight,
                bufFilterDelta, bufBiasDelta, bufTempBias,
                filterSize, stride, bufNewDelta, learningRate);

        OpenCL.getQueue()
                .putReadBuffer(bufNewDelta, true);
        bufNewDelta.getBuffer().get(newDelta);

        bufDelta.release();
        bufNewDelta.release();
        bufTempBias.release();
    }
    public void backward(CLBuffer<FloatBuffer> bufDelta, CLBuffer<FloatBuffer> bufResult,
            CLBuffer<FloatBuffer> bufInput, int inputChannels, int inputWidth, int inputHeight,
            CLBuffer<FloatBuffer> bufFilter, int outputChannels, int outputWidth, int outputHeight,
            CLBuffer<FloatBuffer> bufFilterDelta, CLBuffer<FloatBuffer> bufBiasDelta,
            CLBuffer<FloatBuffer> bufTempBias,
            int filterSize, int stride, CLBuffer<FloatBuffer> bufNewDelta, float learningRate) {
        if(prog == null){
            prog = OpenCL.compile("convolution_backword.cl");
            kernels = prog.createCLKernels();
        }

        CLKernel deltaKernel = prog.createCLKernel("delta");
        deltaKernel
                .rewind()
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

        CLKernel filterKernel = kernels.get("filter");
        filterKernel
                .rewind()
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

        CLKernel biasKernel = kernels.get("bias");
        biasKernel
                .rewind()
                .putArgs(
                        bufResult,
                        bufDelta,
                        bufTempBias)
                .putArg(learningRate);
        OpenCL.execute(biasKernel,
                outputChannels * outputWidth * outputHeight);

        CLKernel biasAfterKernel = kernels.get("biasAfter");
        biasAfterKernel
                .rewind()
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArgs(
                        bufTempBias,
                        bufBiasDelta);
        OpenCL.execute(biasAfterKernel, outputChannels);

        bufTempBias.release();

    }

    public void prepare(float momentam,
            int filterCount, int biasCount,
            CLBuffer<FloatBuffer> bufFilterDelta,
            CLBuffer<FloatBuffer> bufBiasDelta){

        CLKernel kernel = kernels.get("prepare");
        kernel.rewind()
                .putArg(momentam)
                .putArg(bufFilterDelta);
        OpenCL.execute(kernel, filterCount);
        kernel.rewind()
                .putArg(momentam)
                .putArg(bufBiasDelta);
        OpenCL.execute(kernel, biasCount);
    }

    public void join(float weightDecay, float learningRate,
            int filterCount, int biasCount, int count,
            CLBuffer<FloatBuffer> bufFilter, CLBuffer<FloatBuffer> bufFilterDelta,
            CLBuffer<FloatBuffer> bufBias, CLBuffer<FloatBuffer> bufBiasDelta){
        CLKernel kernelFilter = kernels.get("joinFilter");
        kernelFilter.rewind()
                .putArg(weightDecay)
                .putArg(learningRate)
                .putArg(count)
                .putArgs(
                    bufFilter,
                    bufFilterDelta);
        OpenCL.execute(kernelFilter, filterCount);

        CLKernel kernelBias = kernels.get("joinBias");
        kernelBias.rewind()
                .putArg(count)
                .putArgs(
                        bufBias,
                        bufBiasDelta);
        OpenCL.execute(kernelBias, biasCount);

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
