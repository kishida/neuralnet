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
            forwardKernel = prog.createCLKernel("forward_local");
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
        /*
        OpenCL.execute(forwardKernel,
                outputChannels * outputWidth * outputHeight);
                */
        forwardKernel.putArg(outputChannels * outputWidth * outputHeight);
        OpenCL.getQueue().put1DRangeKernel(forwardKernel, 0,
                outputChannels * outputWidth * outputHeight, outputChannels);

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
    public static void main(String[] args) {
        CLProgram prog = OpenCL.compile("convolution_forward.cl");
        CLKernel forwardKernel = prog.createCLKernel("forward");

        int inputChannels = 384;
        int inputWidth = 14;
        int inputHeight = 14;
        int outputChannels = 384;
        int outputWidth = 14;
        int outputHeight = 14;
        int filterSize = 3;
        int stride = 1;
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadWriteBuffer(
                inputChannels * inputWidth * inputHeight);
        CLBuffer<FloatBuffer> bufFilter = OpenCL.createReadWriteBuffer(
                inputChannels * outputChannels * filterSize * filterSize);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadWriteBuffer(
                outputChannels * outputWidth * outputHeight);
        CLBuffer<FloatBuffer> bufBias = OpenCL.createReadWriteBuffer(
                outputChannels);
        long start = System.currentTimeMillis();
        for(int i = 0; i < 5000; ++i){
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
                            bufBias)
                    .putArg(outputChannels * outputWidth * outputHeight);
            int workSize = outputChannels;
            OpenCL.getQueue().put1DRangeKernel(forwardKernel,
                    0, outputChannels * outputWidth * outputHeight,
                    workSize);
        }
        OpenCL.getQueue().putBarrier();
        System.out.println((System.currentTimeMillis() - start) / 1000.);
        bufFilter.release();
        System.out.println((System.currentTimeMillis() - start) / 1000.);
        bufInput.release();
        bufResult.release();
        bufBias.release();
        System.out.println((System.currentTimeMillis() - start) / 1000.);

        forwardKernel.release();
        prog.release();

        OpenCL.getQueue().release();
        OpenCL.getCtx().release();
    }
    static  int roundUp(int groupSize, int globalSize){
        return ((globalSize + groupSize - 1) / groupSize) * groupSize;
    }
}
