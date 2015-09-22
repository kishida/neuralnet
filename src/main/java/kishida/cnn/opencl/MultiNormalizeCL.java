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
public class MultiNormalizeCL {
    public static MultiNormalizeCL INSTANCE = new MultiNormalizeCL();

    CLProgram prog;
    Map<String, CLKernel> kernels;

    public void normalize(int inputChannels, int inputWidth, int inputHeight,
            int size, float threshold,
            float[] input, float[] result){
        if(prog == null){
            prog = OpenCL.compile("multi_normalize.cl");
            kernels = prog.createCLKernels();
        }

        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufAverages = OpenCL.createReadWriteBuffer(inputWidth * inputHeight);
        CLBuffer<FloatBuffer> bufStds = OpenCL.createReadWriteBuffer(inputWidth * inputHeight);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createWriteBuffer(result.length);

        OpenCL.getQueue().putWriteBuffer(bufInput, false);
        CLKernel kernelAverage = kernels.get("average");
        kernelAverage.rewind()
                .putArg(inputChannels)
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArg(size)
                .putArg(threshold)
                .putArgs(
                        bufInput,
                        bufAverages,
                        bufStds);
        OpenCL.execute(kernelAverage, inputWidth * inputHeight);

        CLKernel kernelForward = kernels.get("forward");
        kernelForward.rewind()
                .putArg(inputChannels)
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArgs(
                        bufInput,
                        bufAverages,
                        bufStds,
                        bufResult);
        OpenCL.execute(kernelForward, inputChannels * inputWidth * inputHeight);

        OpenCL.getQueue().putReadBuffer(bufResult, true);

        bufResult.getBuffer().get(result);

        bufInput.release();
        bufAverages.release();
        bufStds.release();
        bufResult.release();

    }

    public static void main(String[] args) {
        int inputChannels = 3;
        int inputWidth = 32;
        int inputHeight = 32;
        int size = 5;
        float threshold = 0.00001f;
        float[] input = new float[inputChannels * inputWidth * inputHeight];
        float[] result = new float[inputChannels * inputWidth * inputHeight];
        new MultiNormalizeCL().normalize(inputChannels, inputWidth, inputHeight, size, threshold, input, result);
    }
}
