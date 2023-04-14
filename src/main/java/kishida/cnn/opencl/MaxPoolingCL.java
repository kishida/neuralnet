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
public class MaxPoolingCL {
    public static MaxPoolingCL INSTANCE = new MaxPoolingCL();

    CLProgram prog;
    Map<String, CLKernel> kernels;

    private MaxPoolingCL() {
    }

    public void forward(int inputChannel, int inputWidth, int inputHeight, int outputWidth, int ouptutHeight,
            int size, int stride, float[] input, float[] result){
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createWriteBuffer(result.length);
        OpenCL.getQueue()
                .putWriteBuffer(bufInput, false);

        forward(inputChannel, inputWidth, inputHeight,
                outputWidth, ouptutHeight, size, stride, bufInput, bufResult);

        OpenCL.getQueue().putReadBuffer(bufResult, true);
        bufResult.getBuffer().get(result);

        bufInput.release();
        bufResult.release();

    }
    public void forward(int inputChannel, int inputWidth, int inputHeight, int outputWidth, int ouptutHeight,
            int size, int stride, CLBuffer<FloatBuffer> bufInput, CLBuffer<FloatBuffer> bufResult){

        if(prog == null){
            prog = OpenCL.compile("maxpooling.cl");
            kernels = prog.createCLKernels();
        }

        CLKernel kernelForward = kernels.get("forward");
        kernelForward.rewind()
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArg(outputWidth)
                .putArg(ouptutHeight)
                .putArg(size)
                .putArg(stride)
                .putArgs(
                    bufInput,
                    bufResult);
        OpenCL.execute(kernelForward,
                inputChannel * outputWidth * ouptutHeight);
    }

    public void backword(int inputChannel, int inputWidth, int inputHeight,
            int outputWidth, int outputHeight,
            int size, int stride,
            float[] input, float[] delta, float[] newDelta){
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufDelta = OpenCL.createReadBuffer(delta);
        CLBuffer<FloatBuffer> bufNewDelta = OpenCL.createReadWriteBuffer(newDelta);
        OpenCL.getQueue()
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufDelta, false);

        backword(inputChannel, inputWidth, inputHeight,
                outputWidth, outputHeight, size, stride,
                bufInput, bufDelta, bufNewDelta);
        
        OpenCL.getQueue().putReadBuffer(bufNewDelta, true);
        bufNewDelta.getBuffer().get(newDelta);

        bufInput.release();
        bufDelta.release();
        bufNewDelta.release();
    }
    public void backword(int inputChannel, int inputWidth, int inputHeight,
            int outputWidth, int outputHeight,
            int size, int stride,
            CLBuffer<FloatBuffer> bufInput,
            CLBuffer<FloatBuffer> bufDelta, CLBuffer<FloatBuffer> bufNewDelta){
        if(prog == null){
            prog = OpenCL.compile("maxpooling.cl");
            kernels = prog.createCLKernels();
        }

        CLKernel kernelForward = kernels.get("backword");
        kernelForward.rewind()
                .putArg(inputWidth)
                .putArg(inputHeight)
                .putArg(outputWidth)
                .putArg(outputHeight)
                .putArg(size)
                .putArg(stride)
                .putArgs(
                    bufInput,
                    bufDelta,
                    bufNewDelta);
        OpenCL.execute(kernelForward,
                inputChannel * inputWidth * inputHeight);

    }

    public static void main(String[] args) {
        int inputChannel = 3;
        int inputWidth = 32;
        int inputHeight = 32;
        int size = 11;
        int stride = 2;
        int outputWidth = inputWidth / stride;
        int outputHeight = inputHeight / stride;
        float[] input = new float[inputChannel * inputWidth * inputHeight];
        float[] result = new float[inputChannel * outputWidth * outputHeight];
        float[] newDelta = new float[input.length];
        float[] delta = new float[result.length];

        INSTANCE.forward(inputChannel, inputWidth, inputHeight, outputWidth, outputHeight, size, stride,
                input,result);
        INSTANCE.backword(inputChannel, inputWidth, inputHeight, outputWidth, outputHeight, size, stride,
                input, delta, newDelta);
    }
}
