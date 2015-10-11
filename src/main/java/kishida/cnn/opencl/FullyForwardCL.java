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
import java.nio.IntBuffer;
import java.util.Map;
import kishida.cnn.activation.ActivationFunction;
import kishida.cnn.activation.SoftMaxFunction;

/**
 *
 * @author naoki
 */
public class FullyForwardCL {
    public static FullyForwardCL INSTANCE = new FullyForwardCL();
    CLProgram progFully;
    CLProgram progActivation;
    CLKernel forwardKernel;
    Map<String, CLKernel> actKernels;
    public FullyForwardCL() {
    }

    public void forward(int inputSize, int outputSize, int[] dropout,
            float[] input, float[] weight, float[] bias, float[] result,
            ActivationFunction activation){
        CLBuffer<FloatBuffer> bufWeight = OpenCL.createReadWriteBuffer(weight, bias);

        forward(inputSize, outputSize, dropout, input, bufWeight, result, activation);

        OpenCL.getQueue()
                .putWriteBuffer(bufWeight, false);
        bufWeight.release();

    }
    public void forward(int inputSize, int outputSize, int[] dropout,
           float[] input, CLBuffer<FloatBuffer> bufWeight,
           float[] result,
           ActivationFunction activation){
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadWriteBuffer(result.length);
        CLBuffer<IntBuffer> bufDropout = OpenCL.createReadBuffer(dropout);

        OpenCL.getQueue()
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufDropout, false);

        forward(inputSize, outputSize, bufDropout, bufInput, bufWeight, bufResult, activation);

        OpenCL.getQueue().putReadBuffer(bufResult, true);
        bufResult.getBuffer().get(result);

        bufInput.release();
        bufResult.release();
        bufDropout.release();

    }
    public void forward(int inputSize, int outputSize, CLBuffer<IntBuffer> bufDropout,
           CLBuffer<FloatBuffer> bufInput, CLBuffer<FloatBuffer> bufWeight,
           CLBuffer<FloatBuffer> bufResult,
           ActivationFunction activation){
        if(progFully == null){
            progFully = OpenCL.compile("fully_forward.cl");
            forwardKernel = progFully.createCLKernel("forward");
        }
        if(progActivation == null){
            progActivation = OpenCL.compile("activation.cl");
            actKernels = progActivation.createCLKernels();
        }

        forwardKernel.rewind()
                .putArg(inputSize + 1)
                .putArg(outputSize)
                .putArgs(
                        bufDropout,
                        bufInput,
                        bufWeight,
                        bufResult);
        OpenCL.execute(forwardKernel, outputSize);

        if(activation instanceof SoftMaxFunction){
            softmax(outputSize, bufResult);

        }else{
            CLKernel kernelAct = actKernels.get(activation.getName());
            kernelAct.rewind()
                    .putArg(bufResult);
            OpenCL.execute(kernelAct, outputSize);
        }

    }

    private void softmax(int outputSize, CLBuffer<FloatBuffer> bufResult) {
        CLBuffer<FloatBuffer> bufExped = OpenCL.createReadWriteBuffer(outputSize);
        CLKernel kernelActPre = actKernels.get("softmax_before");
        kernelActPre.rewind()
                .putArg(bufResult)
                .putArg(bufExped);
        OpenCL.execute(kernelActPre, outputSize);

        CLKernel kernelAct = actKernels.get("softmax");
        kernelAct.rewind()
                .putArg(bufExped)
                .putArg(bufResult);
        OpenCL.execute(kernelAct, outputSize);

        bufExped.release();
    }
}
