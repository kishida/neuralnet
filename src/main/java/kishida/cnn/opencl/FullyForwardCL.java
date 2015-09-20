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
        if(progFully == null){
            progFully = OpenCL.compile("fully_forward.cl");
            forwardKernel = progFully.createCLKernel("forward");
        }
        if(progActivation == null){
            progActivation = OpenCL.compile("activation.cl");
            actKernels = progActivation.createCLKernels();
        }

        CLBuffer<IntBuffer> bufDropout = OpenCL.createReadBuffer(dropout);
        CLBuffer<FloatBuffer> bufInput = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufWeight = OpenCL.createReadBuffer(weight);
        CLBuffer<FloatBuffer> bufBias = OpenCL.createReadBuffer(bias);
        CLBuffer<FloatBuffer> bufResult = OpenCL.createReadWriteBuffer(result.length);

        OpenCL.getQueue()
                .putWriteBuffer(bufDropout, false)
                .putWriteBuffer(bufInput, false)
                .putWriteBuffer(bufWeight, false)
                .putWriteBuffer(bufBias, false);

        forwardKernel.rewind()
                .putArg(input.length)
                .putArg(outputSize)
                .putArgs(
                        bufDropout,
                        bufInput,
                        bufWeight,
                        bufBias,
                        bufResult);
        OpenCL.execute(forwardKernel, outputSize);

        if(activation instanceof SoftMaxFunction){
            CLBuffer<FloatBuffer> bufExped = OpenCL.createReadWriteBuffer(result.length);
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

        }else{
            CLKernel kernelAct = actKernels.get(activation.getName());
            kernelAct.rewind()
                    .putArg(bufResult);
            OpenCL.execute(kernelAct, outputSize);
        }

        OpenCL.getQueue().putReadBuffer(bufResult, true);
        bufResult.getBuffer().get(result);

        bufDropout.release();
        bufInput.release();
        bufWeight.release();
        bufBias.release();
        bufResult.release();

    }
}
