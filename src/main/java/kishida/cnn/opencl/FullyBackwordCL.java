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
import kishida.cnn.activation.RectifiedLinear;

/**
 *
 * @author naoki
 */
public class FullyBackwordCL {
    public static FullyBackwordCL INSTANCE = new FullyBackwordCL();
    CLProgram prog;
    Map<String, CLKernel> kernels;
    CLProgram progActivation;
    Map<String, CLKernel> actKernels;

    private FullyBackwordCL() {
    }

    public void backword(int inputSize, int outputSize,
            int[] dropout, float[] input, float[] delta,
            float[] result, float[] weight,
            float[] weightDelta, float[] biasDelta,
            float[] newDelta,
            float learningRate, ActivationFunction activation){
        if(prog == null){
            prog = OpenCL.compile("fully_backword.cl");
            kernels = prog.createCLKernels();
        }
        if(progActivation == null){
            progActivation = OpenCL.compile("activation.cl");
            actKernels = progActivation.createCLKernels();
        }

        CLBuffer<IntBuffer>   bufDropout     = OpenCL.createReadBuffer(dropout);
        CLBuffer<FloatBuffer> bufInput       = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufDelta       = OpenCL.createReadBuffer(delta);
        CLBuffer<FloatBuffer> bufResult      = OpenCL.createReadBuffer(result);
        CLBuffer<FloatBuffer> bufWeight      = OpenCL.createReadBuffer(weight);
        CLBuffer<FloatBuffer> bufNewDelta    = OpenCL.createWriteBuffer(newDelta.length);
        CLBuffer<FloatBuffer> bufWeightDelta = OpenCL.createReadWriteBuffer(weightDelta);
        CLBuffer<FloatBuffer> bufBiasDelta   = OpenCL.createReadWriteBuffer(biasDelta);
        CLBuffer<FloatBuffer> bufDiffed      = OpenCL.createReadWriteBuffer(result.length);

        OpenCL.getQueue()
            .putWriteBuffer(bufDropout     ,false)
            .putWriteBuffer(bufInput       ,false)
            .putWriteBuffer(bufDelta       ,false)
            .putWriteBuffer(bufResult      ,false)
            .putWriteBuffer(bufWeight      ,false)
            .putWriteBuffer(bufWeightDelta ,false)
            .putWriteBuffer(bufBiasDelta   ,false);

        CLKernel actKernel = actKernels.get(activation.getName() + "_diff");
        actKernel.rewind()
                .putArg(bufResult)
                .putArg(bufDiffed);
        OpenCL.execute(actKernel, outputSize);

        CLKernel kernelDelta = kernels.get("backword_delta");
        kernelDelta.rewind()
                .putArg(outputSize)
                .putArgs(
                        bufDropout,
                        bufDelta,
                        bufDiffed,
                        bufWeight,
                        bufNewDelta);
        OpenCL.execute(kernelDelta, inputSize);

        CLKernel kernelWeight = kernels.get("backword_weight");
        kernelWeight.rewind()
                .putArg(outputSize)
                .putArg(learningRate)
                .putArgs(
                        bufDropout,
                        bufInput,
                        bufDelta,
                        bufDiffed,
                        bufWeight,
                        bufWeightDelta);
        OpenCL.execute(kernelWeight, inputSize * outputSize);

        CLKernel kernelBias = kernels.get("backword_bias");
        kernelBias.rewind()
                .putArg(outputSize)
                .putArg(learningRate)
                .putArg(bufDropout)
                .putArg(bufDelta)
                .putArg(bufDiffed)
                .putArg(bufBiasDelta);
        OpenCL.execute(kernelBias, outputSize);

        OpenCL.getQueue()
            .putReadBuffer(bufNewDelta    ,false)
            .putReadBuffer(bufBiasDelta   ,false)
            .putReadBuffer(bufWeightDelta ,true);
        bufNewDelta.getBuffer().get(newDelta);
        bufBiasDelta.getBuffer().get(biasDelta);
        bufWeightDelta.getBuffer().get(weightDelta);

        bufDropout     .release();
        bufInput       .release();
        bufDelta       .release();
        bufResult      .release();
        bufWeight      .release();
        bufNewDelta    .release();
        bufWeightDelta .release();
        bufBiasDelta   .release();
        bufDiffed      .release();
    }

    public static void main(String[] args) {
        int inputSize = 5;
        int outputSize = 9;
        int[] dropout = new int[outputSize];
        float[] delta = new float[outputSize];
        float[] input = new float[inputSize];
        float[] result = new float[outputSize];
        float[] weight = new float[inputSize * outputSize];
        float[] weightDelta = new float[weight.length];
        float[] biasDelta = new float[outputSize];
        float[] newDelta = new float[inputSize];

        INSTANCE.backword(inputSize, outputSize,
                dropout, input, delta, result, weight,
                weightDelta, biasDelta, newDelta, 0.001f, new RectifiedLinear());
    }
}
