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
        CLBuffer<FloatBuffer> bufWeight      = OpenCL.createReadBuffer(weight);
        CLBuffer<FloatBuffer> bufWeightDelta = OpenCL.createReadWriteBuffer(weightDelta);
        CLBuffer<FloatBuffer> bufBiasDelta   = OpenCL.createReadWriteBuffer(biasDelta);
        OpenCL.getQueue()
            .putWriteBuffer(bufWeightDelta ,false)
            .putWriteBuffer(bufBiasDelta   ,false)
            .putWriteBuffer(bufWeight      ,false);

        backword(inputSize, outputSize,
                dropout, input, delta,
                result, bufWeight, bufWeightDelta, bufBiasDelta,
                newDelta,
                learningRate, activation);

        OpenCL.getQueue()
            .putReadBuffer(bufBiasDelta   ,false)
            .putReadBuffer(bufWeightDelta ,true);
        bufBiasDelta.getBuffer().get(biasDelta);
        bufWeightDelta.getBuffer().get(weightDelta);

        bufWeight      .release();
        bufWeightDelta .release();
        bufBiasDelta   .release();
    }

    public void backword(int inputSize, int outputSize,
            int[] dropout, float[] input, float[] delta,
            float[] result, CLBuffer<FloatBuffer> bufWeight,
            CLBuffer<FloatBuffer> bufWeightDelta, CLBuffer<FloatBuffer> bufBiasDelta,
            float[] newDelta,
            float learningRate, ActivationFunction activation){
        CLBuffer<FloatBuffer> bufInput       = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufDelta       = OpenCL.createReadBuffer(delta);
        CLBuffer<FloatBuffer> bufResult      = OpenCL.createReadBuffer(result);
        CLBuffer<FloatBuffer> bufNewDelta    = OpenCL.createWriteBuffer(newDelta.length);
        OpenCL.getQueue()
            .putWriteBuffer(bufInput       ,false)
            .putWriteBuffer(bufDelta       ,false)
            .putWriteBuffer(bufResult      ,false);

        backword(inputSize, outputSize,
                dropout, bufInput, bufDelta,
                bufResult, bufWeight, bufWeightDelta, bufBiasDelta,
                bufNewDelta,
                learningRate, activation);

        OpenCL.getQueue()
            .putReadBuffer(bufNewDelta    ,false);
        bufNewDelta.getBuffer().get(newDelta);

        bufInput       .release();
        bufDelta       .release();
        bufResult      .release();
        bufNewDelta    .release();

    }
    public void backword(int inputSize, int outputSize,
            int[] dropout, CLBuffer<FloatBuffer> bufInput, CLBuffer<FloatBuffer> bufDelta,
            CLBuffer<FloatBuffer> bufResult, CLBuffer<FloatBuffer> bufWeight,
            CLBuffer<FloatBuffer> bufWeightDelta, CLBuffer<FloatBuffer> bufBiasDelta,
            CLBuffer<FloatBuffer> bufNewDelta,
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
        CLBuffer<FloatBuffer> bufDiffed      = OpenCL.createReadWriteBuffer(outputSize);

        OpenCL.getQueue()
            .putWriteBuffer(bufDropout     ,false);

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

        bufDropout     .release();
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
