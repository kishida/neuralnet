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
        CLBuffer<FloatBuffer> bufWeightDelta = OpenCL.createReadWriteBuffer(weightDelta, biasDelta);
        OpenCL.getQueue()
            .putWriteBuffer(bufWeightDelta ,false)
            .putWriteBuffer(bufWeight      ,false);

        backword(inputSize, outputSize,
                dropout, input, delta,
                result, bufWeight, bufWeightDelta,
                newDelta,
                learningRate, activation);

        OpenCL.getQueue()
            .putReadBuffer(bufWeightDelta ,true);
        bufWeightDelta.getBuffer().get(weightDelta);

        bufWeight      .release();
        bufWeightDelta .release();
    }

    public void backword(int inputSize, int outputSize,
            int[] dropout, float[] input, float[] delta,
            float[] result, CLBuffer<FloatBuffer> bufWeight,
            CLBuffer<FloatBuffer> bufWeightDelta,
            float[] newDelta,
            float learningRate, ActivationFunction activation){
        CLBuffer<FloatBuffer> bufInput       = OpenCL.createReadBuffer(input);
        CLBuffer<FloatBuffer> bufDelta       = OpenCL.createReadBuffer(delta);
        CLBuffer<FloatBuffer> bufResult      = OpenCL.createReadBuffer(result);
        CLBuffer<FloatBuffer> bufNewDelta    = OpenCL.createWriteBuffer(newDelta.length);
        CLBuffer<IntBuffer>   bufDropout     = OpenCL.createReadBuffer(dropout);
        OpenCL.getQueue()
            .putWriteBuffer(bufInput       ,false)
            .putWriteBuffer(bufDelta       ,false)
            .putWriteBuffer(bufResult      ,false)
            .putWriteBuffer(bufDropout     ,false);

        backword(inputSize, outputSize,
                bufDropout, bufInput, bufDelta,
                bufResult, bufWeight, bufWeightDelta,
                bufNewDelta,
                learningRate, activation);

        OpenCL.getQueue()
            .putReadBuffer(bufNewDelta    ,false);
        bufNewDelta.getBuffer().get(newDelta);

        bufInput       .release();
        bufDelta       .release();
        bufResult      .release();
        bufNewDelta    .release();
        bufDropout     .release();

    }
    public void backword(int inputSize, int outputSize,
            CLBuffer<IntBuffer> bufDropout, CLBuffer<FloatBuffer> bufInput, CLBuffer<FloatBuffer> bufDelta,
            CLBuffer<FloatBuffer> bufResult, CLBuffer<FloatBuffer> bufWeight,
            CLBuffer<FloatBuffer> bufWeightDelta,
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

        CLBuffer<FloatBuffer> bufDiffed      = OpenCL.createReadWriteBuffer(outputSize);

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
        OpenCL.execute(kernelWeight, inputSize * outputSize + outputSize);

        bufDiffed      .release();
    }
    public void prepare(float momentam,
            int filterCount, int biasCount,
            CLBuffer<FloatBuffer> bufFilterDelta){

        CLKernel kernel = kernels.get("prepare");
        kernel.rewind()
                .putArg(momentam)
                .putArg(bufFilterDelta);
        OpenCL.execute(kernel, filterCount + biasCount);
    }

    public void join(float weightDecay, float learningRate,
            int filterCount, int biasCount, int count,
            CLBuffer<FloatBuffer> bufFilter, CLBuffer<FloatBuffer> bufFilterDelta){
        CLKernel kernelFilter = kernels.get("joinFilter");
        kernelFilter.rewind()
                .putArg(weightDecay)
                .putArg(learningRate)
                .putArg(count)
                .putArg(filterCount)
                .putArgs(
                    bufFilter,
                    bufFilterDelta);
        OpenCL.execute(kernelFilter, filterCount + biasCount);
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
