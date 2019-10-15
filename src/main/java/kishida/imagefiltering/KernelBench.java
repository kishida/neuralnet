/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Consumer;

/**
 *
 * @author naoki
 */
public class KernelBench {
    static ConvolutionalNet.ConvolutionForwardKernel fKernel = new ConvolutionalNet.ConvolutionForwardKernel();
    static ConvolutionalNet.ConvolutionBackwordKernel bKernel = new ConvolutionalNet.ConvolutionBackwordKernel();
    static ConvolutionalNet.ConvolutionBackwordDeltaKernel bdKernel = new ConvolutionalNet.ConvolutionBackwordDeltaKernel();
    static ConvolutionalNet.ConvolutionBackwordFilterKernel bfKernel = new ConvolutionalNet.ConvolutionBackwordFilterKernel();
    static ConvolutionalNet.ConvolutionBackwordBiasKernel bbKernel = new ConvolutionalNet.ConvolutionBackwordBiasKernel();
    static ConvolutionalNet.NormalizeKernel nKernel = new ConvolutionalNet.NormalizeKernel();
    public static void main(String[] args) {
        Random r = new Random();
        
        double[] input = r.doubles(3 * 256 * 256).toArray();
        double[] filter = r.doubles(48 * 3 * 11 * 11).toArray();
        double[] bias = r.doubles(48).toArray();
        double[] result = r.doubles(48 * 128 * 128).toArray();
        double[] delta = r.doubles(result.length).toArray();
        double[] input2 = r.doubles(48 * 128 * 128).toArray();
        double[] filter2 = r.doubles(96 * 48 * 5 * 5).toArray();
        double[] bias2 = r.doubles(96).toArray();
        double[] result2 = r.doubles(96 * 16 * 16).toArray();
        double[] delta2 = r.doubles(result2.length).toArray();
        
        int insize = 256;
        int outsize3 = 384;
        int width3 = 14;
        int height3 = 14;
        int filtersize3 = 3;
        double[] result3 = r.doubles(outsize3 * width3 * height3).toArray();
        double[] filter3 = r.doubles(outsize3 * insize * filtersize3 * filtersize3).toArray();
        
        ConvolutionalNet.ConvolutionLayer clGpu1 = new ConvolutionalNet.ConvolutionLayer("testConv1 GPU", insize, width3, width3, outsize3, filtersize3, 1, true);
        clGpu1.result = r.doubles(clGpu1.getOutputSize()).toArray();
        ConvolutionalNet.ConvolutionLayer clCpu = new ConvolutionalNet.ConvolutionLayer("testConv CPU", insize, width3, width3, outsize3, filtersize3, 1, false);
        clCpu.result = r.doubles(clCpu.getOutputSize()).toArray();
        System.out.println(clGpu1);
        Consumer<double[]> printDouble = d -> System.out.printf("len:%d NaN:%s Inf:%s%n", d.length, Arrays.stream(d).anyMatch(Double::isNaN), Arrays.stream(d).anyMatch(Double::isInfinite));
        double[] delta4 = r.doubles(clGpu1.getOutputSize()).toArray();
        System.out.println(delta4.length);
        double[] input4 = r.doubles(insize * width3 * width3).toArray();
        System.out.println(input4.length);
        bench("384x14x14 gpu", () ->{
            bdKernel.backword(input4, delta4, result3, 
                    insize, width3, width3, filter3,
                    outsize3, width3, width3, filtersize3, 1, true);
            
        });        
        bench("384x16x16 cpu", () ->{
            clCpu.backward(input4, delta4);
        });
        ConvolutionalNet.RetifierdLinear act = new ConvolutionalNet.RetifierdLinear();
        double[] averages = new double[48 * 32 * 32];
        double[] rates = new double[averages.length];
        bench("normalize gpu", () -> 
            nKernel.normalize(input, 48, 32, 32, 5, averages, rates, .1, true));
        bench("normalize cpu", () -> 
            nKernel.normalize(input, 48, 32, 32, 5, averages, rates, .1, false));
        
        bench("complex optimize gpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true);
            
            bdKernel.backword(input2, delta2, result2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, true);
            bfKernel.backword(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, true);
            bbKernel.backwordBias(delta2, result2, 96, 16, 16, bias2, true);
            
            bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
            bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
            bbKernel.backwordBias(delta, result, 48, 128, 128, bias, true);
        });
        bench("complex gpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true);
            bKernel.backward(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, true);
            bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, true);
        });        
        
        bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, false);
        bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
        bench("1st filter gpu", () ->
            bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, true));
        bench("1st filter cpu", () ->
            bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, false));
        
        bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, false);
        bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
        bench("1st delta gpu", () ->
            bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, true));
        bench("1st delta cpu", () ->
            bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, false));
        
        bbKernel.backwordBias(delta, result, 48, 128, 128, bias, false);
        bbKernel.backwordBias(delta, result, 48, 128, 128, bias, true);
        bench("1st bias gpu", () ->
            bbKernel.backwordBias(delta, result, 48, 128, 128, bias, true));
        bench("1st bias cpu", () ->
            bbKernel.backwordBias(delta, result, 48, 128, 128, bias, false));
        
        bench("1st delta filter bias gpu", () ->{
            bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
            bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
            bbKernel.backwordBias(delta, result, 48, 128, 128, bias, true);
        });
        bench("1st delta filter bias cpu", () ->{
            bdKernel.backword(input, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, false);
            bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, false);
            bbKernel.backwordBias(delta, result, 48, 128, 128, bias, false);
        });
        
        fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, false);
        bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, false);
        fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
        bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, true);
        
        
        
        bench("1st gpu", () -> 
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true));
        bench("1st cpu", () -> 
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, false));
        
        bench("2nd gpu", () -> 
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true)
        );
        bench("2nd cpu", () -> 
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, false)
        );
        
        
        bench("1st back gpu", () -> 
            bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, true));
        bench("1st back cpu", () -> 
            bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, false));

        bench("2st back gpu", () -> 
            bKernel.backward(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, true));
        bench("2st back cpu", () -> 
            bKernel.backward(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, false));
        
        bench("complex optimize gpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true);
            
            bdKernel.backword(input2, delta2, result2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, true);
            bfKernel.backword(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, true);
            bbKernel.backwordBias(delta2, result2, 96, 16, 16, bias2, true);
            
            bdKernel.backword(input2, delta, result, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
            bfKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, true);
            bbKernel.backwordBias(delta, result, 48, 128, 128, bias, true);
        });
        bench("complex gpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true);
            bKernel.backward(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, true);
            bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, true);
        });
        bench("complex cpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, false);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, false);
            bKernel.backward(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, false);
            bKernel.backward(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, false);
        });
    }
    
    
    static void bench(String name, Runnable proc) {
        for (int i = 0; i < 10; ++i) {
            proc.run();
        }
        long start = System.currentTimeMillis();
        for (int i = 0; i < 100; ++i) {
            proc.run();
        }
        System.out.printf("%s:%.3fs%n", name, (System.currentTimeMillis() - start) / 1000.);
    }
}
