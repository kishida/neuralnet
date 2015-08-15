/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.util.Random;

/**
 *
 * @author naoki
 */
public class KernelBench {
    static ConvolutionalNet.ConvolutionForwardKernel fKernel = new ConvolutionalNet.ConvolutionForwardKernel();
    static ConvolutionalNet.ConvolutionBackwordKernel bKernel = new ConvolutionalNet.ConvolutionBackwordKernel();
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
        
        ConvolutionalNet.RetifierdLinear act = new ConvolutionalNet.RetifierdLinear();

        fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, false);
        bKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, act, false);
        //fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
        //bKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, act, true);
        
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
            bKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, act, true));
        bench("1st back cpu", () -> 
            bKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, act, false));

        bench("2st back gpu", () -> 
            bKernel.backword(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true));
        bench("2st back cpu", () -> 
            bKernel.backword(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, false));
        
        bench("complex gpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, true);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true);
            bKernel.backword(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, true);
            bKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, act, true);
        });
        bench("complex cpu", () -> {
            fKernel.forward(input, 3, 256, 256, filter, 48, 256 / 2, 256 / 2, 11, 2, bias, act, false);
            fKernel.forward(input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, false);
            bKernel.backword(delta2, result2, input2, 48, 32, 32, filter2, 96, 16, 16, 5, 2, bias2, act, false);
            bKernel.backword(delta, result, input, 3, 256, 256, filter, 48, 128, 128, 11, 2, bias, act, false);
        });
    }
    
    
    static void bench(String name, Runnable proc) {
        for (int i = 0; i < 10; ++i) {
            proc.run();
        }
        long start = System.currentTimeMillis();
        for (int i = 0; i < 50; ++i) {
            proc.run();
        }
        System.out.printf("%s:%.3fs%n", name, (System.currentTimeMillis() - start) / 1000.);
    }
}
