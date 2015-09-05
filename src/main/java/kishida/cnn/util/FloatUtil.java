package kishida.cnn.util;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class FloatUtil {

    private FloatUtil() {
    }

    public static DoubleSummaryStatistics summary(float[] data){
        return summary(data, 0, data.length);
    }
    public static DoubleSummaryStatistics summary(float[] data, int start, int end){
        return IntStream.range(start, end).parallel().mapToDouble(i -> data[i]).summaryStatistics();
    }
    public static float floatSum(float[] data){
        return (float)IntStream.range(0, data.length).parallel().
                mapToDouble(i -> data[i]).sum();
    }

    public static float[] createGaussianArray(int size, float std, Random random){
        float[] result = new float[size];
        for(int i = 0; i < result.length; ++i){
            result[i] = (float)(random.nextGaussian() * std);
        }
        return result;
    }

    public static float[] createArray(int size, float value){
        float[] result = new float[size];
        Arrays.fill(result, value);
        return result;
    }

    public static DoubleStream toDoubleStream(float[] data){
        return IntStream.range(0, data.length).mapToDouble(i -> data[i]);
    }
}
