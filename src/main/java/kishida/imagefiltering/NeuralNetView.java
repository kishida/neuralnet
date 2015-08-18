/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import static kishida.imagefiltering.ConvolutionalNet.random;

/**
 *
 * @author kishida
 */
public class NeuralNetView {
	public static void main(String[] args) {
		System.out.println("aa");
		new NeuralNetView("ニューラルネット");
	}
	public interface LearningMachine {
		//学習
		void learn(int cls, double[] data);
		//評価
		int trial(double[] data);
	}

	static class NeuralNet implements LearningMachine{
		ConvolutionalNet.FullyConnect fc1;
		ConvolutionalNet.FullyConnect fc2;
		ConvolutionalNet.ActivationFunction act = new ConvolutionalNet.LogisticFunction();

		NeuralNet(){
			double[][] wight = Stream.generate(() ->
					DoubleStream.generate(() -> (random.nextDouble() * 2 - 1)).limit(2).toArray()
			).limit(2).toArray(double[][]::new);
			double[] bias = DoubleStream.generate(() -> .1).limit(2).toArray();
			fc1 = new ConvolutionalNet.FullyConnect("fc1", 2, 2, wight, bias, 1, .3);

			double[][] wight2 = Stream.generate(() ->
					DoubleStream.generate(() -> (random.nextDouble() * 2 - 1)).limit(1).toArray()
			).limit(2).toArray(double[][]::new);
			double[] bias2 = DoubleStream.generate(() -> .1).limit(1).toArray();
			fc2 = new ConvolutionalNet.FullyConnect("fc2", 2, 1, wight2, bias2, 1, .3);
		}
		@Override
		public void learn(int cls, double[] data) {
			double p = cls == 1 ? 1 : 0;
			double[] mid = fc1.forward(data);
			double[] appmid = Arrays.stream(mid).map(d -> act.apply(d)).toArray();
			double[] result = fc2.forward(appmid);
			double[] appresult = Arrays.stream(result).map(d -> act.apply(d)).toArray();
			double[] delta = Arrays.stream(appresult).map(d -> p - d).toArray();
			double[] backmid = fc2.backward(appmid, delta, act);
			fc1.backward(data, backmid, act);
		}

		@Override
		public int trial(double[] data) {
			double[] mid = fc1.forward(data);
			double[] appmid = Arrays.stream(mid).map(d -> act.apply(d)).toArray();
			double[] result = fc2.forward(appmid);
			double[] appresult = Arrays.stream(result).map(d -> act.apply(d)).toArray();
			return appresult[0] > .5 ? 1 : 0;
		}

	}

	LearningMachine createLearningMachine(){
		return new NeuralNet();
	}

    public NeuralNetView(String title) {
        JFrame f = new JFrame(title);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setSize(420, 300);
        f.setLayout(new GridLayout(1, 2));

        //線形分離可能
        double[] linear1X = {0.15, 0.3, 0.35, 0.4, 0.55};
        double[] linear1Y = {0.3,  0.6, 0.25, 0.5, 0.4};
        double[] linear2X = {0.4,  0.7, 0.7, 0.85, 0.9};
        double[] linear2Y = {0.85, 0.9, 0.8, 0.7,  0.6};
        f.add(createGraph("線形分離可能",
                linear1X, linear1Y, linear2X, linear2Y));
        //線形分離不可能
        double[] nonlinear1X = {0.15, 0.45, 0.6, 0.3, 0.75, 0.9};
        double[] nonlinear1Y = {0.5,  0.85, 0.75,  0.75, 0.7, 0.55};
        double[] nonlinear2X = {0.2,  0.55, 0.4,  0.6, 0.8, 0.85};
        double[] nonlinear2Y = {0.3,  0.6,  0.55, 0.4, 0.55, 0.2};
        f.add(createGraph("線形分離不可能",
                nonlinear1X, nonlinear1Y, nonlinear2X, nonlinear2Y));

        f.setVisible(true);
    }

    JLabel createGraph(String title, double[] linear1X, double[] linear1Y, double[] linear2X, double[] linear2Y) {
        LearningMachine lm = createLearningMachine();
        //学習
        for(int i = 0; i < linear1X.length; ++i){
            lm.learn(-1, new double[]{linear1X[i], linear1Y[i]});
        }
        for(int i = 0; i < linear2X.length; ++i){
            lm.learn( 1, new double[]{linear2X[i], linear2Y[i]});
        }
        Image img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 200, 200);

        //判定結果
        for (int x = 0; x < 180; x += 2) {
            for (int y = 0; y < 180; y += 2) {
                int cls = lm.trial(new double[]{x / 180., y / 180.});
                g.setColor(cls == 1 ? new Color(192, 192, 255) : new Color(255, 192, 192));
                g.fillRect(x + 10, y + 10, 5, 5);
            }
        }
        //学習パターン
        for (int i = 0; i < linear1X.length; ++i) {
            int x = (int) (linear1X[i] * 180) + 10;
            int y = (int) (linear1Y[i] * 180) + 10;
            g.setColor(Color.RED);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        for (int i = 0; i < linear2X.length; ++i) {
            int x = (int) (linear2X[i] * 180) + 10;
            int y = (int) (linear2Y[i] * 180) + 10;
            g.setColor(Color.BLUE);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        //ラベル作成
        JLabel l = new JLabel(title, new ImageIcon(img), JLabel.CENTER);
        l.setVerticalTextPosition(JLabel.BOTTOM);
        l.setHorizontalTextPosition(JLabel.CENTER);
        return l;
    }
}
