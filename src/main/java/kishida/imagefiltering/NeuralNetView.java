/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 *
 * @author kishida
 */
public class NeuralNetView {

    static Random random = new Random();

    LearningMachine createLearningMachine() {
        return new NeuralNet();
    }
    
    public static void main(String[] args) {
        new NeuralNetView("ニューラルネット");
    }

    public interface LearningMachine {

        //学習
        void learn(int cls, double[] data);

        //評価
        int trial(double[] data);
    }

    static class OldNeuralNet implements LearningMachine {

        List<Map.Entry<Integer, double[]>> patterns = new ArrayList<>();
        double[][] w;//入力→中間層の係数
        double[] midweight;//中間層→出力の係数
        int dim = 2;//入力パラメータ数
        int hiddendim = 3;//中間層の数+1

        public OldNeuralNet() {
            w = new double[hiddendim - 1][dim + 1];
            for (int i = 0; i < w.length; ++i) {
                for (int j = 0; j < w[i].length; ++j) {
                    w[i][j] = random.nextDouble() * 2 - 1;
                }
            }
            midweight = new double[hiddendim];
            for (int i = 0; i < hiddendim; ++i) {
                midweight[i] = random.nextDouble() * 2 - 1;
            }

        }

        @Override
        public void learn(int cls, double[] data) {

            int pcls = cls == 1 ? 1 : 0;

            final double localEp = .3;//学習係数
            double[] pattern = DoubleStream.concat(
                    DoubleStream.of(1),
                    Arrays.stream(data)).toArray();

            double[] middleout = new double[hiddendim];//中間層の出力値
            //入力層→中間層
            for (int j = 0; j < w.length; ++j) {
                double in = 0;
                for (int i = 0; i < pattern.length; ++i) {
                    in += pattern[i] * w[j][i];
                }
                middleout[j + 1] = sigmoid(in);
            }
            middleout[0] = 1;
            //中間層→出力層
            double out = 0;//出力
            for (int i = 0; i < middleout.length; ++i) {
                out += midweight[i] * middleout[i];
            }
            out = sigmoid(out);
            //出力層→中間層
            double d = (pcls - out) * out * (1 - out);
            double[] newDelta = new double[hiddendim];//中間層の補正値
            double[] oldhidden = midweight.clone();//補正前の係数
            for (int i = 0; i < hiddendim; ++i) {
                newDelta[i] = d * middleout[i];
                midweight[i] += newDelta[i] * localEp;
            }
            //中間層→入力層
            for (int i = 1; i < hiddendim; ++i) {
                double ek = newDelta[i] * oldhidden[i] * middleout[i] * (1 - middleout[i]);
                for (int j = 0; j < dim + 1; ++j) {
                    w[i - 1][j] += pattern[j] * ek * localEp;
                }
            }

        }

        private double sigmoid(double d) {
            return 1 / (1 + Math.exp(-d));
        }

        @Override
        public int trial(double[] data) {
            double[] pattern = new double[data.length + 1];
            for (int i = 0; i < data.length; ++i) {
                pattern[i + 1] = data[i];
            }
            pattern[0] = 1;

            double[] hiddendata = new double[hiddendim];
            //入力層→中間層
            for (int j = 0; j < w.length; ++j) {
                double in = 0;
                for (int i = 0; i < pattern.length; ++i) {
                    in += pattern[i] * w[j][i];
                }
                hiddendata[j + 1] = sigmoid(in);
            }
            hiddendata[0] = 1;
            //中間層→出力層
            double out = 0;
            for (int i = 0; i < hiddendata.length; ++i) {
                out += hiddendata[i] * midweight[i];
            }
            return (sigmoid(out) > .5) ? 1 : -1;
        }

    }

    static class NeuralNet implements LearningMachine {

        ConvolutionalNet.FullyConnect fc1;
        ConvolutionalNet.FullyConnect fc2;
        ConvolutionalNet.ActivationFunction act = new ConvolutionalNet.LogisticFunction();

        NeuralNet() {
            double[][] wight = Stream.generate(()
                    -> DoubleStream.generate(() -> (random.nextDouble() * 2 - 1)).limit(2).toArray()
            ).limit(2).toArray(double[][]::new);
            double[] bias = DoubleStream.generate(() -> random.nextDouble() * 2 - 1).limit(2).toArray();
            fc1 = new ConvolutionalNet.FullyConnect("fc1", 2, 2, wight, bias, 1, .3);

            double[][] wight2 = Stream.generate(()
                    -> DoubleStream.generate(() -> (random.nextDouble() * 2 - 1)).limit(1).toArray()
            ).limit(2).toArray(double[][]::new);
            double[] bias2 = DoubleStream.generate(() -> random.nextDouble() * 2 - 1).limit(1).toArray();
            fc2 = new ConvolutionalNet.FullyConnect("fc2", 2, 1, wight2, bias2, 1, .3);
        }

        @Override
        public void learn(int cls, double[] data) {
            double p = cls == 1 ? 1 : 0;
            double[] mid = fc1.forward(data);
            double[] appmid = Arrays.stream(mid).map(act::apply).toArray();
            fc1.result = appmid;
            double[] result = fc2.forward(appmid);
            double[] appresult = Arrays.stream(result).map(act::apply).toArray();
            fc2.result = appresult;
            double[] delta = Arrays.stream(appresult).map(d -> p - d).toArray();
            double[] backmid = fc2.backward(appmid, delta, act);
            fc1.backward(data, backmid, act);
        }

        @Override
        public int trial(double[] data) {
            double[] mid = fc1.forward(data);
            double[] appmid = Arrays.stream(mid).map(act::apply).toArray();
            double[] result = fc2.forward(appmid);
            double[] appresult = Arrays.stream(result).map(act::apply).toArray();
            return appresult[0] > .5 ? 1 : 0;
        }

    }

    JLabel lblCounter;

    public NeuralNetView(String title) {
        JFrame f = new JFrame(title);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setSize(420, 300);
        JPanel center = new JPanel(new GridLayout(1, 2));
        f.add(BorderLayout.CENTER, center);

        JPanel bottom = new JPanel(new FlowLayout());
        f.add(BorderLayout.SOUTH, bottom);
        lblCounter = new JLabel("Waiting");
        lblCounter.setHorizontalAlignment(JLabel.CENTER);
        bottom.add(BorderLayout.SOUTH, lblCounter);

        //線形分離可能
        double[] linear1X = {0.15, 0.3, 0.35, 0.4, 0.55};
        double[] linear1Y = {0.3, 0.6, 0.25, 0.5, 0.4};
        double[] linear2X = {0.4, 0.7, 0.7, 0.85, 0.9};
        double[] linear2Y = {0.85, 0.9, 0.8, 0.7, 0.6};
        center.add(lblLinear = createLabel("線形分離可能"));
        //線形分離不可能
        double[] nonlinear1X = {0.15, 0.45, 0.6, 0.3, 0.75, 0.9};
        double[] nonlinear1Y = {0.5, 0.85, 0.75, 0.75, 0.7, 0.55};
        double[] nonlinear2X = {0.2, 0.55, 0.4, 0.6, 0.8, 0.85};
        double[] nonlinear2Y = {0.3, 0.6, 0.55, 0.4, 0.55, 0.2};
        center.add(lblNonlinear = createLabel("線形分離不可能"));

        JButton btn = new JButton("Start");
        bottom.add(btn);
        f.setVisible(true);

        btn.addActionListener(ae -> {
            btn.setEnabled(false);
            new Thread(() -> {
                LearningMachine lmLinear = createLearningMachine();
                LearningMachine lmNonlinear = createLearningMachine();
                //学習
                List<Param> paramsLinear = Stream.concat(
                        IntStream.range(0, linear1X.length).mapToObj(i -> new Param(-1, linear1X[i], linear1Y[i])),
                        IntStream.range(0, linear2X.length).mapToObj(i -> new Param(1, linear2X[i], linear2Y[i]))
                ).collect(Collectors.toList());
                Collections.shuffle(paramsLinear);
                List<Param> paramsNonLinear = Stream.concat(
                        IntStream.range(0, nonlinear1X.length).mapToObj(i -> new Param(-1, nonlinear1X[i], nonlinear1Y[i])),
                        IntStream.range(0, nonlinear1X.length).mapToObj(i -> new Param(1, nonlinear2X[i], nonlinear2Y[i]))
                ).collect(Collectors.toList());
                Collections.shuffle(paramsNonLinear);
                for (int i = 0; i < 5000; ++i) {
                    paramsLinear.stream().forEach(p -> lmLinear.learn(p.supervise, new double[]{p.x, p.y}));
                    Image imgLinear = createGraphImg(lmLinear, linear1X, linear1Y, linear2X, linear2Y);

                    paramsNonLinear.stream().forEach(p -> lmNonlinear.learn(p.supervise, new double[]{p.x, p.y}));
                    Image imgNonlinear = createGraphImg(lmNonlinear, nonlinear1X, nonlinear1Y, nonlinear2X, nonlinear2Y);

                    if (i % 10 == 0) {
                        String strCount = String.valueOf(i);
                        EventQueue.invokeLater(() -> {
                            lblLinear.setIcon(new ImageIcon(imgLinear));
                            lblNonlinear.setIcon(new ImageIcon(imgNonlinear));
                            lblCounter.setText(strCount);
                        });
                    }
                }
                EventQueue.invokeLater(() -> {
                    lblCounter.setText("Finish");
                    btn.setEnabled(true);
                });
            }).start();
        });
    }

    JLabel lblLinear;
    JLabel lblNonlinear;

    JLabel createLabel(String title) {
        JLabel l = new JLabel(title);
        l.setVerticalTextPosition(JLabel.BOTTOM);
        l.setHorizontalTextPosition(JLabel.CENTER);
        return l;
    }

    static class Param {

        int supervise;
        double x;
        double y;

        public Param(int supervise, double x, double y) {
            this.supervise = supervise;
            this.x = x;
            this.y = y;
        }
    }

    JLabel createGraph(String title, double[] linear1X, double[] linear1Y, double[] linear2X, double[] linear2Y) {
        LearningMachine lm = createLearningMachine();
        //学習
        List<Param> params = Stream.concat(
                IntStream.range(0, linear1X.length).mapToObj(i -> new Param(-1, linear1X[i], linear1Y[i])),
                IntStream.range(0, linear2X.length).mapToObj(i -> new Param(1, linear2X[i], linear2Y[i]))
        ).collect(Collectors.toList());
        Collections.shuffle(params);
        for (int i = 0; i < 10000; ++i) {
            params.stream().forEach(p -> lm.learn(p.supervise, new double[]{p.x, p.y}));
        }
        Image img = createGraphImg(lm, linear1X, linear1Y, linear2X, linear2Y);
        //ラベル作成
        JLabel l = new JLabel(title, new ImageIcon(img), JLabel.CENTER);
        l.setVerticalTextPosition(JLabel.BOTTOM);
        l.setHorizontalTextPosition(JLabel.CENTER);
        return l;
    }

    private Image createGraphImg(LearningMachine lm,
            double[] linear1X, double[] linear1Y, double[] linear2X, double[] linear2Y) {
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
        return img;
    }
}
