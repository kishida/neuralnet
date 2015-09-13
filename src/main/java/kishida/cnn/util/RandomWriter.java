/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author naoki
 */
public class RandomWriter {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        Random r1 = new Random(2345);
        r1.nextInt();
        byte[] myRandom =
            {115, 113, 0, 126,    0,  0,  0,   0,
               0,   0, 0,   0,    0,  0,  0,   0,
               0, -17, 6,  84, -120, -9, -1, 120};
        Random r2 = getRandomFromState(myRandom);
        System.out.println(r1.nextInt());
        System.out.println(r2.nextInt());
        System.out.println(r1.nextInt());
        System.out.println(r2.nextInt());
    }

    public static byte[] getRandomState(Random r){
        try(ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(new Random(1234));//仮のランダム
            int headerSize = baos.size();
            oos.writeObject(r);
            byte[] randomInfo = Arrays.copyOfRange(baos.toByteArray(), headerSize, baos.size());
            return randomInfo;
        }catch(IOException ex){
            throw new UncheckedIOException(ex);
        }
    }
    public static Random getRandomFromState(byte[] state){
        try(ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(new Random(1234));
            baos.write(state);
            try(ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
                ObjectInputStream ois = new ObjectInputStream(bais);){
                ois.readObject();//ランダムをひとつ捨てる
                return (Random) ois.readObject();
            }
        }catch(IOException ex){
            throw new UncheckedIOException(ex);
        } catch (ClassNotFoundException ex) {
            throw new RuntimeException(ex);
        }
    }
}
