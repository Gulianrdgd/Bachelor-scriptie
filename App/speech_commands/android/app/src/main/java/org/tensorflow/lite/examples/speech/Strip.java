package org.tensorflow.lite.examples.speech;


import static android.content.ContentValues.TAG;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import com.google.gson.Gson;

import org.tensorflow.lite.Interpreter;

import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

public class Strip {

    private final int MFCC_SIZE_ROW = 40;
    private final int MFCC_SIZE_COL = 19;
    private final int N_SAMPLES = 20;
    private final int N_TEST = 200;
    private final Interpreter tfLite;
    private final int labelSize;
    private final ReentrantLock tfLiteLock;

    private final List<float[][]> test_mfccs;

    private final List<String> labels;

    public Strip(List<float[][]> test_mfccs, Interpreter tfLite, int labelSize, ReentrantLock tfLiteLock, List<String> labels) {
//        readMfccFromJson();
        this.labels = labels;
        this.test_mfccs = test_mfccs;
        this.tfLite = tfLite;
        this.labelSize = labelSize;
        this.tfLiteLock = tfLiteLock;
    }

    private float[][] superimpose(float[][] mfcc1, float[][] mfcc2){
        float[][] superimposed = new float[MFCC_SIZE_ROW][MFCC_SIZE_COL];

        for(int i=0; i<MFCC_SIZE_ROW; i++){
            for(int j=0; j<MFCC_SIZE_COL; j++){
//                superimposed[i][j] = mfcc1[i][j] +  mfcc2[i][j];
                superimposed[i][j] = mfcc1[i][j] +  (float) Math.random();
            }
        }
        return superimposed;
    }

    private String getPredictedWord(float[] prediction) {
        int maxAt = 0;

        for (int i = 0; i < prediction.length; i++) {
            maxAt = prediction[i] > prediction[maxAt] ? i : maxAt;
        }
        System.out.println("MAX AT:" + maxAt + " with val: "+ prediction[maxAt]);
        return labels.get(maxAt);
    }

    private float[] predict(float[][]mfcc){
        float[][][][] mfccs_correct = new float[1][MFCC_SIZE_ROW][MFCC_SIZE_COL][1];

        for (int i = 0; i < MFCC_SIZE_ROW; i++) {
            for (int j = 0; j < MFCC_SIZE_COL; j++) {
                mfccs_correct[0][i][j][0] = mfcc[i][j];
            }
        }

        float[][] outputScores = new float[1][labelSize];

        // Run our model
        Object[] inputArray_2 = {mfccs_correct};
        Map<Integer, Object> outputMap2 = new HashMap<>();
        outputMap2.put(0, outputScores);

        // Run the model
        tfLiteLock.lock();
        try {
            tfLite.runForMultipleInputsOutputs(inputArray_2, outputMap2);
        } finally {
            tfLiteLock.unlock();
        }

        return outputScores[0];
    }

    public void getEntropy(float[][] mfcc){
        float[] entropy = new float[N_TEST];
        float[][] random_mfcc = new float[1][1];

        // For the number of tests we superimpose the audio
        for(int i=0; i<N_TEST; i++){
            // Clone the array so that we never modify the original
            float[][] mfcc_copy = mfcc.clone();

            // For the number of samples we superimpose everytime we get a new random mfcc
            for (int j=0; j<N_SAMPLES; j++){
//                int random_int = (int)Math.floor(Math.random() * (test_mfccs.size() - 1));
//                float[][] random_mfcc = test_mfccs.get(random_int);
//
//                // Sanity check
//                if(random_mfcc[0].length != 19){
//                    continue;
//                }
                mfcc_copy = superimpose(mfcc_copy,  random_mfcc);
            }

            float[] result = predict(mfcc_copy);

            // Checks to make sure it is working
            System.out.println(Arrays.toString(result));
            System.out.println(getPredictedWord(result));

            // Calculate entropy by summing the log of the probability
            float calculatedEntropy = 0;
            for (float x : result) {
                float temp = (float) (x * Math.log(x) / Math.log(2));
                if (Float.isNaN(temp)) {
                    temp = 0;
                }
                calculatedEntropy += temp;
            }
            entropy[i] = -1 * calculatedEntropy;
        }

        // In [15]:
        float[] finalEntropy = new float[entropy.length];
        for (int k=0; k<entropy.length; k++){
            float temp = entropy[k] / N_SAMPLES;
            finalEntropy[k] = temp;
        }

        Arrays.sort(finalEntropy);
        System.out.println(Arrays.toString(finalEntropy));
        System.out.println("Entropy: min(" + finalEntropy[0] + "), max("+ finalEntropy[N_TEST-1] + ")");

    }


}
