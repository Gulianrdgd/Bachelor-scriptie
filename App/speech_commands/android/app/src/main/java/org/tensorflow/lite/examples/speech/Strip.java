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
    private final int N_SAMPLES = 100;
    private final int N_TEST = 2000;
    private Interpreter tfLite;
    private int labelSize;
    private ReentrantLock tfLiteLock;

    private List<float[][]> test_mfccs;


    public Strip(List<float[][]> test_mfccs, Interpreter tfLite, int labelSize, ReentrantLock tfLiteLock) {
//        readMfccFromJson();
        this.test_mfccs = test_mfccs;
        this.tfLite = tfLite;
        this.labelSize = labelSize;
        this.tfLiteLock = tfLiteLock;
    }

    private float[][] superimpose(float[][] mfcc1, float[][] mfcc2){
        float[][] superimposed = new float[MFCC_SIZE_ROW][MFCC_SIZE_COL];

        for(int i=0; i<MFCC_SIZE_ROW; i++){
            for(int j=0; j<MFCC_SIZE_COL; j++){
                superimposed[i][j] = mfcc1[i][j] + mfcc2[i][j];
            }
        }
        return superimposed;
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

//        Log.d(TAG, "input shape " + Arrays.toString(tfLite.getInputTensor(0).shape()));
//        Log.d(TAG, "output shape " + Arrays.toString(tfLite.getOutputTensor(0).shape()));

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
        for(int i=0; i<N_TEST; i++){
            float[][] mfcc_copy = mfcc.clone();

            int random_int = (int)Math.floor(Math.random() * (test_mfccs.size() - 1));
            float[][] random_mfcc = test_mfccs.get(random_int);
            if(random_mfcc[0].length != 19){
                continue;
            }

            for (int j=0; j<N_SAMPLES; j++){
                mfcc_copy = superimpose(mfcc_copy, random_mfcc);
            }

            float[] result = predict(mfcc_copy);
            float calculatedEntropy = 0;
            for (float x : result) {
                calculatedEntropy += x * Math.log(x) / Math.log(2);
            }
            System.out.println("Entropy: " + calculatedEntropy);
            entropy[i] = calculatedEntropy;
        }

        float[] finalEntropy = new float[entropy.length];
        for (int k=0; k<entropy.length; k++){
            finalEntropy[k] = (entropy[k] / N_SAMPLES);
        }

        Arrays.sort(finalEntropy);
        System.out.println("Entropy: min(" + finalEntropy[0] + "), max("+ finalEntropy[N_TEST-1] + ")");

    }


}
