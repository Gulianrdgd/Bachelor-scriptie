package org.tensorflow.lite.examples.speech;


import static android.content.ContentValues.TAG;

import android.util.JsonReader;
import android.util.Log;

import com.google.gson.Gson;
import com.jlibrosa.audio.JLibrosa;

import org.tensorflow.lite.Interpreter;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

public class Strip {

    private int MFCC_SIZE_ROW = 40;
    private int MFCC_SIZE_COL = 19;
    private int n_samples = 100;
    private Interpreter tfLite;
    private int labelSize;
    private ReentrantLock tfLiteLock;
    private RecognizeCommands recognizeCommands;

    private List<float[][]> test_mfccs;

    public Strip(Interpreter tfLite, int labelSize, ReentrantLock tfLiteLock, RecognizeCommands recognizeCommands) {
        readMfccFromJson();
        this.tfLite = tfLite;
        this.labelSize = labelSize;
        this.tfLiteLock = tfLiteLock;
        this.recognizeCommands = recognizeCommands;
    }
    private void readMfccFromJson(){
        try {
            JSONFile jsonData = new Gson().fromJson(new FileReader(
                    "mfccs.json"), JSONFile.class);
            test_mfccs = jsonData.getMfccs();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }

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

    private float predict(float[][]mfcc){
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

        Log.d(TAG, "input shape " + Arrays.toString(tfLite.getInputTensor(0).shape()));
        Log.d(TAG, "output shape " + Arrays.toString(tfLite.getOutputTensor(0).shape()));

        // Run the model
        tfLiteLock.lock();
        try {
            tfLite.runForMultipleInputsOutputs(inputArray_2, outputMap2);
        } finally {
            tfLiteLock.unlock();
        }

        // Use the smoother to figure out if we've had a real recognition event.
        long currentTime = System.currentTimeMillis();
        final RecognizeCommands.RecognitionResult result =
                recognizeCommands.processLatestResults(outputScores[0], currentTime);

        return result.score;
    }
    private float getEntropy(float[][] mfcc){

        int random_int = (int)Math.floor(Math.random() * (test_mfccs.size() + 1));
        float[][] random_mfcc = test_mfccs.get(random_int);

        for (int i=0; i<n_samples; i++){
            mfcc = superimpose(mfcc, random_mfcc);
        }

        predict(mfcc);


    }


}
