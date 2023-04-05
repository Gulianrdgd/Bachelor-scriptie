package org.tensorflow.lite.examples.speech;

import org.tensorflow.lite.Interpreter;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import it.unimi.dsi.util.XoRoShiRo128PlusRandom;

public class Strip {

    private final int MFCC_SIZE_ROW = 40;
    private final int MFCC_SIZE_COL = 19;
    private final int N_SAMPLES = 20;
    public final int N_TEST = 200;
    private final Interpreter tfLite;
    private final int labelSize;

    private final XoRoShiRo128PlusRandom randomNumGenerator = new XoRoShiRo128PlusRandom(42);
    private final ReentrantLock tfLiteLock;

    private final List<String> labels;

    public Strip(Interpreter tfLite, int labelSize, ReentrantLock tfLiteLock, List<String> labels) {
        this.labels = labels;
        this.tfLite = tfLite;
        this.labelSize = labelSize;
        this.tfLiteLock = tfLiteLock;

    }

    private float[][] superimpose(float[][] mfcc){
        float[][] superimposed = new float[MFCC_SIZE_ROW][MFCC_SIZE_COL];

        for(int i=0; i<MFCC_SIZE_ROW; i++){
            for(int j=0; j<MFCC_SIZE_COL; j++){
                superimposed[i][j] = mfcc[i][j] +  (float) randomNumGenerator.nextDoubleFast();
            }
        }
        return superimposed;
    }

    private String getPredictedWord(float[] prediction) {
        int maxAt = 0;

        for (int i = 0; i < prediction.length; i++) {
            maxAt = prediction[i] > prediction[maxAt] ? i : maxAt;
        }

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

    public float[] getEntropy(float[][] mfcc){
        float[] entropy = new float[N_TEST];

        // For the number of tests we superimpose the audio
        for(int i=0; i<N_TEST; i++){
            // Clone the array so that we never modify the original
            float[][] mfcc_copy = mfcc.clone();

            // For the number of samples we superimpose everytime we get a new random mfcc
            for (int j=0; j<N_SAMPLES; j++){
                mfcc_copy = superimpose(mfcc_copy);
            }

            float[] result = predict(mfcc_copy);

            // Checks to make sure it is working
//            System.out.println(Arrays.toString(result));
//            System.out.println(getPredictedWord(result));

            // Calculate entropy by summing the log of the probability
            float calculatedEntropy = 0;
            for (float x : result) {
                float temp = (float) (x * Math.log(x) / Math.log(2));
                if (Float.isNaN(temp)) {
                    temp = 0;
                }
                calculatedEntropy += temp;
            }
            entropy[i] = (-1 * calculatedEntropy) / N_SAMPLES;
        }

        // Sort and print the results
        Arrays.sort(entropy);
        System.out.println(Arrays.toString(entropy));
        System.out.println("Entropy: min(" + entropy[0] + "), max("+ entropy[N_TEST-1] + ") diff(" + (entropy[N_TEST-1] - entropy[0]) + ")");
        return entropy;
    }


}
