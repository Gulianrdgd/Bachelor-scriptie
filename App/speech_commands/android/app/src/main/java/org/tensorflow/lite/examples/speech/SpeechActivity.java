 /*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.lite.examples.speech;

import static android.content.ContentValues.TAG;

import android.Manifest;
import android.content.Context;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioAttributes;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;

import androidx.appcompat.widget.SwitchCompat;

import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import com.google.android.material.bottomsheet.BottomSheetBehavior;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import org.tensorflow.lite.Interpreter;

import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

import android.widget.Button;

/**
 * An activity that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words.
 */
public class SpeechActivity extends Activity
        implements View.OnClickListener, CompoundButton.OnCheckedChangeListener {

  // Constants that control the behavior of the recognition code and model
  // settings. See the audio recognition tutorial for a detailed explanation of
  // all these, but you should customize them to match your training settings if
  // you are running your own model.

  private static final int SAMPLE_RATE = 8000;
  private static final int SAMPLE_DURATION_MS = 1000;
  private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
  private static final long AVERAGE_WINDOW_DURATION_MS = 1000;
  private static final float DETECTION_THRESHOLD = 0.30f;
  private static final int SUPPRESSION_MS = 1500;


  // TODO: Investigate how these value affect our app, so that it becomes more
  // stable.
  private static final int MINIMUM_COUNT = 1;
  private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 1000;

  private static final String LABEL_FILENAME = "file:///android_asset/30.txt";
  private static final Integer NO_COMMANDS = 30;
  private static final String MODEL_FILENAME = "file:///android_asset/MFCC_8K_3.tflite";
  private static final String HANDLE_THREAD_NAME = "CameraBackground";

  // UI elements.
  private static final int REQUEST_RECORD_AUDIO = 13;
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

  // Working variables.
  short[] recordingBuffer = new short[RECORDING_LENGTH];
  int recordingOffset = 0;
  volatile boolean completeRecording = false;
  volatile boolean completeRecognition = false;
  volatile boolean processing = false;
  boolean shouldContinue = true;
  private Thread recordingThread;
  private Thread handlerT;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  private final ReentrantLock recordingBufferLock = new ReentrantLock();
  private final ReentrantLock tfLiteLock = new ReentrantLock();

  private List<String> labels = new ArrayList<String>();
  private List<String> displayedLabels = new ArrayList<>();
  private RecognizeCommands recognizeCommands = null;
  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private final Interpreter.Options tfLiteOptions = new Interpreter.Options();
  private MappedByteBuffer tfLiteModel;
  private Interpreter tfLite;
  private ImageView bottomSheetArrowImageView;

//  private TextView backgroundTextView,
//          upTextView,
//          downTextView,
//          leftTextView,
//          rightTextView,
//          onTextView,
//          offTextView,
//          stopTextView,
//          goTextView;
  private TextView sampleRateTextView, inferenceTimeTextView, resultTextView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private long lastProcessingTimeMs;
  private Handler handler = new Handler();
  private TextView selectedTextView = null;
  private HandlerThread backgroundThread;
  private Handler backgroundHandler;
  private AudioTrack track = null;
  int bufferSize;
  private AudioRecord record = null;

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  // Many ideas were taken from sound echo application shown in
  // http://androidsourcecode.blogspot.com/2013/07/android-audio-demo-audiotrack.html
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_sc_activity_speech);

    // Load the labels for the model, but only display those that don't start
    // with an underscore.
    String actualLabelFilename = LABEL_FILENAME.split("file:///android_asset/", -1)[1];
    Log.i(LOG_TAG, "Reading labels from: " + actualLabelFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(getAssets().open(actualLabelFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
        if (line.charAt(0) != '_') {
          displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
        }
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!", e);
    }

    // Set up an object to smooth recognition results to increase accuracy.
    recognizeCommands =
            new RecognizeCommands(
                    labels,
                    AVERAGE_WINDOW_DURATION_MS,
                    DETECTION_THRESHOLD,
                    SUPPRESSION_MS,
                    MINIMUM_COUNT,
                    MINIMUM_TIME_BETWEEN_SAMPLES_MS);

    String actualModelFilename = MODEL_FILENAME.split("file:///android_asset/", -1)[1];
    try {
      tfLiteModel = loadModelFile(getAssets(), actualModelFilename);
      recreateInterpreter();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    bufferSize =
            AudioRecord.getMinBufferSize(
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = SAMPLE_RATE * 2;
    }
    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
      // TODO: Consider calling
      //    ActivityCompat#requestPermissions
      // here to request the missing permissions, and then overriding
      //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
      //                                          int[] grantResults)
      // to handle the case where the user grants the permission. See the documentation
      // for ActivityCompat#requestPermissions for more details.
      return;
    }
    record =
            new AudioRecord(
                    MediaRecorder.AudioSource.DEFAULT,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize);

    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }

    Button btn = (Button) findViewById(R.id.record_button);
    btn.setOnClickListener(new View.OnClickListener() {
        public void onClick(View v) {
            startRecording();
       }
    });

    // Start the recording and recognition threads.
    requestMicrophonePermission();
    //startRecording();
    recordingHandler();
    startRecognition();

    sampleRateTextView = findViewById(R.id.sample_rate);
    inferenceTimeTextView = findViewById(R.id.inference_info);
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);

    resultTextView = findViewById(R.id.result);
//    upTextView = findViewById(R.id.up);
//    downTextView = findViewById(R.id.down);
//    leftTextView = findViewById(R.id.left);
//    rightTextView = findViewById(R.id.right);
//    onTextView = findViewById(R.id.on);
//    offTextView = findViewById(R.id.off);
//    stopTextView = findViewById(R.id.stop);
//    goTextView = findViewById(R.id.go);
//    backgroundTextView = findViewById(R.id.background);

    apiSwitchCompat.setOnCheckedChangeListener(this);

    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            int height = gestureLayout.getMeasuredHeight();

            sheetBehavior.setPeekHeight(height);
          }
        });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
        new BottomSheetBehavior.BottomSheetCallback() {
          @Override
          public void onStateChanged(@NonNull View bottomSheet, int newState) {
            switch (newState) {
              case BottomSheetBehavior.STATE_HIDDEN:
                break;
              case BottomSheetBehavior.STATE_EXPANDED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                }
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                }
                break;
              case BottomSheetBehavior.STATE_DRAGGING:
                break;
              case BottomSheetBehavior.STATE_SETTLING:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                break;
            }
          }

          @Override
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });

    plusImageView.setOnClickListener(this);
    minusImageView.setOnClickListener(this);

    sampleRateTextView.setText(SAMPLE_RATE + " Hz");
  }

  private void requestMicrophonePermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestPermissions(
          new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }
  }

  // TODO: In the new implementation here I have to just close the application
  // when no recording permission is given. I should not continuously record
  // audio for better control and debugging.
  // See https://developer.android.com/guide/topics/media/mediarecorder#java
  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_RECORD_AUDIO
        && grantResults.length > 0
        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
      //startRecording();
      startRecognition();
    }
  }

  public synchronized void startRecording() {
    Log.i(LOG_TAG, "In startRecording");
    if ((recordingThread != null) || processing) {
        System.out.println("Still processing");
        return;
    }
    shouldContinue = true;
    recordingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                record();
              }
            });
    recordingThread.start();
  }

  private void handler() {
    while (true) {
      //Log.v(LOG_TAG, "In handler");
      if (completeRecognition) {
          // This sets the recording thread to null but does not free up any
          // resources. For this reason we need to free the AudioTrack manually
          // by calling track.release in recognize. These variables here help
          // with the thread synchronization.
          recordingThread = null;
          completeRecognition = false;
      }
      try {
          Thread.sleep(200);
      } catch (InterruptedException e) {
          // ignore
      }
    }
  }

  public synchronized void recordingHandler() {
    handlerT =
        new Thread (
            new Runnable() {
                @Override
                public void run() {
                    handler();
                }
            });
    handlerT.start();
  }

  public synchronized void stopRecording() {
    if (recordingThread == null) {
      return;
    }
    shouldContinue = false;
    recordingThread = null;
  }

  private void record() {
    processing = true;
    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

    short[] audioBuffer = new short[bufferSize / 2];

    record.startRecording();
    Log.v(LOG_TAG, "Start recording");

    int readTimes = 0;
    // Loop, gathering audio data and copying it to a round-robin buffer.
    while (shouldContinue) {
      int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
//      if(readTimes <= 0){
//        readTimes++;
//        continue;
//      }

      int maxLength = recordingBuffer.length;
      int newRecordingOffset = recordingOffset + numberRead;
      int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
      int firstCopyLength = numberRead - secondCopyLength;

      // We store off all the data for the recognition thread to access. The ML
      // thread will copy out of this buffer into its own, while holding the
      // lock, so this should be thread safe.
      recordingBufferLock.lock();
      try {

        System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
        System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);

        recordingOffset = newRecordingOffset % maxLength;
        if (newRecordingOffset >= maxLength) {
            shouldContinue = false;
        }


      } finally {
        recordingBufferLock.unlock();
      }
    }

    Log.v(LOG_TAG, "Stopped recording");
    /*
     * https://stackoverflow.com/questions/20814932/android-encoded-pcm-16-8-bit-what-does-it-mean
     * PCM (pulse-code modulation) is a standard encoding scheme used in the
     * WAV file format. It consists of 8- or 16-bit samples; there are a number
     * of these per second of audio - that number is called the sample rate.
     * AudioTrack is used to play back PCM data; this can be done in real-time
     * while you write to its internal buffer (i.e. MODE_STREAM), or you can
     * fill the buffer and then play back (MODE_STATIC). If you go with the
     * streaming mode, it's important to continuously call write() to keep
     * filling the buffer during playback, otherwise the AudioTrack will stop
     * playing until it receives more data.
     */
    // Create a new track each time so that the sound is reproduced.
    track = new AudioTrack(
            AudioManager.MODE_IN_COMMUNICATION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            SAMPLE_RATE * 2,
            AudioTrack.MODE_STATIC);

    if (track == null) {
        record.stop();
        return;
    }

    track.write(recordingBuffer, 0, RECORDING_LENGTH, AudioTrack.WRITE_BLOCKING);
    track.play();
    shouldContinue = true;

    record.stop();
    //record.release();

    // Thread completed its execution
    completeRecording = true;
  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                recognize();
              }
            });
    recognitionThread.start();
  }

  // TODO: Use this after one recognition.
  public synchronized void stopRecognition() {
    if (recognitionThread == null) {
      return;
    }
    shouldContinueRecognition = false;
    recognitionThread = null;
  }

  private void recognize() {

    Log.v(LOG_TAG, "Start recognition");

    short[] inputBuffer = new short[RECORDING_LENGTH];
    float[][] floatInputBuffer = new float[RECORDING_LENGTH][1];
    float[][] outputScores = new float[1][labels.size()];
    int[] sampleRateList = new int[] {SAMPLE_RATE};

    // Loop, grabbing recorded data and running the recognition model on it.
    while (shouldContinueRecognition) {
      if (completeRecording) {
        long startTime = new Date().getTime();
        // The recording thread places data in this round-robin buffer, so lock to
        // make sure there's no writing happening and then copy it to our own
        // local version.
        recordingBufferLock.lock();
        try {
          int maxLength = recordingBuffer.length;
          int firstCopyLength = maxLength - recordingOffset;
          int secondCopyLength = recordingOffset;
          System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
          System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
        } finally {
          recordingBufferLock.unlock();
        }

        // We need to feed in float values between -1.0f and 1.0f, so divide the
        // signed 16-bit inputs.
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
          floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f;
        }

        /*
        Object[] inputArray = {floatInputBuffer, sampleRateList};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputScores);
        */

        /*
         * Start my implementation testing
         */
        // Declare variables that are used in my model.
        float[] floatInputBuffer2 = new float[SAMPLE_RATE];
        float[][] outputScores2 = new float[1][NO_COMMANDS];
        for (int i = 0; i < SAMPLE_RATE; i++) {
            floatInputBuffer2[i] = floatInputBuffer[i][0];
        }

        JLibrosa jlibrosa = new JLibrosa();
        int N_MFCC = 40;
        int N_FFT = 1103;
        int N_MELS = 128;
        int L_HOP = 441;
        int len_row, len_col, rows, cols;
        recordingBufferLock.lock();
        float[][] mfccs;
        try {
            // with right/6e74c582_nohash_1.wav it worked, but then it stopped.
            // with up/997867e7_nohash_0.wav it works as expected.
            // with up/15c563d7_nohash_3.wav it also works
            mfccs = jlibrosa.generateMFCCFeatures(floatInputBuffer2, SAMPLE_RATE, N_MFCC, N_FFT, N_MELS, L_HOP);
        } finally {
            recordingBufferLock.unlock();
        }
        len_row = mfccs.length;
        len_col = mfccs[0].length;
        System.out.println("MFCCs => rows: " + len_row + "\tcols: " + len_col);
        // TODO: This is an ugly hack to keep only the coefficients that we need
        // for the neural network based on its input shape. This should be
        // fixed in the future so that jlibrosa and librosa yield the same
        // results.
        rows = 40;
        cols = SAMPLE_RATE == 16000 ? 37 : 19;
        float[][][][] mfccs_correct = new float[1][rows][cols][1];
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            mfccs_correct[0][i][j][0] = mfccs[i][j];
          }
        }


        // Run our model
        Object[] inputArray_2 = {mfccs_correct};
        Map<Integer, Object> outputMap2 = new HashMap<>();
        outputMap2.put(0, outputScores2);

        Log.d(TAG, "input shape " + Arrays.toString(tfLite.getInputTensor(0).shape()));
        Log.d(TAG, "output shape " + Arrays.toString(tfLite.getOutputTensor(0).shape()));

        // Run the model
        tfLiteLock.lock();
        try {
            tfLite.runForMultipleInputsOutputs(inputArray_2, outputMap2);
            //tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        } finally {
            tfLiteLock.unlock();
        }

        // Use the smoother to figure out if we've had a real recognition event.
        long currentTime = System.currentTimeMillis();
        final RecognizeCommands.RecognitionResult result =
            recognizeCommands.processLatestResults(outputScores2[0], currentTime);

        System.out.println("BBBBBBBBB");
        System.out.println(Arrays.toString(outputScores2[0]));
        System.out.println(result.foundCommand);
        System.out.println("BBBBBBBBB");

        lastProcessingTimeMs = new Date().getTime() - startTime;
        runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                inferenceTimeTextView.setText(lastProcessingTimeMs + " ms");
                // If we do have a new command, highlight the right list entry.
                Log.v(LOG_TAG, "Result: " + result.foundCommand);
                resultTextView.setText(String.format("\n\n %s \n\n confidence: %s", result.foundCommand, result.score));
              }
            });

        // Set semaphores for a new recognition
        completeRecording = false;
        completeRecognition = true;
        processing = false;
        // Free up Audiotrack to avoid ENOMEM
        track.release();

        try {
          // We don't need to run too frequently, so snooze for a bit.
          Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
        } catch (InterruptedException e) {
          // Ignore
        }
      } else {
          //Log.v(LOG_TAG, "Not yet :)");
      }
    }
    Log.v(LOG_TAG, "End recognition");
  }

  @Override
  public void onClick(View v) {
    if ((v.getId() != R.id.plus) && (v.getId() != R.id.minus)) {
      return;
    }

    String threads = threadsTextView.getText().toString().trim();
    int numThreads = Integer.parseInt(threads);
    if (v.getId() == R.id.plus) {
      numThreads++;
    } else {
      if (numThreads == 1) {
        return;
      }
      numThreads--;
    }

    final int finalNumThreads = numThreads;
    threadsTextView.setText(String.valueOf(finalNumThreads));
    backgroundHandler.post(
        () -> {
          tfLiteOptions.setNumThreads(finalNumThreads);
          recreateInterpreter();
        });
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    backgroundHandler.post(
        () -> {
          tfLiteOptions.setUseNNAPI(isChecked);
          recreateInterpreter();
        });
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  private void recreateInterpreter() {
    tfLiteLock.lock();
    try {
      if (tfLite != null) {
        tfLite.close();
        tfLite = null;
      }
      tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
      tfLite.resizeInput(0, new int[] {100, 40, 1});
      //tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
      //tfLite.resizeInput(0, new int[] {RECORDING_LENGTH, 1});
      //tfLite.resizeInput(1, new int[] {1});
    } finally {
      tfLiteLock.unlock();
    }
  }

  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("amlan", "Interrupted when stopping background thread", e);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();
    startBackgroundThread();
  }

  @Override
  protected void onStop() {
    super.onStop();
    stopBackgroundThread();
  }
}
