package com.exynos.vkcoopmat;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity {

    static {
        System.loadLibrary("native-bench");
    }

    // Implemented in native-bench.cpp.
    public native String runBenchmark(String shaderDir);

    private TextView output;
    private Button runButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        android.widget.LinearLayout root = new android.widget.LinearLayout(this);
        root.setOrientation(android.widget.LinearLayout.VERTICAL);
        int pad = (int) (12 * getResources().getDisplayMetrics().density);
        root.setPadding(pad, pad, pad, pad);

        runButton = new Button(this);
        runButton.setText("Run Cooperative-Matrix Benchmark");
        root.addView(runButton);

        output = new TextView(this);
        output.setTypeface(android.graphics.Typeface.MONOSPACE);
        output.setTextIsSelectable(true);
        output.setText("Press the button to run the benchmark.\n");
        root.addView(output);

        scroll.addView(root);
        setContentView(scroll);

        runButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { startRun(); }
        });

        // Run once automatically on launch.
        startRun();
    }

    private void startRun() {
        runButton.setEnabled(false);
        output.setText("Running... (also see logcat tag VKCOOP)\n");
        final String shaderDir = copyShaders();
        new Thread(new Runnable() {
            @Override public void run() {
                String result;
                try {
                    result = runBenchmark(shaderDir);
                } catch (Throwable t) {
                    result = "Native call failed: " + t;
                }
                final String finalResult = result;
                saveReport(finalResult);
                new Handler(Looper.getMainLooper()).post(new Runnable() {
                    @Override public void run() {
                        output.setText(finalResult);
                        runButton.setEnabled(true);
                    }
                });
            }
        }).start();
    }

    /** Copies bundled SPIR-V from assets/shaders into filesDir/shaders, returns that path. */
    private String copyShaders() {
        File dir = new File(getFilesDir(), "shaders");
        dir.mkdirs();
        try {
            String[] files = getAssets().list("shaders");
            if (files != null) {
                for (String name : files) {
                    InputStream in = getAssets().open("shaders/" + name);
                    OutputStream out = new FileOutputStream(new File(dir, name));
                    byte[] buf = new byte[8192];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                    in.close();
                    out.close();
                }
            }
        } catch (IOException e) {
            // Non-fatal; native side reports missing shaders.
        }
        return dir.getAbsolutePath();
    }

    /** Saves the report to app external files dir for easy adb pull. */
    private void saveReport(String text) {
        try {
            File f = new File(getExternalFilesDir(null), "coopmat_report.txt");
            FileOutputStream out = new FileOutputStream(f);
            out.write(text.getBytes("UTF-8"));
            out.close();
        } catch (IOException e) {
            // ignore
        }
    }
}
