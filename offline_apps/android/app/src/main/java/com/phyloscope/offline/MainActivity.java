package com.phyloscope.offline;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("phyloclade_jni");
    }

    private native boolean nativeInitModel(String modelPath);
    private native String nativeInfer(String prompt);

    private volatile boolean modelReady = false;
    private WebView webView;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        webView = new WebView(this);
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setAllowFileAccess(true);
        ws.setAllowContentAccess(true);

        webView.setWebViewClient(new WebViewClient());
        webView.addJavascriptInterface(new AndroidBridge(), "AndroidBridge");
        setContentView(webView);
        webView.loadUrl("file:///android_asset/index.html");

        new Thread(() -> {
            try {
                File outFile = new File(getFilesDir(), "phyloclade-135m-q4_k_m.gguf");
                if (!outFile.exists() || outFile.length() < 1000000L) {
                    try (InputStream is = getAssets().open("phyloclade-135m-q4_k_m.gguf");
                         FileOutputStream fos = new FileOutputStream(outFile)) {
                        byte[] buf = new byte[65536];
                        int n;
                        while ((n = is.read(buf)) > 0) {
                            fos.write(buf, 0, n);
                        }
                    }
                }
                modelReady = nativeInitModel(outFile.getAbsolutePath());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    public class AndroidBridge {
        @JavascriptInterface
        public boolean isModelReady() {
            return modelReady;
        }

        @JavascriptInterface
        public String infer(String prompt) {
            if (!modelReady) return "ERROR: Model is still loading, please wait a moment.";
            return nativeInfer(prompt);
        }

        @JavascriptInterface
        public String fetchDatabase(String query) {
            try {
                String q = query == null ? "" : query.trim();
                if (q.isEmpty()) return "ERROR: Empty database link or accession list.";
                String targetUrl;
                if (q.startsWith("http://") || q.startsWith("https://")) {
                    Pattern p = Pattern.compile("ncbi\\.nlm\\.nih\\.gov/(?:nuccore|nucleotide)/([A-Za-z0-9_.]+)", Pattern.CASE_INSENSITIVE);
                    Matcher m = p.matcher(q);
                    if (m.find()) {
                        String acc = m.group(1);
                        targetUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=" +
                                URLEncoder.encode(acc, "UTF-8") + "&rettype=fasta&retmode=text";
                    } else {
                        targetUrl = q;
                    }
                } else {
                    String joined = q.replaceAll("[\\s,;]+", ",");
                    targetUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=" +
                            URLEncoder.encode(joined, "UTF-8") + "&rettype=fasta&retmode=text";
                }
                HttpURLConnection conn = (HttpURLConnection) new URL(targetUrl).openConnection();
                conn.setRequestProperty("User-Agent", "PhyloScopeAndroid/1.0");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(30000);
                int code = conn.getResponseCode();
                if (code < 200 || code >= 300) {
                    return "ERROR: HTTP " + code;
                }
                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        sb.append(line).append("\n");
                    }
                }
                return sb.toString();
            } catch (Exception e) {
                return "ERROR: " + e.getMessage();
            }
        }
    }
}
