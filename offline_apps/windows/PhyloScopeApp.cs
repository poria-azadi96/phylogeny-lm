using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using System.Collections.Generic;

namespace PhyloScopeDesktop
{
    static class Program
    {
        private static Process _llamaProc;
        private static HttpListener _listener;
        private static int _uiPort;
        private static int _aiPort;
        private static string _baseDir;

        [STAThread]
        static void Main()
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            _baseDir = AppDomain.CurrentDomain.BaseDirectory;
            _uiPort = GetFreePort();
            _aiPort = GetFreePort();

            StartLlamaServer();
            StartHttpServer();

            Process.Start("http://127.0.0.1:" + _uiPort + "/");

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            Form form = new Form();
            form.Text = "PhyloScope Offline (PhyloClade-135M)";
            form.Size = new Size(420, 190);
            form.StartPosition = FormStartPosition.CenterScreen;
            form.FormBorderStyle = FormBorderStyle.FixedDialog;
            form.MaximizeBox = false;
            form.BackColor = Color.FromArgb(14, 22, 35);
            form.ForeColor = Color.White;

            Label lbl = new Label();
            lbl.Text = "PhyloScope Offline is running locally on http://127.0.0.1:" + _uiPort + "\n\n" +
                       "AI Engine: PhyloClade-135M (100% Offline GGUF)\n" +
                       "Keep this window open while using PhyloScope.";
            lbl.Location = new Point(20, 18);
            lbl.Size = new Size(370, 75);
            lbl.Font = new Font("Segoe UI", 9.5f);
            form.Controls.Add(lbl);

            Button btnOpen = new Button();
            btnOpen.Text = "Open PhyloScope UI";
            btnOpen.Location = new Point(20, 105);
            btnOpen.Size = new Size(175, 34);
            btnOpen.FlatStyle = FlatStyle.Flat;
            btnOpen.BackColor = Color.FromArgb(52, 211, 153);
            btnOpen.ForeColor = Color.Black;
            btnOpen.Click += (s, e) => Process.Start("http://127.0.0.1:" + _uiPort + "/");
            form.Controls.Add(btnOpen);

            Button btnExit = new Button();
            btnExit.Text = "Stop & Exit";
            btnExit.Location = new Point(210, 105);
            btnExit.Size = new Size(175, 34);
            btnExit.FlatStyle = FlatStyle.Flat;
            btnExit.BackColor = Color.FromArgb(30, 45, 68);
            btnExit.ForeColor = Color.White;
            btnExit.Click += (s, e) => form.Close();
            form.Controls.Add(btnExit);

            form.FormClosing += (s, e) => Shutdown();
            Application.Run(form);
        }

        static int GetFreePort()
        {
            TcpListener l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int p = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return p;
        }

        static void StartLlamaServer()
        {
            string exe = Path.Combine(_baseDir, "bin", "llama-server.exe");
            string model = Path.Combine(_baseDir, "models", "phyloclade-135m-q8_0.gguf");
            if (!File.Exists(exe) || !File.Exists(model)) return;

            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = exe;
            psi.Arguments = string.Format("-m \"{0}\" --host 127.0.0.1 --port {1} -c 1024 -t 4", model, _aiPort);
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.WorkingDirectory = Path.Combine(_baseDir, "bin");
            _llamaProc = Process.Start(psi);
        }

        static void StartHttpServer()
        {
            _listener = new HttpListener();
            _listener.Prefixes.Add("http://127.0.0.1:" + _uiPort + "/");
            _listener.Start();

            Thread t = new Thread(() =>
            {
                while (_listener != null && _listener.IsListening)
                {
                    try
                    {
                        HttpListenerContext ctx = _listener.GetContext();
                        ThreadPool.QueueUserWorkItem(_ => HandleRequest(ctx));
                    }
                    catch { break; }
                }
            });
            t.IsBackground = true;
            t.Start();
        }

        static void HandleRequest(HttpListenerContext ctx)
        {
            try
            {
                string path = ctx.Request.Url.AbsolutePath;
                if (path == "/" || path == "/index.html")
                {
                    ServeFile(ctx, Path.Combine(_baseDir, "web", "index.html"), "text/html; charset=utf-8");
                    return;
                }
                if (path == "/phylo_engine.js")
                {
                    ServeFile(ctx, Path.Combine(_baseDir, "web", "phylo_engine.js"), "application/javascript; charset=utf-8");
                    return;
                }
                if (path == "/api/health")
                {
                    bool aiReady = false;
                    try
                    {
                        using (WebClient wc = new WebClient())
                        {
                            string h = wc.DownloadString("http://127.0.0.1:" + _aiPort + "/health");
                            aiReady = h.Contains("ok");
                        }
                    }
                    catch { aiReady = false; }
                    WriteJson(ctx, 200, new Dictionary<string, object> {
                        { "status", "ok" },
                        { "ai_ready", aiReady },
                        { "model", "PhyloClade-135M (GGUF Q8_0)" }
                    });
                    return;
                }
                if (path == "/api/fetch_db" && ctx.Request.HttpMethod == "POST")
                {
                    string body = ReadBody(ctx);
                    var js = new JavaScriptSerializer();
                    var req = js.Deserialize<Dictionary<string, object>>(body);
                    string query = req != null && req.ContainsKey("query") ? Convert.ToString(req["query"]).Trim() : "";
                    string fasta = FetchFromDatabase(query);
                    WriteJson(ctx, 200, new Dictionary<string, object> { { "fasta", fasta } });
                    return;
                }
                if (path == "/api/infer" && ctx.Request.HttpMethod == "POST")
                {
                    string body = ReadBody(ctx);
                    var js = new JavaScriptSerializer();
                    var req = js.Deserialize<Dictionary<string, object>>(body);
                    string prompt = req != null && req.ContainsKey("prompt") ? Convert.ToString(req["prompt"]) : "";
                    string pred = CallLocalLlama(prompt);
                    WriteJson(ctx, 200, new Dictionary<string, object> { { "prediction", pred } });
                    return;
                }
                ctx.Response.StatusCode = 404;
                ctx.Response.Close();
            }
            catch (Exception ex)
            {
                try
                {
                    WriteJson(ctx, 500, new Dictionary<string, object> { { "error", ex.Message } });
                }
                catch { }
            }
        }

        static string FetchFromDatabase(string query)
        {
            if (string.IsNullOrEmpty(query)) throw new Exception("Empty database link or accession list.");
            using (WebClient wc = new WebClient())
            {
                wc.Encoding = Encoding.UTF8;
                wc.Headers.Add("User-Agent", "PhyloScopeOffline/1.0");
                if (query.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                    query.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                {
                    // Check if it is an NCBI nuccore page URL
                    Match m = Regex.Match(query, @"ncbi\.nlm\.nih\.gov/(?:nuccore|nucleotide)/([A-Za-z0-9_.]+)", RegexOptions.IgnoreCase);
                    if (m.Success)
                    {
                        string acc = m.Groups[1].Value;
                        return wc.DownloadString("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=" + Uri.EscapeDataString(acc) + "&rettype=fasta&retmode=text");
                    }
                    return wc.DownloadString(query);
                }
                // Comma or whitespace separated NCBI accessions
                string[] parts = Regex.Split(query, @"[\s,;]+");
                List<string> ids = new List<string>();
                foreach (string p in parts) if (!string.IsNullOrWhiteSpace(p)) ids.Add(p.Trim());
                if (ids.Count == 0) throw new Exception("No valid accession IDs provided.");
                string joined = string.Join(",", ids.ToArray());
                return wc.DownloadString("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=" + Uri.EscapeDataString(joined) + "&rettype=fasta&retmode=text");
            }
        }

        static string CallLocalLlama(string prompt)
        {
            var js = new JavaScriptSerializer();
            var payload = new Dictionary<string, object> {
                { "messages", new object[] { new Dictionary<string, object> { { "role", "user" }, { "content", prompt } } } },
                { "temperature", 0.0 },
                { "max_tokens", 24 }
            };
            string json = js.Serialize(payload);
            using (WebClient wc = new WebClient())
            {
                wc.Encoding = Encoding.UTF8;
                wc.Headers[HttpRequestHeader.ContentType] = "application/json";
                string resp = wc.UploadString("http://127.0.0.1:" + _aiPort + "/v1/chat/completions", "POST", json);
                var parsed = js.Deserialize<Dictionary<string, object>>(resp);
                var choices = parsed["choices"] as System.Collections.ArrayList;
                var first = choices[0] as Dictionary<string, object>;
                var msg = first["message"] as Dictionary<string, object>;
                return Convert.ToString(msg["content"]);
            }
        }

        static string ReadBody(HttpListenerContext ctx)
        {
            using (StreamReader sr = new StreamReader(ctx.Request.InputStream, Encoding.UTF8))
                return sr.ReadToEnd();
        }

        static void ServeFile(HttpListenerContext ctx, string filePath, string contentType)
        {
            byte[] bytes = File.ReadAllBytes(filePath);
            ctx.Response.ContentType = contentType;
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
            ctx.Response.Close();
        }

        static void WriteJson(HttpListenerContext ctx, int status, object obj)
        {
            var js = new JavaScriptSerializer();
            byte[] bytes = Encoding.UTF8.GetBytes(js.Serialize(obj));
            ctx.Response.StatusCode = status;
            ctx.Response.ContentType = "application/json; charset=utf-8";
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
            ctx.Response.Close();
        }

        static void Shutdown()
        {
            try { if (_listener != null) _listener.Stop(); } catch { }
            try
            {
                if (_llamaProc != null && !_llamaProc.HasExited)
                {
                    _llamaProc.Kill();
                    _llamaProc.WaitForExit(2000);
                }
            }
            catch { }
        }
    }
}
