#define _WIN32_WINNT 0x0601
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <cctype>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "shell32.lib")

static int g_uiPort = 0;
static int g_aiPort = 0;
static std::string g_baseDir;
static PROCESS_INFORMATION g_llamaPi = {};
static volatile bool g_running = true;

static std::string getExeDir() {
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(NULL, buf, MAX_PATH);
    std::string s(buf, len);
    size_t pos = s.find_last_of("\\/");
    return pos == std::string::npos ? "." : s.substr(0, pos);
}

static int findFreePort() {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    bind(s, (sockaddr*)&addr, sizeof(addr));
    int len = sizeof(addr);
    getsockname(s, (sockaddr*)&addr, &len);
    int port = ntohs(addr.sin_port);
    closesocket(s);
    return port;
}

static std::string readFileBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) { char buf[8]; wsprintfA(buf, "\\u%04x", c); out += buf; }
        else out.push_back((char)c);
    }
    return out;
}

static std::string jsonUnescape(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            if (n == 'n') out.push_back('\n');
            else if (n == 'r') out.push_back('\r');
            else if (n == 't') out.push_back('\t');
            else if (n == '"') out.push_back('"');
            else if (n == '\\') out.push_back('\\');
            else if (n == '/') out.push_back('/');
            else out.push_back(n);
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

static std::string extractJsonStringField(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + pat.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    std::string raw;
    for (size_t i = pos + 1; i < json.size(); ++i) {
        if (json[i] == '\\' && i + 1 < json.size()) {
            raw.push_back(json[i]);
            raw.push_back(json[++i]);
        } else if (json[i] == '"') {
            break;
        } else {
            raw.push_back(json[i]);
        }
    }
    return jsonUnescape(raw);
}

static std::string httpRequest(const std::wstring& host, INTERNET_PORT port, bool useHttps,
                               const std::wstring& method, const std::wstring& path,
                               const std::string& body, const std::wstring& contentType, int& outCode) {
    outCode = 0;
    HINTERNET hSess = WinHttpOpen(L"PhyloScopeOffline/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, NULL, NULL, 0);
    if (!hSess) return "";
    HINTERNET hConn = WinHttpConnect(hSess, host.c_str(), port, 0);
    if (!hConn) { WinHttpCloseHandle(hSess); return ""; }
    DWORD flags = useHttps ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hReq = WinHttpOpenRequest(hConn, method.c_str(), path.c_str(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hReq) { WinHttpCloseHandle(hConn); WinHttpCloseHandle(hSess); return ""; }

    WinHttpSetTimeouts(hReq, 10000, 15000, 30000, 60000);
    std::wstring hdrs = contentType.empty() ? L"" : (L"Content-Type: " + contentType + L"\r\n");
    BOOL ok = WinHttpSendRequest(hReq, hdrs.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : hdrs.c_str(),
                                 (DWORD)hdrs.size(),
                                 body.empty() ? WINHTTP_NO_REQUEST_DATA : (LPVOID)body.data(),
                                 (DWORD)body.size(), (DWORD)body.size(), 0);
    std::string resp;
    if (ok && WinHttpReceiveResponse(hReq, NULL)) {
        DWORD status = 0, sz = sizeof(status);
        WinHttpQueryHeaders(hReq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, NULL, &status, &sz, NULL);
        outCode = (int)status;
        DWORD avail = 0;
        while (WinHttpQueryDataAvailable(hReq, &avail) && avail > 0) {
            std::vector<char> buf(avail);
            DWORD read = 0;
            if (WinHttpReadData(hReq, buf.data(), avail, &read) && read > 0) {
                resp.append(buf.data(), read);
            }
        }
    }
    WinHttpCloseHandle(hReq);
    WinHttpCloseHandle(hConn);
    WinHttpCloseHandle(hSess);
    return resp;
}

static std::string fetchUrlFull(const std::string& url, int& outCode) {
    std::wstring wurl(url.begin(), url.end());
    URL_COMPONENTS uc = {};
    uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {}, path[2048] = {};
    uc.lpszHostName = host; uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path; uc.dwUrlPathLength = 2048;
    if (!WinHttpCrackUrl(wurl.c_str(), (DWORD)wurl.size(), 0, &uc)) {
        outCode = 400;
        return "";
    }
    return httpRequest(host, uc.nPort, uc.nScheme == INTERNET_SCHEME_HTTPS, L"GET", path, "", L"", outCode);
}

static std::string fetchFromDatabase(const std::string& query, bool& ok) {
    ok = false;
    std::string q = query;
    while (!q.empty() && isspace((unsigned char)q.front())) q.erase(q.begin());
    while (!q.empty() && isspace((unsigned char)q.back())) q.pop_back();
    if (q.empty()) return "Empty database URL or accession list.";

    std::string targetUrl;
    if (q.rfind("http://", 0) == 0 || q.rfind("https://", 0) == 0) {
        size_t pos = q.find("ncbi.nlm.nih.gov/nuccore/");
        if (pos == std::string::npos) pos = q.find("ncbi.nlm.nih.gov/nucleotide/");
        if (pos != std::string::npos) {
            size_t slash = q.find('/', pos + 18);
            std::string acc = q.substr(slash + 1);
            size_t endp = acc.find_first_of("?#/");
            if (endp != std::string::npos) acc = acc.substr(0, endp);
            targetUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=" + acc + "&rettype=fasta&retmode=text";
        } else {
            targetUrl = q;
        }
    } else {
        std::string ids;
        for (char c : q) {
            if (isspace((unsigned char)c) || c == ';' || c == ',') {
                if (!ids.empty() && ids.back() != ',') ids.push_back(',');
            } else {
                ids.push_back(c);
            }
        }
        if (!ids.empty() && ids.back() == ',') ids.pop_back();
        targetUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=" + ids + "&rettype=fasta&retmode=text";
    }

    int code = 0;
    std::string data = fetchUrlFull(targetUrl, code);
    if (code >= 200 && code < 300 && !data.empty()) {
        ok = true;
        return data;
    }
    return "Failed to fetch database (HTTP " + std::to_string(code) + ").";
}

static void sendResponse(SOCKET client, int code, const std::string& ctype, const std::string& body) {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << code << (code == 200 ? " OK" : " Error") << "\r\n"
        << "Content-Type: " << ctype << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n";
    std::string full = hdr.str() + body;
    send(client, full.data(), (int)full.size(), 0);
}

static void handleClient(SOCKET client) {
    std::string req;
    char buf[4096];
    int n = recv(client, buf, sizeof(buf), 0);
    if (n <= 0) { closesocket(client); return; }
    req.append(buf, n);

    size_t hdrEnd = req.find("\r\n\r\n");
    while (hdrEnd == std::string::npos) {
        n = recv(client, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, n);
        hdrEnd = req.find("\r\n\r\n");
    }

    size_t clPos = req.find("Content-Length:");
    if (clPos == std::string::npos) clPos = req.find("content-length:");
    int contentLen = 0;
    if (clPos != std::string::npos && hdrEnd != std::string::npos && clPos < hdrEnd) {
        contentLen = atoi(req.c_str() + clPos + 15);
    }
    std::string body = hdrEnd != std::string::npos ? req.substr(hdrEnd + 4) : "";
    while ((int)body.size() < contentLen) {
        n = recv(client, buf, sizeof(buf), 0);
        if (n <= 0) break;
        body.append(buf, n);
    }
    if ((int)body.size() > contentLen && contentLen > 0) body.resize(contentLen);

    std::istringstream ls(req);
    std::string method, path;
    ls >> method >> path;

    if (path == "/" || path == "/index.html") {
        std::string html = readFileBytes(g_baseDir + "\\web\\index.html");
        sendResponse(client, 200, "text/html; charset=utf-8", html);
    } else if (path == "/phylo_engine.js") {
        std::string js = readFileBytes(g_baseDir + "\\web\\phylo_engine.js");
        sendResponse(client, 200, "application/javascript; charset=utf-8", js);
    } else if (path == "/api/health") {
        int code = 0;
        std::string h = httpRequest(L"127.0.0.1", (INTERNET_PORT)g_aiPort, false, L"GET", L"/health", "", L"", code);
        bool ready = (code == 200 && h.find("ok") != std::string::npos);
        std::string json = std::string("{\"status\":\"ok\",\"ai_ready\":") + (ready ? "true" : "false") +
                           ",\"model\":\"PhyloClade-135M (GGUF Q8_0)\"}";
        sendResponse(client, 200, "application/json; charset=utf-8", json);
    } else if (path == "/api/fetch_db" && method == "POST") {
        std::string query = extractJsonStringField(body, "query");
        bool ok = false;
        std::string res = fetchFromDatabase(query, ok);
        if (ok) {
            sendResponse(client, 200, "application/json; charset=utf-8", "{\"fasta\":\"" + jsonEscape(res) + "\"}");
        } else {
            sendResponse(client, 500, "application/json; charset=utf-8", "{\"error\":\"" + jsonEscape(res) + "\"}");
        }
    } else if (path == "/api/infer" && method == "POST") {
        std::string prompt = extractJsonStringField(body, "prompt");
        std::string payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"" + jsonEscape(prompt) + "\"}],\"temperature\":0.0,\"max_tokens\":24}";
        int code = 0;
        std::string aiResp = httpRequest(L"127.0.0.1", (INTERNET_PORT)g_aiPort, false, L"POST", L"/v1/chat/completions", payload, L"application/json", code);
        if (code == 200) {
            std::string pred = extractJsonStringField(aiResp, "content");
            sendResponse(client, 200, "application/json; charset=utf-8", "{\"prediction\":\"" + jsonEscape(pred) + "\"}");
        } else {
            sendResponse(client, 500, "application/json; charset=utf-8", "{\"error\":\"Local AI engine not ready yet (HTTP " + std::to_string(code) + ")\"}");
        }
    } else {
        sendResponse(client, 404, "text/plain", "Not Found");
    }
    closesocket(client);
}

static void serverThread() {
    SOCKET srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((u_short)g_uiPort);
    bind(srv, (sockaddr*)&addr, sizeof(addr));
    listen(srv, 16);
    while (g_running) {
        SOCKET c = accept(srv, NULL, NULL);
        if (c == INVALID_SOCKET) break;
        std::thread(handleClient, c).detach();
    }
    closesocket(srv);
}

static void startLlamaServer() {
    std::string exe = g_baseDir + "\\bin\\llama-server.exe";
    std::string model = g_baseDir + "\\models\\phyloclade-135m-q8_0.gguf";
    std::string cmd = "\"" + exe + "\" -m \"" + model + "\" --host 127.0.0.1 --port " + std::to_string(g_aiPort) + " -c 1024 -t 4";
    std::string workDir = g_baseDir + "\\bin";
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    std::vector<char> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back('\0');
    CreateProcessA(NULL, cmdBuf.data(), NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, workDir.c_str(), &si, &g_llamaPi);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_COMMAND && LOWORD(wParam) == 1001) {
        std::string url = "http://127.0.0.1:" + std::to_string(g_uiPort) + "/";
        ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
        return 0;
    }
    if (msg == WM_COMMAND && LOWORD(wParam) == 1002) {
        DestroyWindow(hwnd);
        return 0;
    }
    if (msg == WM_DESTROY) {
        g_running = false;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    g_baseDir = getExeDir();
    g_uiPort = findFreePort();
    g_aiPort = findFreePort();

    startLlamaServer();
    std::thread(serverThread).detach();

    std::string url = "http://127.0.0.1:" + std::to_string(g_uiPort) + "/";
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);

    WNDCLASSA wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "PhyloScopeWnd";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassA(&wc);

    HWND hwnd = CreateWindowExA(0, "PhyloScopeWnd", "PhyloScope Offline (PhyloClade-135M)",
                                WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                CW_USEDEFAULT, CW_USEDEFAULT, 420, 180, NULL, NULL, hInst, NULL);
    std::string info = "PhyloScope Offline is running locally on:\r\n" + url + "\r\n\r\nClose this window to stop the local AI engine.";
    CreateWindowA("STATIC", info.c_str(), WS_CHILD | WS_VISIBLE, 18, 16, 370, 68, hwnd, NULL, hInst, NULL);
    CreateWindowA("BUTTON", "Open PhyloScope UI", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 18, 96, 175, 34, hwnd, (HMENU)1001, hInst, NULL);
    CreateWindowA("BUTTON", "Stop & Exit", WS_CHILD | WS_VISIBLE, 210, 96, 175, 34, hwnd, (HMENU)1002, hInst, NULL);

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    if (g_llamaPi.hProcess) {
        TerminateProcess(g_llamaPi.hProcess, 0);
        CloseHandle(g_llamaPi.hProcess);
        CloseHandle(g_llamaPi.hThread);
    }
    WSACleanup();
    return 0;
}
