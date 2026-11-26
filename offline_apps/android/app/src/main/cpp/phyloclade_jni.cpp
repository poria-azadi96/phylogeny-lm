#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include "llama.h"

namespace {
std::mutex g_mutex;
llama_model * g_model = nullptr;
const llama_vocab * g_vocab = nullptr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_phyloscope_offline_MainActivity_nativeInitModel(JNIEnv * env, jobject /*thiz*/, jstring modelPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model != nullptr) return JNI_TRUE;

    const char * path = env->GetStringUTFChars(modelPath, nullptr);
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);

    if (g_model == nullptr) return JNI_FALSE;
    g_vocab = llama_model_get_vocab(g_model);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_phyloscope_offline_MainActivity_nativeInfer(JNIEnv * env, jobject /*thiz*/, jstring jprompt) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model == nullptr || g_vocab == nullptr) {
        return env->NewStringUTF("ERROR: Model not initialized");
    }

    const char * raw = env->GetStringUTFChars(jprompt, nullptr);
    std::string formatted = "<|im_start|>user\n";
    formatted += raw;
    formatted += "<|im_end|>\n<|im_start|>assistant\n";
    env->ReleaseStringUTFChars(jprompt, raw);

    int32_t n_prompt = -llama_tokenize(g_vocab, formatted.c_str(), (int32_t)formatted.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) n_prompt = (int32_t)formatted.size() + 16;
    std::vector<llama_token> tokens(n_prompt);
    int32_t n_tok = llama_tokenize(g_vocab, formatted.c_str(), (int32_t)formatted.size(), tokens.data(), (int32_t)tokens.size(), true, true);
    if (n_tok < 0) return env->NewStringUTF("ERROR: Tokenize failed");
    tokens.resize(n_tok);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    llama_context * ctx = llama_init_from_model(g_model, cparams);
    if (!ctx) return env->NewStringUTF("ERROR: Context init failed");

    llama_sampler * smpl = llama_sampler_init_greedy();

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        return env->NewStringUTF("ERROR: Decode failed");
    }

    std::string output;
    for (int step = 0; step < 24; ++step) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(g_vocab, tok)) break;

        char buf[256];
        int32_t n = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) output.append(buf, n);

        llama_token next_arr[1] = { tok };
        llama_batch next_batch = llama_batch_get_one(next_arr, 1);
        if (llama_decode(ctx, next_batch) != 0) break;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    return env->NewStringUTF(output.c_str());
}
