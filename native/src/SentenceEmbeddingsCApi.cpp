#define _CRT_SECURE_NO_WARNINGS

#include "onnx/experimental_onnxruntime_cxx_api.h"
#include "onnx/coreml_provider_factory.h"

#include "SentenceEmbeddingsCApi.h"

#include <array>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <future>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <cmath>

#define EXPORT __attribute__((visibility("default"))) __attribute__((used))

using namespace std;

namespace embedding_retrieval {

const string TAG = "SentenceEmbeddings";

#ifdef _WIN32
__declspec( dllexport ) void sentence_embeddings_load_model();
#endif

extern "C" {

static Ort::Env* _env;
static Ort::Experimental::Session* _session;
int64_t _embeddingDim = -1;

EXPORT bool sentence_embeddings_load_model(void* const data, size_t length)
{
    if(_session) {
        delete _session;
        _session = nullptr;
    }

    _env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, TAG.c_str());

    Ort::SessionOptions opts;
    opts.DisableMemPattern();
    opts.DisableCpuMemArena();
    
    _session = new Ort::Experimental::Session  {
        *_env,
        (void*)data,
        (size_t)length,
        opts
    };
    std::cout << "Getting output shapes" << std::endl; 
    auto oshapes = _session->GetOutputShapes();
    std::cout << "got shapes" << std::endl; 

    _embeddingDim = oshapes[1][1];
    return true;
}

EXPORT void sentence_embeddings_unload_model() {
    delete _session;
    delete _env;
    _session = nullptr;
    _env = nullptr;
    _embeddingDim = -1;
}

EXPORT int32_t sentence_embeddings_get_model_dimension() {
    return (int32_t) _embeddingDim;
}


EXPORT void sentence_embeddings_free(float* result) {
    free(result);
}

EXPORT void sentence_embeddings_embed(int64_t* tokenIds, int32_t length, float* out) {
    
    auto inames = _session->GetInputNames();
    auto onames = _session->GetOutputNames();

    vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<int64_t>(tokenIds, length, vector<int64_t>{1, (int64_t)length}));

    auto output = _session->Run(inames, input_tensors, onames);

    Ort::Value& outputTensor = output.at(1);
    auto typeInfo = outputTensor.GetTensorTypeAndShapeInfo();
    auto shape = typeInfo.GetShape();

    assert(shape[1] == _embeddingDim);
        
    memcpy(out, outputTensor.GetTensorData<float>(), _embeddingDim * sizeof(float));
}
}
}


int main() {
    return 0;
}


