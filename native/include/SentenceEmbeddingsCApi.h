#ifndef _SENTENCE_EMBEDDINGS_C_API
#define _SENTENCE_EMBEDDINGS_C_API

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>

#define EXPORT __attribute__((visibility("default"))) __attribute__((used))

#ifdef __cplusplus
extern "C" {
#endif

bool sentence_embeddings_load_model(void* const data, size_t length);
void sentence_embeddings_unload_model();

int32_t sentence_embeddings_get_model_dimension();
void sentence_embeddings_embed(int64_t* tokens, int32_t length, float* out);
EXPORT void sentence_embeddings_free(float* result);

#ifdef __cplusplus
}
#endif
#endif
