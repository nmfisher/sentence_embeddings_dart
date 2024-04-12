import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:sentence_embeddings_dart/src/sentence_embeddings_dart.g.dart';
import 'package:sentencepiece_dart_bindings/sentencepiece_dart_bindings.dart';

class SentenceEmbeddings {
  late final SentencePieceTokenizer tokenizer;

  late final int _embeddingDim;

  SentenceEmbeddings(
    Uint8List tokenizerModel,
    Uint8List embeddingModel,
  ) {
    tokenizer = SentencePieceTokenizer(tokenizerModel);
    final modelDataPtr = calloc<Char>(embeddingModel.length);
    for (int i = 0; i < embeddingModel.length; i++) {
      modelDataPtr[i] = embeddingModel[i];
    }
    sentence_embeddings_load_model(
        modelDataPtr.cast<Void>(), embeddingModel.length);
    _embeddingDim = sentence_embeddings_get_model_dimension();
    calloc.free(modelDataPtr);
    print("Loaded model with embedding dimension $_embeddingDim");
  }

  void dispose() {
    sentence_embeddings_unload_model();
  }

  List<double> embed(String sentence) {
    var tokens = tokenizer.tokenize(sentence);
    final ptr = calloc<Int64>(tokens.length);
    for (int i = 0; i < tokens.length; i++) {
      ptr[i] = tokens[i];
    }

    final out = calloc<Float>(_embeddingDim);
    sentence_embeddings_embed(ptr, tokens.length, out);
    calloc.free(ptr);

    var embeddings = List<double>.from(out.asTypedList(_embeddingDim));
    sentence_embeddings_free(out);
    return embeddings;
  }
}
