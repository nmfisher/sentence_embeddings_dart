import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:sentence_embeddings_dart/src/sentence_embeddings_dart.g.dart';
import 'package:sentencepiece_dart_bindings/sentencepiece_dart_bindings.dart';

class SentenceEmbeddings {
  final int increment;
  final int? bos;
  final int? eos;
  late final SentencePieceTokenizer tokenizer;

  late final int _embeddingDim;
  int get embeddingDim => _embeddingDim;

  SentenceEmbeddings(
    Uint8List tokenizerModel,
    Uint8List embeddingModel, {
    this.bos = 0,
    this.eos = 2,
    this.increment = 1,
  }) {
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

  (Pointer<Float>, List<int>) embedAsPointer(String sentence) {
    final out = calloc<Float>(_embeddingDim);

    var tokens = tokenizer.tokenize(sentence);

    if (increment > 0) {
      tokens = tokens.map((t) => t + increment).toList();
    }
    if (bos != null) {
      tokens = [bos!] + tokens;
    }
    if (eos != null) {
      tokens = tokens + [eos!];
    }

    final ptr = calloc<Int64>(tokens.length);
    for (int i = 0; i < tokens.length; i++) {
      ptr[i] = tokens[i];
    }

    sentence_embeddings_embed(ptr, tokens.length, out);
    calloc.free(ptr);
    return (out, tokens);
  }

  List<double> embed(String sentence) {
    final (out, tokens) = embedAsPointer(sentence);
    var embeddings = List<double>.from(out.asTypedList(_embeddingDim));
    return embeddings;
    calloc.free(out);
  }

  void free(Pointer<Float> ptr) {
    calloc.free(ptr);
  }
}
