import 'dart:io';

import 'package:sentence_embeddings_dart/sentence_embeddings_dart.dart';
import 'package:sentencepiece_dart_bindings/sentencepiece_dart_bindings.dart';

void main() {
  var scriptDir = File(Platform.script.path).parent.path;
  var tokenizerModel = File("$scriptDir/sentencepiece.bpe.model");
  var tokenizerModelData = tokenizerModel.readAsBytesSync();

  var embeddingsModel = File("$scriptDir/minilm.quant.ort");
  var embeddingsModelData = embeddingsModel.readAsBytesSync();
  var sentenceEmbedddings =
      SentenceEmbeddings(tokenizerModelData, embeddingsModelData);
  for (int i = 0; i < 100; i++) {
    print(i);
    var embeddings = sentenceEmbedddings.embed("Hello world");
  }
  // print(embeddings);
}
