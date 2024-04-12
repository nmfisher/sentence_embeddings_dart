// Copyright (c) 2023, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:io';

import 'package:logging/logging.dart';
import 'package:native_assets_cli/native_assets_cli.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

void main(List<String> args) async {
  await build(args, (config, output) async {
    var onnxdir = File(config.packageRoot.path).parent.path + "/flutter_onnx";
    final packageName = config.packageName;

    final cbuilder = CBuilder.library(
      name: packageName,
      language: Language.cpp,
      assetName: '$packageName.dart',
      sources: [
        'native/src/SentenceEmbeddingsCApi.cpp',
      ],
      includes: ['native/include', '../flutter_onnx/ios/include'],
      flags: [
        '-std=c++17',
        // "-F$onnxdir/macos/lib",
        "-F$onnxdir/ios/lib",
        '-framework',
        'onnxruntime',
        '-framework',
        'Foundation',
      ],
      dartBuildFiles: ['hook/build.dart'],
    );
    await cbuilder.run(
      buildConfig: config,
      buildOutput: output,
      logger: Logger('')
        ..level = Level.ALL
        ..onRecord.listen((record) => print(record.message)),
    );
  });
}
