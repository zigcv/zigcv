#!/usr/bin/env -S deno run -A
import { $ } from "jsr:@david/dax@0.41.0";

$.setPrintCommand(true);

const MODEL_URLS = [
  "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
  "https://raw.githubusercontent.com/opencv/opencv_extra/20d18acad1bcb312045ea64a239ebe68c8728b88/testdata/dnn/bvlc_googlenet.prototxt",
  "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
  "https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx",
  "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
  "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
] as const;

/** get the root of the git repository */
const git_root = await $`git rev-parse --show-toplevel`.text();

/** save the models in the zig-cache/tmp/ directory */
const saveDir = $.path(git_root).join("./zig-cache/tmp/");

/** create the directory if it does not exist */
await $`mkdir -p ${saveDir.toString()}`;

$.cd(saveDir);

/** list the files in the directory */
await $`ls -lh ${saveDir}`;

/** download the models */
const downloadedPath = await Promise.all(
  MODEL_URLS.map((url) => $.request(url).showProgress().pipeToPath()),
);

/** decompress the zip files */
await Promise.all(
  downloadedPath
    .filter((path) => path.extname() === ".zip")
    .map((path) => $`unzip -o ${path}`),
);

/** list the files in the directory */
await $`ls -lh ${saveDir}`;
