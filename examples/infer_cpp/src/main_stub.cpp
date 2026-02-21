#include <iostream>
#include <string>
#include <vector>

#include "yolozu_predictions.h"

static void print_usage() {
  std::cerr << "yolozu_infer_stub --image <path> --output <predictions.json>\n";
  std::cerr << "\n";
  std::cerr << "Emits schema-correct YOLOZU predictions JSON with empty detections.\n";
}

int main(int argc, char** argv) {
  std::string image_path;
  std::string output_path;

  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }
    if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
      continue;
    }
    if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
      continue;
    }
    std::cerr << "unknown arg: " << arg << "\n";
    print_usage();
    return 2;
  }

  if (image_path.empty() || output_path.empty()) {
    print_usage();
    return 2;
  }

  std::vector<YolozuDetection> dets;
  yolozu_write_predictions_json(output_path, image_path, dets, "stub", "", false);
  std::cout << output_path << "\n";
  return 0;
}
