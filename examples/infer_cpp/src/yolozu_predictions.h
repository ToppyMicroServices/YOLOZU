#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json_escape.h"

struct YolozuDetection {
  int class_id = -1;
  float score = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
};

inline std::vector<YolozuDetection> yolozu_from_combined_xyxy_score_class(
    const std::vector<float>& flat,
    int input_width,
    int input_height,
    const std::string& boxes_scale,
    float min_score,
    int topk) {
  if (flat.size() % 6 != 0) {
    throw std::runtime_error("combined output must have length divisible by 6");
  }
  const size_t rows = flat.size() / 6;
  std::vector<YolozuDetection> dets;
  dets.reserve(rows);

  const bool is_abs = (boxes_scale == "abs");
  if (boxes_scale != "abs" && boxes_scale != "norm") {
    throw std::runtime_error("boxes_scale must be 'abs' or 'norm'");
  }
  if (is_abs && (input_width <= 0 || input_height <= 0)) {
    throw std::runtime_error("input_width/input_height must be set for boxes_scale=abs");
  }

  for (size_t i = 0; i < rows; i++) {
    const float x1 = flat[i * 6 + 0];
    const float y1 = flat[i * 6 + 1];
    const float x2 = flat[i * 6 + 2];
    const float y2 = flat[i * 6 + 3];
    const float score = flat[i * 6 + 4];
    const float class_id_f = flat[i * 6 + 5];

    if (!(score >= min_score)) {
      continue;
    }

    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = (x2 - x1);
    float h = (y2 - y1);

    if (is_abs) {
      cx /= static_cast<float>(input_width);
      cy /= static_cast<float>(input_height);
      w /= static_cast<float>(input_width);
      h /= static_cast<float>(input_height);
    }

    YolozuDetection det;
    det.class_id = static_cast<int>(std::llround(static_cast<double>(class_id_f)));
    det.score = score;
    det.cx = cx;
    det.cy = cy;
    det.w = w;
    det.h = h;
    dets.push_back(det);
  }

  std::stable_sort(dets.begin(), dets.end(), [](const YolozuDetection& a, const YolozuDetection& b) {
    return a.score > b.score;
  });

  if (topk > 0 && static_cast<size_t>(topk) < dets.size()) {
    dets.resize(static_cast<size_t>(topk));
  }

  return dets;
}

inline void yolozu_write_predictions_json(
    const std::string& output_path,
    const std::string& image_path,
    const std::vector<YolozuDetection>& detections,
    const std::string& backend,
    const std::string& model_path,
    bool include_meta = false) {
  std::ofstream file(output_path);
  if (!file) {
    throw std::runtime_error("failed to open predictions output: " + output_path);
  }

  std::ostringstream out;
  out << "{\n";
  out << "  \"predictions\": [\n";
  out << "    {\n";
  out << "      \"image\": \"" << yolozu_json_escape(image_path) << "\",\n";
  out << "      \"detections\": [\n";
  for (size_t i = 0; i < detections.size(); i++) {
    const auto& d = detections[i];
    out << "        {";
    out << "\"class_id\": " << d.class_id << ", ";
    out << "\"score\": " << d.score << ", ";
    out << "\"bbox\": {";
    out << "\"cx\": " << d.cx << ", \"cy\": " << d.cy << ", \"w\": " << d.w << ", \"h\": " << d.h;
    out << "}}";
    if (i + 1 < detections.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "      ]\n";
  out << "    }\n";
  out << "  ]";
  if (include_meta) {
    out << ",\n";
    out << "  \"meta\": {\n";
    out << "    \"backend\": \"" << yolozu_json_escape(backend) << "\",\n";
    out << "    \"model\": \"" << yolozu_json_escape(model_path) << "\"\n";
    out << "  }\n";
  } else {
    out << "\n";
  }
  out << "}\n";

  file << out.str();
  if (!file) {
    throw std::runtime_error("failed to write predictions output: " + output_path);
  }
}
