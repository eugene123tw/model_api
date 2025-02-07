/*
// Copyright (C) 2020-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "models/image_model.h"

namespace ov {
class InferRequest;
class Model;
}  // namespace ov
struct HumanPose;
struct InferenceResult;
struct InputData;
struct InternalModelData;
struct ResultBase;

class HPEOpenPose : public ImageModel {
public:
    /// Constructor
    /// @param modelFile name of model to load
    /// @param aspectRatio - the ratio of input width to its height.
    /// @param targetSize - the height used for model reshaping.
    /// @param confidence_threshold - threshold to eliminate low-confidence keypoints.
    /// @param layout - model input layout
    HPEOpenPose(const std::string& modelFile,
                double aspectRatio,
                int targetSize,
                float confidence_threshold,
                const std::string& layout = "");

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, InferenceInput& input) override;

    static const size_t keypointsNumber = 18;

protected:
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;

    static const int minJointsNumber = 3;
    static const int stride = 8;
    static const int upsampleRatio = 4;
    static const cv::Vec3f meanPixel;
    static const float minPeaksDistance;
    static const float midPointsScoreThreshold;
    static const float foundMidPointsRatioThreshold;
    static const float minSubsetScore;
    cv::Size inputLayerSize;
    double aspectRatio;
    int targetSize;
    float confidence_threshold;

    std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps, const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;

    void changeInputSize(std::shared_ptr<ov::Model>& model);
};
