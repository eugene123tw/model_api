/*
// Copyright (C) 2018-2023 Intel Corporation
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

#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/instance_segmentation.h>
#include <models/input_data.h>
#include <models/results.h>

#include <chrono>


int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <path_to_model> <path_to_image>");
    }

    std::chrono::time_point<std::chrono::system_clock> start, end;

    std::vector<cv::String> imagePaths;
    cv::String folderPath = argv[2];
    folderPath.append("/*.jpg");
    cv::glob(folderPath, imagePaths, false);

    std::vector<cv::Mat> cachedImages;
    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }
        cachedImages.push_back(image);
    }

    // Instantiate MaskRCNN model
    auto model = MaskRCNNModel::create_model(argv[1]);

    start = std::chrono::system_clock::now();
    // Run the inference
    for (size_t i = 0; i < cachedImages.size(); i++){
        auto result = model->infer(cachedImages[i]);
    }
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "FPS: " << cachedImages.size() / elapsed_seconds.count() << "s\n";

    // Process detections
    // for (auto& obj : result->segmentedObjects) {
    //     std::cout << " " << std::left << std::setw(9) << obj.label << " | " << std::setw(10) << obj.confidence
    //         << " | " << std::setw(4) << int(obj.x) << " | " << std::setw(4) << int(obj.y) << " | "
    //         << std::setw(4) << int(obj.x + obj.width) << " | " << std::setw(4) << int(obj.y + obj.height) << "\n";
    // }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
