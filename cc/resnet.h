#ifndef RESNET_H
#define RESNET_H

#include <opencv2/opencv.hpp>

#include "model.h"

namespace resnet {

    class ResNet {
    public:
        explicit ResNet(int64_t num_classes, cv::Size inp_size);

        torch::Tensor calFeature(const cv::Mat& input);

    private:
        std::shared_ptr<ResNet_Base<Bottleneck>> model;

        static cv::Mat imgProcess(const cv::Mat &inp_mat, const cv::Size& r_size);

        torch::Tensor cvMat2Tensor(const cv::Mat &inp_mat);

    private:
        cv::Size _size;
    };

} // resnet

#endif // RESNET_H
