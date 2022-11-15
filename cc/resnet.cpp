#include "resnet.h"

namespace resnet {
    ResNet::ResNet(int64_t num_classes, cv::Size inp_size){
        this->model = resNet50(num_classes);
        model->eval();
        this->_size = inp_size;
    };

    cv::Mat ResNet::imgProcess(const cv::Mat &inp_mat, const cv::Size& r_size) {
        cv::Mat dst_mat;// = inp_mat(cv::Rect(inp_mat.cols/3, inp_mat.rows/3, inp_mat.cols/3, inp_mat.rows/3));
        cv::cvtColor(inp_mat, dst_mat, cv::COLOR_BGR2RGB);
        cv::resize(dst_mat, dst_mat , r_size);

        //normalization
        dst_mat.convertTo(dst_mat, CV_64FC3, 1.0f / 255.0f);
        cv::Mat mean = cv::Mat(dst_mat.size(), dst_mat.type(), cv::Scalar_<double>(0.485, 0.456, 0.406));
        dst_mat -= mean;
        cv::Mat std = cv::Mat(dst_mat.size(), dst_mat.type(), cv::Scalar_<double>(0.229, 0.224, 0.225));
        dst_mat /= std;

        return dst_mat;
    }

    torch::Tensor ResNet::cvMat2Tensor(const cv::Mat &inp_mat){
        cv::Mat inp_rgb = imgProcess(inp_mat, this->_size);

        //opencv format H*W*C
        auto out_tensor = torch::from_blob(inp_rgb.data, {1, inp_rgb.rows, inp_rgb.cols, 3}, torch::kByte);
        //pytorch format N*C*H*W
        out_tensor = out_tensor.permute({0, 3, 1, 2});

        return out_tensor.toType(torch::kFloat);
    }

    torch::Tensor ResNet::calFeature(const cv::Mat& input) {
        torch::Tensor x = this->cvMat2Tensor(input);
        return model->forward(x);
    }

} // resnet