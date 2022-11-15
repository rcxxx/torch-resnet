#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

static torch::nn::Conv2dOptions
create_conv_options(int64_t in_planes,      int64_t out_planes,     int64_t kernel_size,
                    int64_t stride = 1,     int64_t padding = 0,    int64_t groups = 1,
                    int64_t dilation = 1,   bool bias = false)
{
    torch::nn::Conv2dOptions conv_opt =
            torch::nn::Conv2dOptions (in_planes, out_planes, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .bias(bias)
                    .groups(groups)
                    .dilation(dilation);

    return conv_opt;
}

static torch::nn::Conv2dOptions create_conv3x3_options(int64_t in_planes,
                                                       int64_t out_planes,
                                                       int64_t stride = 1,
                                                       int64_t groups = 1,
                                                       int64_t dilation = 1)
{
    torch::nn::Conv2dOptions conv_opt = create_conv_options(
            in_planes,
            out_planes,
            3, /* kernel_size */
            stride,
            dilation, /* padding */
            groups,
            dilation, /* dilation */
            false);
    return conv_opt;
}

static torch::nn::Conv2dOptions create_conv1x1_options(int64_t in_planes,
                                                       int64_t out_planes,
                                                       int64_t stride = 1)
{
    torch::nn::Conv2dOptions conv_opt = create_conv_options(
            in_planes
            , out_planes
            , 1
            , stride
            , 0 /*padding */
            , 1  /*groups */
            , 1 /*dilation */
            , false);
    return conv_opt;
}

struct BasicBlock : torch::nn::Module {
    BasicBlock(int64_t in_planes, int64_t planes, int64_t stride = 1,
               torch::nn::Sequential down_sample = torch::nn::Sequential(),
               int64_t groups = 1, int64_t base_width = 64,
               int64_t dilation = 1)
    {
        if ((groups != 1) || (base_width != 64))
        {
            throw std::invalid_argument{
                    "BasicBlock only supports groups=1 and base_width=64"};
        }
        if (dilation > 1)
        {
            throw std::invalid_argument{
                    "Dilation > 1 not supported in BasicBlock"};
        }
        m_conv_1 = register_module("conv_1", torch::nn::Conv2d{create_conv3x3_options(in_planes, planes, stride)});
        m_bn_1   = register_module("bn_1", torch::nn::BatchNorm2d{planes});
        m_relu   = register_module("relu", torch::nn::ReLU{true});
        m_conv_2 = register_module("conv_2", torch::nn::Conv2d{create_conv3x3_options(planes, planes)});
        m_bn_2   = register_module("bn_2", torch::nn::BatchNorm2d{planes});
        if (!down_sample->is_empty())
        {
            m_down_sample = register_module("down_sample", down_sample);
        }
        m_stride = stride;
    }

    static const int64_t m_expansion = 1;

    torch::nn::Conv2d       m_conv_1{nullptr};
    torch::nn::Conv2d       m_conv_2{nullptr};
    torch::nn::BatchNorm2d  m_bn_1{nullptr};
    torch::nn::BatchNorm2d  m_bn_2{nullptr};
    torch::nn::ReLU         m_relu{nullptr};
    torch::nn::Sequential   m_down_sample = torch::nn::Sequential();

    int64_t m_stride;

    torch::Tensor forward(const torch::Tensor& x)
    {
        torch::Tensor identity = x;

        torch::Tensor out;
        out = m_conv_1  -> forward(x);
        out = m_bn_1    -> forward(out);
        out = m_relu    -> forward(out);

        out = m_conv_2  -> forward(out);
        out = m_bn_2    -> forward(out);

        if (!m_down_sample->is_empty())
        {
            identity = m_down_sample -> forward(x);
        }

        out += identity;
        out = m_relu    -> forward(out);

        return out;
    }
};

struct Bottleneck : torch::nn::Module
{
    Bottleneck(int64_t in_planes, int64_t planes, int64_t stride = 1,
               torch::nn::Sequential down_sample = torch::nn::Sequential(),
               int64_t groups = 1, int64_t base_width = 64,
               int64_t dilation = 1)
    {
        int64_t width = planes * (base_width / 64) * groups;

        m_conv_1 = register_module("conv_1",torch::nn::Conv2d{create_conv1x1_options(in_planes, width)});
        m_bn_1   = register_module("bn_1",torch::nn::BatchNorm2d{width});
        m_conv_2 = register_module("conv_2",torch::nn::Conv2d{create_conv3x3_options(width, width, stride, groups, dilation)});
        m_bn_2   = register_module("bn_2",torch::nn::BatchNorm2d{width});
        m_conv_3 = register_module("conv_3",torch::nn::Conv2d{create_conv1x1_options(width, planes * m_expansion)});
        m_bn_3   = register_module("bn_3",torch::nn::BatchNorm2d{planes * m_expansion});
        m_relu   = register_module("relu",torch::nn::ReLU{true});
        if (!down_sample->is_empty())
        {
            m_down_sample = register_module("down_sample", down_sample);
        }
        m_stride = stride;
    }

    static const int64_t m_expansion = 4;

    torch::nn::Conv2d       m_conv_1{nullptr};
    torch::nn::Conv2d       m_conv_2{nullptr};
    torch::nn::Conv2d       m_conv_3{nullptr};
    torch::nn::BatchNorm2d  m_bn_1{nullptr};
    torch::nn::BatchNorm2d  m_bn_2{nullptr};
    torch::nn::BatchNorm2d  m_bn_3{nullptr};
    torch::nn::ReLU         m_relu{nullptr};
    torch::nn::Sequential   m_down_sample = torch::nn::Sequential();

    int64_t m_stride;

    torch::Tensor forward(const torch::Tensor& x)
    {
        torch::Tensor identity = x;

        torch::Tensor out;
        out = m_conv_1  -> forward(x);
        out = m_bn_1    -> forward(out);
        out = m_relu    -> forward(out);

        out = m_conv_2  -> forward(out);
        out = m_bn_2    -> forward(out);
        out = m_relu    -> forward(out);

        out = m_conv_3  -> forward(out);
        out = m_bn_3    -> forward(out);

        if (!m_down_sample->is_empty())
        {
            identity = m_down_sample -> forward(x);
        }

        out += identity;
        out = m_relu    -> forward(out);

        return out;
    }
};

template <typename Block>
struct ResNet_Base : torch::nn::Module
{
    explicit ResNet_Base(const std::vector<int64_t>& layers
        , int64_t num_classes = 1000
        , bool zero_init_residual = false
        , int64_t groups = 1
        , int64_t width_per_group = 64
        , std::vector<int64_t> replace_stride_with_dilation = {})
    {
        if (replace_stride_with_dilation.empty())
        {
            replace_stride_with_dilation = {false, false, false};
        }
        if (replace_stride_with_dilation.size() != 3)
        {
            throw std::invalid_argument{
                "replace_stride_with_dilation should be empty or have exactly "
                "three elements."
            };
        }
        m_groups = groups;
        m_base_width = width_per_group;

        m_conv_1 = register_module("conv_1"
                , torch::nn::Conv2d{create_conv_options(3           /*in_planes = */
                                                        , m_in_planes   /*out_planes = */
                                                        , 7             /*kernel_size = */
                                                        , 2                 /*stride = */
                                                        , 3               /*padding = */
                                                        , 1                /*groups = */
                                                        , 1                /*dilation = */
                                                        , false)});           /*bias = */

        m_bn_1 = register_module("bn_1", torch::nn::BatchNorm2d{m_in_planes});
        m_relu = register_module("relu", torch::nn::ReLU{true});
        m_max_pool = register_module("max_pool"
                , torch::nn::MaxPool2d{torch::nn::MaxPool2dOptions({3, 3})
                    .stride({2, 2})
                    .padding({1, 1})});

        m_layer_1 = register_module("layer_1"
                , _makeLayer(64, layers.at(0)));
        m_layer_2 = register_module("layer_2"
                , _makeLayer(128, layers.at(1), 2, replace_stride_with_dilation.at(0)));
        m_layer_3 = register_module("layer_3"
                , _makeLayer(256, layers.at(2), 2, replace_stride_with_dilation.at(1)));
        m_layer_4 = register_module("layer_4"
                , _makeLayer(512, layers.at(3), 2, replace_stride_with_dilation.at(2)));

        m_avg_pool = register_module("avg_pool"
                , torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));

        m_fc = register_module(
                "fc", torch::nn::Linear(512 * Block::m_expansion, num_classes));

        for (const auto& m : modules(false))
        {
            if (m->name() == "torch::nn::Conv2dImpl")
            {
                torch::OrderedDict<std::string, torch::Tensor> named_parameters = m->named_parameters(false);
                torch::Tensor* ptr_w = named_parameters.find("weight");
                torch::nn::init::kaiming_normal_(*ptr_w, 0, torch::kFanOut,
                                                 torch::kReLU);
            }
            else if ((m->name() == "torch::nn::BatchNormImpl") || (m->name() == "torch::nn::GroupNormImpl"))
            {
                torch::OrderedDict<std::string, torch::Tensor> named_parameters;
                named_parameters = m->named_parameters(false);
                torch::Tensor* ptr_w = named_parameters.find("weight");
                torch::nn::init::constant_(*ptr_w, 1.0);
                torch::Tensor* ptr_b = named_parameters.find("bias");
                torch::nn::init::constant_(*ptr_b, 0.0);
            }
        }

        if (zero_init_residual)
        {
            for (const auto& m : modules(false))
            {
                if (m->name() == "Bottleneck")
                {
                    torch::OrderedDict<std::string, torch::Tensor> named_parameters;
                    named_parameters = m->named_modules()["bn3"]->named_parameters(false);
                    torch::Tensor* ptr_w = named_parameters.find("weight");
                    torch::nn::init::constant_(*ptr_w, 0.0);
                }
                else if (m->name() == "BasicBlock")
                {
                    torch::OrderedDict<std::string, torch::Tensor> named_parameters;
                    named_parameters = m->named_modules()["bn2"]->named_parameters(false);
                    torch::Tensor* ptr_w = named_parameters.find("weight");
                    torch::nn::init::constant_(*ptr_w, 0.0);
                }
            }
        }
    }

    int64_t m_in_planes = 64;
    int64_t m_dilation = 1;
    int64_t m_groups = 1;
    int64_t m_base_width = 64;

    torch::nn::Conv2d               m_conv_1{nullptr};
    torch::nn::BatchNorm2d          m_bn_1{nullptr};
    torch::nn::ReLU                 m_relu{nullptr};
    torch::nn::MaxPool2d            m_max_pool{nullptr};
    torch::nn::Sequential           m_layer_1{nullptr};
    torch::nn::Sequential           m_layer_2{nullptr};
    torch::nn::Sequential           m_layer_3{nullptr};
    torch::nn::Sequential           m_layer_4{nullptr};
    torch::nn::AdaptiveAvgPool2d    m_avg_pool{nullptr};
    torch::nn::Linear               m_fc{nullptr};

    torch::nn::Sequential _makeLayer(int64_t planes, int64_t blocks,
                                      int64_t stride = 1, bool dilate = false)
    {
        torch::nn::Sequential down_sample = torch::nn::Sequential();
        int64_t previous_dilation = m_dilation;
        if (dilate)
        {
            m_dilation *= stride;
            stride = 1;
        }
        if ((stride != 1) || (m_in_planes != planes * Block::m_expansion))
        {
            down_sample = torch::nn::Sequential(
                    torch::nn::Conv2d(create_conv1x1_options(m_in_planes
                            , planes * Block::m_expansion
                            , stride))
                    , torch::nn::BatchNorm2d(planes * Block::m_expansion));
        }

        torch::nn::Sequential layers;

        layers->push_back(Block(m_in_planes
                                , planes
                                , stride
                                , down_sample
                                , m_groups
                                , m_base_width
                                , previous_dilation));

        m_in_planes = planes * Block::m_expansion;
        for (int64_t i = 0; i < blocks; i++)
        {
            layers->push_back(Block(m_in_planes
                                    , planes
                                    , 1
                                    , torch::nn::Sequential()
                                    , m_groups
                                    , m_base_width, m_dilation));
        }

        return layers;
    }

    torch::Tensor _forward_impl(torch::Tensor x)
    {

        x = m_conv_1->forward(x);
        x = m_bn_1->forward(x);
        x = m_relu->forward(x);
        x = m_max_pool->forward(x);

        x = m_layer_1->forward(x);
        x = m_layer_2->forward(x);
        x = m_layer_3->forward(x);
        x = m_layer_4->forward(x);

        x = m_avg_pool->forward(x);
        x = torch::flatten(x, 1);
        x = m_fc->forward(x);

        return x;
    }

    torch::Tensor _forward_rm_fc(torch::Tensor x)
    {

        x = m_conv_1->forward(x);
        x = m_bn_1->forward(x);
        x = m_relu->forward(x);
        x = m_max_pool->forward(x);

        x = m_layer_1->forward(x);
        x = m_layer_2->forward(x);
        x = m_layer_3->forward(x);
        x = m_layer_4->forward(x);

        x = m_avg_pool->forward(x);
        x = torch::flatten(x, 1);
        x = torch::nn::functional::normalize(x, torch::nn::functional::NormalizeFuncOptions().dim(1));

        return x;
    }

    torch::Tensor forward(torch::Tensor x) { return _forward_impl(x); }
};

template <class Block>
std::shared_ptr<ResNet_Base<Block>>
resNet_base(const std::vector<int64_t>& layers
        , int64_t num_classes = 1000
        , bool zero_init_residual = false
        , int64_t groups = 1
        , int64_t width_per_group = 64
        , const std::vector<int64_t>& replace_stride_with_dilation = {})
{
    std::shared_ptr<ResNet_Base<Block>> model = std::make_shared<ResNet_Base<Block>>(
            layers
            , num_classes
            , zero_init_residual
            , groups
            , width_per_group
            , replace_stride_with_dilation);

    return model;
}

static std::shared_ptr<ResNet_Base<BasicBlock>>
resNet18(int64_t num_classes = 1000
        , bool zero_init_residual = false
        , int64_t groups = 1
        , int64_t width_per_group = 64
        , const std::vector<int64_t>& replace_stride_with_dilation = {})
{
    const std::vector<int64_t> layers{2, 2, 2, 2};
    std::shared_ptr<ResNet_Base<BasicBlock>> model = resNet_base<BasicBlock>(
            layers
            , num_classes
            , zero_init_residual
            , groups
            , width_per_group
            , replace_stride_with_dilation);

    return model;
}

static std::shared_ptr<ResNet_Base<BasicBlock>>
resNet34(int64_t num_classes = 1000
        , bool zero_init_residual = false
        , int64_t groups = 1
        , int64_t width_per_group = 64
        , const std::vector<int64_t>& replace_stride_with_dilation = {})
{
    const std::vector<int64_t> layers{3, 4, 6, 3};
    std::shared_ptr<ResNet_Base<BasicBlock>> model = resNet_base<BasicBlock>(
            layers
            , num_classes
            , zero_init_residual
            , groups
            , width_per_group
            , replace_stride_with_dilation);

    return model;
}

static std::shared_ptr<ResNet_Base<Bottleneck>>
resNet50(int64_t num_classes = 1000
        , bool zero_init_residual = false
        , int64_t groups = 1
        , int64_t width_per_group = 64
        , const std::vector<int64_t>& replace_stride_with_dilation = {})
{
    const std::vector<int64_t> layers{3, 4, 6, 3};
    std::shared_ptr<ResNet_Base<Bottleneck>> model = resNet_base<Bottleneck>(
            layers
            , num_classes
            , zero_init_residual
            , groups
            , width_per_group
            , replace_stride_with_dilation);

    return model;
}

static std::shared_ptr<ResNet_Base<Bottleneck>>
resnet101(int64_t num_classes = 1000
        , bool zero_init_residual = false
        , int64_t groups = 1
        , int64_t width_per_group = 64
        , const std::vector<int64_t>& replace_stride_with_dilation = {})
{
    const std::vector<int64_t> layers{3, 4, 23, 3};
    std::shared_ptr<ResNet_Base<Bottleneck>> model = resNet_base<Bottleneck>(
            layers
            , num_classes
            , zero_init_residual
            , groups
            , width_per_group
            , replace_stride_with_dilation);

    return model;
}

#endif //MODEL_H
