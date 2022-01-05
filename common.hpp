#ifndef _WPOD_COMMON_HPP_
#define _WPOD_COMMON_HPP_

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include "NvInfer.h"
#include <chrono>
#include <algorithm>
#include "singular/singular.h"
#include "singular/Svd.h"
#define NMS_THRESH 0.5
#define CONF_THRESH 0.95
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Max(a, b) ((a) > (b) ? (a) : (b))

using namespace nvinfer1;

void get_pos(int x, std::vector<float> &v)
{

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            if (i * 16 + j == x)
            {
                v.push_back(j + 0.5);
                v.push_back(i + 0.5);
                return;
            }
        }
    }
}

float iou(float lbox[], float lbox2[], float rbox[], float rbox2[])
{
    assert(lbox[0] < lbox2[0]);
    assert(lbox[1] < lbox2[1]);
    assert(rbox[0] < rbox2[0]);
    assert(rbox[1] < rbox2[1]);

    float x_left = Max(lbox[0], rbox[0]);
    float x_right = Min(lbox2[0], rbox2[0]);
    float y_top = Max(lbox[1], rbox[1]);
    float y_bottom = Min(lbox2[1], rbox2[1]);
    if ((x_right < x_left) || (y_bottom < y_top))
        return 0.0;
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float bb1_area = (lbox2[0] - lbox[0]) * (lbox2[1] - lbox[1]);
    float bb2_area = (rbox2[0] - rbox[0]) * (rbox2[1] - rbox[1]);
    float iou = intersection_area / float(bb1_area + bb2_area - intersection_area);
    assert(iou <= 1.0);
    assert(iou >= 0);
    return iou;
}

void find_T_matrix(float pts[][4], float t_pts[][4], std::vector<std::vector<float>> &r)
{
    double temp[8][9];
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            temp[i][j] = 0;
        }
    }
    for (int i = 0; i < 4; i++)
    {
        float xi[3], xil[3];
        for (int j = 0; j < 3; j++)
        {
            xi[j] = t_pts[j][i];
            xil[j] = pts[j][i];
        }
        // xi = xi.T
        for (int j = 3; j < 6; j++)
        {
            temp[i * 2][j] = -xil[2] * xi[j - 3];
        }
        for (int j = 6; j < 9; j++)
        {
            temp[i * 2][j] = xil[1] * xi[j - 6];
        }
        for (int j = 0; j < 3; j++)
        {
            temp[i * 2 + 1][j] = xil[2] * xi[j];
        }
        for (int j = 6; j < 9; j++)
        {
            temp[i * 2 + 1][j] = -xil[0] * xi[j - 6];
        }
    }
    // check done
    double *mat = new double[sizeof(temp) * sizeof(double) / sizeof(float)];
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            mat[i * 9 + j] = temp[i][j];
        }
    }
    singular::Matrix<8, 9> A;
    A.fill(mat);
    // check done
    singular::Svd<8, 9>::USV usv = singular::Svd<8, 9>::decomposeUSV(A);
    auto V1 = singular::Svd<8, 9>::getV(usv).clone();
    auto V2 = V1.transpose();
    auto c = V2.row(8);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            r[i][j] = c[i * 3 + j];
        }
    }
}

void normal(float pts[][4], float side, std::vector<float> mn, float MN[])
{

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            pts[i][j] *= side;
        }
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[0][j] += mn[0];
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[1][j] += mn[1];
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[0][j] /= MN[0];
    }
    for (size_t j = 0; j < 4; j++)
    {
        pts[1][j] /= MN[1];
    }
}
void getRectPts(float r[][4], float tlx, float tly, float brx, float bry)
{
    r[0][0] = tlx;
    r[0][1] = brx;
    r[0][2] = brx;
    r[0][3] = tlx;
    r[1][0] = tly;
    r[1][1] = tly;
    r[1][2] = bry;
    r[1][3] = bry;
    r[2][0] = 1;
    r[2][1] = 1;
    r[2][2] = 1;
    r[2][3] = 1;
}
class Label
{
public:
    float tl[2];
    float br[2];
    float prob;
    float cl;
    float wh[2];

public:
    Label() {}
    Label(float t[], float b[], float c, float p)
    {
        tl[0] = t[0];
        tl[1] = t[1];
        br[0] = b[0];
        br[1] = b[1];
        cl = c;
        prob = p;
        wh[0] = abs(br[0] - tl[0]);
        wh[1] = abs(br[1] - tl[1]);
    }
    Label(const Label &l)
    {
        tl[0] = l.tl[0];
        tl[1] = l.tl[1];
        br[0] = l.br[0];
        br[1] = l.br[1];
        cl = l.cl;
        prob = l.prob;
    }
    void cc(float *t)
    {
        t[0] = tl[0] + wh[0] / 2;
        t[1] = tl[1] + wh[1] / 2;
    }
    void tr(float *t)
    {
        t[0] = br[0];
        t[1] = tl[1];
    }
    void bl(float *t)
    {
        t[0] = tl[0];
        t[1] = br[1];
    }
    float area()
    {
        return wh[0] * wh[1];
    }
};
class DLabel : public Label
{
public:
    float pts[2][4];

public:
    DLabel() {}
    DLabel(float cl, float pts[][4], float p)
    {

        tl[0] = *std::min_element(&pts[0][0], &pts[0][4]);
        tl[1] = *std::min_element(&pts[1][0], &pts[1][4]);

        br[0] = *std::max_element(&pts[0][0], &pts[0][4]);
        br[1] = *std::max_element(&pts[1][0], &pts[1][4]);
        this->cl = cl;
        this->prob = p;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                this->pts[i][j] = pts[i][j];
            }
        }
        wh[0] = abs(br[0] - tl[0]);
        wh[1] = abs(br[1] - tl[1]);
    }
    DLabel(const DLabel &l)
    {
        tl[0] = l.tl[0];
        tl[1] = l.tl[1];
        br[0] = l.br[0];
        br[1] = l.br[1];
        cl = l.cl;
        prob = l.prob;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                this->pts[i][j] = l.pts[i][j];
            }
        }
        wh[0] = abs(br[0] - tl[0]);
        wh[1] = abs(br[1] - tl[1]);
    }
};
float IOU_Label(DLabel l1, DLabel l2)
{
    return iou(l1.tl, l1.br, l2.tl, l2.br);
}
bool comp(DLabel l1, DLabel l2)
{
    return l1.prob < l2.prob;
}
void nms(std::vector<DLabel> l, float iou_threshold, std::vector<DLabel> &v)
{
    std::sort(l.begin(), l.end(), comp);
    std::reverse(l.begin(), l.end());

    for (auto lb : l)
    {
        bool non_overlap = true;
        for (auto x : v)
        {

            if (IOU_Label(lb, x) > 0.1)
            {
                non_overlap = false;
                break;
            }
        }
        if (non_overlap)
            v.push_back(lb);
    }
}
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}
IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
    float *gamma = (float *)weightMap[lname + "/gamma:0"].values;
    float *beta = (float *)weightMap[lname + "/beta:0"].values;
    float *mean = (float *)weightMap[lname + "/moving_mean:0"].values;
    float *var = (float *)weightMap[lname + "/moving_variance:0"].values;
    int len = weightMap[lname + "/moving_variance:0"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}
IActivationLayer *addResBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor *input, int channels, int &num)
{
    IConvolutionLayer *conv1 = network->addConvolutionNd(*input, channels, DimsHW{3, 3}, weightMap["conv2d_" + std::to_string(num) + "/kernel:0"], weightMap["conv2d_" + std::to_string(num) + "/bias:0"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "batch_normalization_" + std::to_string(num++), 0.001);
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), channels, DimsHW{3, 3}, weightMap["conv2d_" + std::to_string(num) + "/kernel:0"], weightMap["conv2d_" + std::to_string(num) + "/bias:0"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});
    auto bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "batch_normalization_" + std::to_string(num++), 0.001);
    IElementWiseLayer *sum = network->addElementWise(*bn2->getOutput(0), *input, ElementWiseOperation::kSUM);
    IActivationLayer *relu2 = network->addActivation(*sum->getOutput(0), ActivationType::kRELU);
    return relu2;
}
IActivationLayer *addConvBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor *input, int channels, int &num)
{
    IConvolutionLayer *conv = network->addConvolutionNd(*input, channels, DimsHW{3, 3}, weightMap["conv2d_" + std::to_string(num) + "/kernel:0"], weightMap["conv2d_" + std::to_string(num) + "/bias:0"]);
    assert(conv);
    conv->setPaddingNd(DimsHW{1, 1});
    conv->setStrideNd(DimsHW{1, 1});
    auto bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), "batch_normalization_" + std::to_string(num++), 0.001);
    IActivationLayer *relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    return relu;
}



void post_process(cv::Mat image, std::vector<cv::Mat> &out_img, float *prob, int h, int w)
{
    float net_stride = pow(2, 4);
    float side = ((208 + 40) / 2) / net_stride;
    float one_line[] = {470, 110};
    float two_line[] = {280, 200};
    float MN[] = {w / net_stride, h / net_stride};
    float alpha = 0.5;
    float b[3][4] = {{-0.5, 0.5, 0.5, -0.5},
                     {-0.5, -0.5, 0.5, 0.5},
                     {1.0, 1.0, 1.0, 1.0}};
    std::vector<std::vector<float>> o;
    for (int i = 0; i < 256; i++)
    {
        if (prob[i] > CONF_THRESH)
        {
            std::cout << prob[i] << "\t";
            std::vector<float> v;
            v.push_back(i);
            for (int j = 0; j < 8; j++)
            {
                v.push_back(prob[i + j * 256]);
            }
            o.push_back(v);
        }
    }
    if (o.empty())
    {
        std::cout << "\nNo licence plate found!\n";
        return;
    }
    std::vector<DLabel> label;
    std::vector<DLabel> label_frontal;
    for (int i = 0; i < o.size(); i++)
    {
        auto v = o.at(i);
        std::vector<float> mn;
        get_pos(v[0], mn);
        float conf = v.at(1);
        float A[2][3] = {{Max(v[3], 0), v[4], v[5]}, {v[6], Max(v[7], 0), v[8]}};
        float B[2][3] = {{Max(v[3], 0), 0, 0}, {0, Max(v[7], 0), 0}};
        float pts[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    pts[i][j] += A[i][k] * b[k][j];
                }
            }
        }

        float pts_frontal[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    pts_frontal[i][j] += B[i][k] * b[k][j];
                }
            }
        }

        // check ok
        normal(pts, side, mn, MN);
        normal(pts_frontal, side, mn, MN);
        label.push_back(DLabel(0, pts, conf));
        label_frontal.push_back(DLabel(0, pts_frontal, conf));
    }
    std::vector<DLabel> final_label;

    nms(label, NMS_THRESH, final_label);
    for (auto l : final_label)
    {
        std::cout << "\n"
                  << l.tl[0] << "\t" << l.tl[1];
        cv::Rect crop(l.tl[0] * 256, l.tl[1] * 256, l.wh[0] * 256, l.wh[1] * 256);
        auto img = image.clone();
        cv::rectangle(img, crop, cv::Scalar(0, 255, 255), 2);
        // cv::imwrite("output/d.jpg", img);
    }
    std::vector<DLabel> final_label_frontal;
    nms(label_frontal, NMS_THRESH, final_label_frontal);

    assert(!final_label_frontal.empty());
    int type = 0;
    cv::Size out_size;
    if (final_label_frontal[0].wh[0] / final_label_frontal[1].wh[1] < 1.7)
    {
        type = 1;
        out_size.width = 280;
        out_size.height = 200;
    }
    else
    {
        type = 2;
        out_size.width = 470;
        out_size.height = 110;
    }
    if (!final_label.empty())
    {
        std::sort(final_label.begin(), final_label.end(), comp);
        std::reverse(final_label.begin(), final_label.end());
        for (auto l : final_label)
        {
            float t_pts[3][4];
            getRectPts(t_pts, 0, 0, out_size.width, out_size.height);

            float ptsh[3][4];

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    l.pts[i][j] *= 256.0;
                    ptsh[i][j] = l.pts[i][j];
                }
            }
            // for (int i = 0; i < 2; i++)
            // {
            //     for (int j = 0; j < 4; j++)
            //     {
            //         ptsh[i][j] *= 256.0;
            //     }
            // }
            for (int i = 0; i < 4; i++)
            {
                ptsh[2][i] = 1.0;
            }

            //check done
            std::vector<std::vector<float>> H = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

            find_T_matrix(t_pts, ptsh, H);
            // check done
            float Hf[3][3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Hf[i][j] = H[i][j];
                }
            }
            cv::Mat H1(3, 3, CV_32F);
            std::memcpy(H1.data, Hf, 3 * 3 * sizeof(float));
            // check done
            cv::Mat tmp_img;
            cv::warpPerspective(image, tmp_img, H1, out_size, 1, 0);
            out_img.push_back(tmp_img.clone());
        }
    }
}
#endif