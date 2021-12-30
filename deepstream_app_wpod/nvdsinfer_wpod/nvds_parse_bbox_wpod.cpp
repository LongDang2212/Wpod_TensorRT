#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <vector>
#include <omp.h>
#include <gst/gst.h>

#define INPUT_W 256
#define INPUT_H 256
#define NMS_THRESH 0.5
#define CONF_THRESH 0.4
#define BATCH_SIZE 1
#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Max(a, b) ((a) > (b) ? (a) : (b))

extern "C" bool NvDsInferParseCustomWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

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
    return iou;
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

            if (IOU_Label(lb, x) > iou_threshold)
            {
                non_overlap = false;
                break;
            }
        }
        if (non_overlap)
            v.push_back(lb);
    }
}
void post_process(std::vector<DLabel> &out, float *prob, int h, int w)
{
    float net_stride = pow(2, 4);
    float side = ((208 + 40) / 2) / net_stride;
    float MN[] = {w / net_stride, h / net_stride};
    float b[3][4] = {{-0.5, 0.5, 0.5, -0.5},
                     {-0.5, -0.5, 0.5, 0.5},
                     {1.0, 1.0, 1.0, 1.0}};
    std::vector<std::vector<float>> o;
    // #pragma omp parallel for
    for (int i = 0; i < 256; i++)
    {

        if (prob[i] > CONF_THRESH)
        {
            //  gst_print("\nprob: %f", prob[i]);
            std::vector<float> v;
            v.push_back(i);
            for (int j = 0; j < 8; j++)
            {
                v.push_back(prob[i + j * 256]);
            }
            o.push_back(v);
        }
    }
    std::vector<DLabel> label;
    for (int i = 0; i < o.size(); i++)
    {
        auto v = o.at(i);
        std::vector<float> mn;
        get_pos(v[0], mn);
        float conf = v.at(1);
        float A[2][3] = {{Max(v[3], 0), v[4], v[5]}, {v[6], Max(v[7], 0), v[8]}};
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

        normal(pts, side, mn, MN);
        out.push_back(DLabel(0, pts, conf));
    }
    nms(label, NMS_THRESH, out);
}
bool not_ok(float x)
{
    return (x < 0) || (x > 1);
}
static bool NvDsInferParseWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    std::vector<DLabel> out_labels;
    post_process(out_labels, (float *)outputLayersInfo[0].buffer, INPUT_H, INPUT_W);
    for (auto &l : out_labels)
    {
        if (not_ok(l.tl[0]) || not_ok(l.tl[1]) || not_ok(l.wh[0]) || not_ok(l.wh[1]))
            continue;
        //g_print("\nleft:%f\ttop:%f\twidth:%f\theight:%f", l.tl[0], l.tl[1], l.wh[0], l.wh[1]);
        NvDsInferParseObjectInfo oinfo;
        oinfo.classId = 1;
        oinfo.left = static_cast<unsigned int>(l.tl[0] * 256);
        oinfo.top = static_cast<unsigned int>(l.tl[1] * 256);
        oinfo.width = static_cast<unsigned int>(l.wh[0] * 256);
        oinfo.height = static_cast<unsigned int>(l.wh[1] * 256);
        oinfo.detectionConfidence = l.prob;
        objectList.push_back(oinfo);
        // g_print("\nprob: %f", oinfo.detectionConfidence);
        //g_print("\nleft:%f\ttop:%f\twidth:%f\theight:%f", oinfo.left, oinfo.top, oinfo.width, oinfo.height);
    }

    return true;
}
bool NvDsInferParseCustomWpod(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseWpod(outputLayersInfo, networkInfo, detectionParams, objectList);
}