#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <math.h>
#define USE_FP16 // comment out this if want to use FP32
#define DEVICE 0 // GPU id

static const int SHORT_INPUT = 256;
static const int MAX_INPUT_SIZE = 1440; // 32x
static const int MIN_INPUT_SIZE = 256;
static const int OPT_INPUT_W = 256;
static const int OPT_INPUT_H = 256;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "out";
const char *WEIGHT_PATH = "/opt/nvidia/deepstream/deepstream-5.0/sources/alpr_ds/wpod/build/wpod.wts";
static Logger gLogger;

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, 256, 256});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(WEIGHT_PATH);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    /* ------------------- Wpod backbone ------------------ */

    // conv + bn + relu 3x3 16
    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 16, DimsHW{3, 3}, weightMap["conv2d_1/kernel:0"], weightMap["conv2d_1/bias:0"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    // conv1->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv1->setPaddingNd(DimsHW{1, 1});
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "batch_normalization_1", 0.001);
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    // conv + bn + relu 3x3 16
    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), 16, DimsHW{3, 3}, weightMap["conv2d_2/kernel:0"], weightMap["conv2d_2/bias:0"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    // conv2->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv2->setPaddingNd(DimsHW{1, 1});
    auto bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "batch_normalization_2", 0.001);
    IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    // max pooling 1
    IPoolingLayer *maxPool1 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    maxPool1->setStrideNd(DimsHW{2, 2});

    // conv + bn + relu 3x3 32
    IConvolutionLayer *conv3 = network->addConvolutionNd(*maxPool1->getOutput(0), 32, DimsHW{3, 3}, weightMap["conv2d_3/kernel:0"], weightMap["conv2d_3/bias:0"]);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    // conv3->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv3->setPaddingNd(DimsHW{1, 1});
    auto bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "batch_normalization_3", 0.001);
    IActivationLayer *relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);

    // resblock 32
    IConvolutionLayer *conv4 = network->addConvolutionNd(*relu3->getOutput(0), 32, DimsHW{3, 3}, weightMap["conv2d_4/kernel:0"], weightMap["conv2d_4/bias:0"]);
    assert(conv4);
    conv4->setStrideNd(DimsHW{1, 1});
    // conv4->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv4->setPaddingNd(DimsHW{1, 1});
    auto bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), "batch_normalization_4", 0.001);
    IActivationLayer *relu4 = network->addActivation(*bn4->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv5 = network->addConvolutionNd(*relu4->getOutput(0), 32, DimsHW{3, 3}, weightMap["conv2d_5/kernel:0"], weightMap["conv2d_5/bias:0"]);
    assert(conv5);
    conv5->setStrideNd(DimsHW{1, 1});
    // conv5->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv5->setPaddingNd(DimsHW{1, 1});
    auto bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), "batch_normalization_5", 0.001);
    IElementWiseLayer *sum1 = network->addElementWise(*bn5->getOutput(0), *relu3->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu5 = network->addActivation(*sum1->getOutput(0), ActivationType::kRELU);

    // max pooling 2
    IPoolingLayer *maxPool2 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    maxPool2->setStrideNd(DimsHW{2, 2});

    // conv + bn + relu 3x3 64
    IConvolutionLayer *conv6 = network->addConvolutionNd(*maxPool2->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_6/kernel:0"], weightMap["conv2d_6/bias:0"]);
    assert(conv6);
    conv6->setStrideNd(DimsHW{1, 1});
    // conv6->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv6->setPaddingNd(DimsHW{1, 1});
    auto bn6 = addBatchNorm2d(network, weightMap, *conv6->getOutput(0), "batch_normalization_6", 0.001);
    IActivationLayer *relu6 = network->addActivation(*bn6->getOutput(0), ActivationType::kRELU);

    // resblock 64
    IConvolutionLayer *conv7 = network->addConvolutionNd(*relu6->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_7/kernel:0"], weightMap["conv2d_7/bias:0"]);
    assert(conv7);
    // conv7->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv7->setPaddingNd(DimsHW{1, 1});
    conv7->setStrideNd(DimsHW{1, 1});
    auto bn7 = addBatchNorm2d(network, weightMap, *conv7->getOutput(0), "batch_normalization_7", 0.001);
    IActivationLayer *relu7 = network->addActivation(*bn7->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv8 = network->addConvolutionNd(*relu7->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_8/kernel:0"], weightMap["conv2d_8/bias:0"]);
    assert(conv8);
    // conv8->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv8->setPaddingNd(DimsHW{1, 1});
    conv8->setStrideNd(DimsHW{1, 1});
    auto bn8 = addBatchNorm2d(network, weightMap, *conv8->getOutput(0), "batch_normalization_8", 0.001);
    IElementWiseLayer *sum2 = network->addElementWise(*bn8->getOutput(0), *relu6->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu8 = network->addActivation(*sum2->getOutput(0), ActivationType::kRELU);

    // resblock 64
    IConvolutionLayer *conv9 = network->addConvolutionNd(*relu8->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_9/kernel:0"], weightMap["conv2d_9/bias:0"]);
    assert(conv9);
    conv9->setStrideNd(DimsHW{1, 1});
    // conv9->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv9->setPaddingNd(DimsHW{1, 1});
    auto bn9 = addBatchNorm2d(network, weightMap, *conv9->getOutput(0), "batch_normalization_9", 0.001);
    IActivationLayer *relu9 = network->addActivation(*bn9->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv10 = network->addConvolutionNd(*relu9->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_10/kernel:0"], weightMap["conv2d_10/bias:0"]);
    assert(conv10);
    conv10->setStrideNd(DimsHW{1, 1});
    // conv10->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv10->setPaddingNd(DimsHW{1, 1});
    auto bn10 = addBatchNorm2d(network, weightMap, *conv10->getOutput(0), "batch_normalization_10", 0.001);
    IElementWiseLayer *sum3 = network->addElementWise(*bn10->getOutput(0), *relu8->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu10 = network->addActivation(*sum3->getOutput(0), ActivationType::kRELU);

    // max pooling 3
    IPoolingLayer *maxPool3 = network->addPoolingNd(*relu10->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    maxPool3->setStrideNd(DimsHW{2, 2});

    // conv + bn + relu 3x3 64
    IConvolutionLayer *conv11 = network->addConvolutionNd(*maxPool3->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_11/kernel:0"], weightMap["conv2d_11/bias:0"]);
    assert(conv11);
    // conv11->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv11->setPaddingNd(DimsHW{1, 1});
    conv11->setStrideNd(DimsHW{1, 1});
    auto bn11 = addBatchNorm2d(network, weightMap, *conv11->getOutput(0), "batch_normalization_11", 0.001);
    IActivationLayer *relu11 = network->addActivation(*bn11->getOutput(0), ActivationType::kRELU);

    // resblock 64
    IConvolutionLayer *conv12 = network->addConvolutionNd(*relu11->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_12/kernel:0"], weightMap["conv2d_12/bias:0"]);
    assert(conv12);
    conv12->setStrideNd(DimsHW{1, 1});
    // conv12->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv12->setPaddingNd(DimsHW{1, 1});
    auto bn12 = addBatchNorm2d(network, weightMap, *conv12->getOutput(0), "batch_normalization_12", 0.001);
    IActivationLayer *relu12 = network->addActivation(*bn12->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv13 = network->addConvolutionNd(*relu12->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_13/kernel:0"], weightMap["conv2d_13/bias:0"]);
    assert(conv13);
    conv13->setStrideNd(DimsHW{1, 1});
    // conv13->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv13->setPaddingNd(DimsHW{1, 1});
    auto bn13 = addBatchNorm2d(network, weightMap, *conv13->getOutput(0), "batch_normalization_13", 0.001);
    IElementWiseLayer *sum4 = network->addElementWise(*relu11->getOutput(0), *bn13->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu13 = network->addActivation(*sum4->getOutput(0), ActivationType::kRELU);

    // resblock 64
    IConvolutionLayer *conv14 = network->addConvolutionNd(*relu13->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_14/kernel:0"], weightMap["conv2d_14/bias:0"]);
    assert(conv14);
    conv14->setStrideNd(DimsHW{1, 1});
    // conv14->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv14->setPaddingNd(DimsHW{1, 1});
    auto bn14 = addBatchNorm2d(network, weightMap, *conv14->getOutput(0), "batch_normalization_14", 0.001);
    IActivationLayer *relu14 = network->addActivation(*bn14->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv15 = network->addConvolutionNd(*relu14->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2d_15/kernel:0"], weightMap["conv2d_15/bias:0"]);
    assert(conv15);
    conv15->setStrideNd(DimsHW{1, 1});
    // conv15->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv15->setPaddingNd(DimsHW{1, 1});
    auto bn15 = addBatchNorm2d(network, weightMap, *conv15->getOutput(0), "batch_normalization_15", 0.001);
    IElementWiseLayer *sum5 = network->addElementWise(*bn15->getOutput(0), *relu13->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu15 = network->addActivation(*sum5->getOutput(0), ActivationType::kRELU);

    // max pooling 4
    IPoolingLayer *maxPool4 = network->addPoolingNd(*relu15->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    maxPool4->setStrideNd(DimsHW{2, 2});

    // conv + bn + relu 3x3 128
    IConvolutionLayer *conv16 = network->addConvolutionNd(*maxPool4->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_16/kernel:0"], weightMap["conv2d_16/bias:0"]);
    assert(conv16);
    // conv16->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv16->setPaddingNd(DimsHW{1, 1});
    conv16->setStrideNd(DimsHW{1, 1});
    auto bn16 = addBatchNorm2d(network, weightMap, *conv16->getOutput(0), "batch_normalization_16", 0.001);
    IActivationLayer *relu16 = network->addActivation(*bn16->getOutput(0), ActivationType::kRELU);

    // resblock 128
    IConvolutionLayer *conv17 = network->addConvolutionNd(*relu16->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_17/kernel:0"], weightMap["conv2d_17/bias:0"]);
    assert(conv17);
    // conv17->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv17->setPaddingNd(DimsHW{1, 1});
    conv17->setStrideNd(DimsHW{1, 1});
    auto bn17 = addBatchNorm2d(network, weightMap, *conv17->getOutput(0), "batch_normalization_17", 0.001);
    IActivationLayer *relu17 = network->addActivation(*bn17->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv18 = network->addConvolutionNd(*relu17->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_18/kernel:0"], weightMap["conv2d_18/bias:0"]);
    assert(conv18);
    conv18->setStrideNd(DimsHW{1, 1});
    // conv18->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv18->setPaddingNd(DimsHW{1, 1});
    auto bn18 = addBatchNorm2d(network, weightMap, *conv18->getOutput(0), "batch_normalization_18", 0.001);
    IElementWiseLayer *sum6 = network->addElementWise(*bn18->getOutput(0), *relu16->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu18 = network->addActivation(*sum6->getOutput(0), ActivationType::kRELU);

    // resblock 128
    IConvolutionLayer *conv19 = network->addConvolutionNd(*relu18->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_19/kernel:0"], weightMap["conv2d_19/bias:0"]);
    assert(conv19);
    // conv19->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv19->setPaddingNd(DimsHW{1, 1});
    conv19->setStrideNd(DimsHW{1, 1});
    auto bn19 = addBatchNorm2d(network, weightMap, *conv19->getOutput(0), "batch_normalization_19", 0.001);
    IActivationLayer *relu19 = network->addActivation(*bn19->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv20 = network->addConvolutionNd(*relu19->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_20/kernel:0"], weightMap["conv2d_20/bias:0"]);
    assert(conv20);
    conv20->setStrideNd(DimsHW{1, 1});
    // conv20->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv20->setPaddingNd(DimsHW{1, 1});
    auto bn20 = addBatchNorm2d(network, weightMap, *conv20->getOutput(0), "batch_normalization_20", 0.001);
    IElementWiseLayer *sum7 = network->addElementWise(*bn20->getOutput(0), *relu18->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu20 = network->addActivation(*sum7->getOutput(0), ActivationType::kRELU);

    // resblock 128
    IConvolutionLayer *conv21 = network->addConvolutionNd(*relu20->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_21/kernel:0"], weightMap["conv2d_21/bias:0"]);
    assert(conv21);
    conv21->setStrideNd(DimsHW{1, 1});
    // conv21->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv21->setPaddingNd(DimsHW{1, 1});
    auto bn21 = addBatchNorm2d(network, weightMap, *conv21->getOutput(0), "batch_normalization_21", 0.001);
    IActivationLayer *relu21 = network->addActivation(*bn21->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv22 = network->addConvolutionNd(*relu21->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_22/kernel:0"], weightMap["conv2d_22/bias:0"]);
    assert(conv22);
    conv22->setStrideNd(DimsHW{1, 1});
    // conv22->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv22->setPaddingNd(DimsHW{1, 1});
    auto bn22 = addBatchNorm2d(network, weightMap, *conv22->getOutput(0), "batch_normalization_22", 0.001);
    IElementWiseLayer *sum8 = network->addElementWise(*bn22->getOutput(0), *relu20->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu22 = network->addActivation(*sum8->getOutput(0), ActivationType::kRELU);

    // resblock 128
    IConvolutionLayer *conv23 = network->addConvolutionNd(*relu22->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_23/kernel:0"], weightMap["conv2d_23/bias:0"]);
    assert(conv23);
    conv23->setStrideNd(DimsHW{1, 1});
    // conv23->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv23->setPaddingNd(DimsHW{1, 1});
    auto bn23 = addBatchNorm2d(network, weightMap, *conv23->getOutput(0), "batch_normalization_23", 0.001);
    IActivationLayer *relu23 = network->addActivation(*bn23->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv24 = network->addConvolutionNd(*relu23->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2d_24/kernel:0"], weightMap["conv2d_24/bias:0"]);
    assert(conv24);
    conv24->setStrideNd(DimsHW{1, 1});
    // conv24->setPaddingMode(PaddingMode::kSAME_UPPER);
    conv24->setPaddingNd(DimsHW{1, 1});
    auto bn24 = addBatchNorm2d(network, weightMap, *conv24->getOutput(0), "batch_normalization_24", 0.001);
    IElementWiseLayer *sum9 = network->addElementWise(*bn24->getOutput(0), *relu22->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer *relu24 = network->addActivation(*sum9->getOutput(0), ActivationType::kRELU);

    // detection
    IConvolutionLayer *conv25 = network->addConvolutionNd(*relu24->getOutput(0), 2, DimsHW{3, 3}, weightMap["conv2d_25/kernel:0"], weightMap["conv2d_25/bias:0"]);
    assert(conv25);
    conv25->setStrideNd(DimsHW{1, 1});
    conv25->setPaddingNd(DimsHW{1, 1});
    ISoftMaxLayer *softmax = network->addSoftMax(*conv25->getOutput(0));
    IConvolutionLayer *conv26 = network->addConvolutionNd(*relu24->getOutput(0), 6, DimsHW{3, 3}, weightMap["conv2d_26/kernel:0"], weightMap["conv2d_26/bias:0"]);
    assert(conv26);
    conv26->setStrideNd(DimsHW{1, 1});
    conv26->setPaddingNd(DimsHW{1, 1});
    ITensor *c[] = {softmax->getOutput(0), conv26->getOutput(0)};
    IConcatenationLayer *concat = network->addConcatenation(c, 2);
    concat->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*concat->getOutput(0));

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    // profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
    // profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, OPT_INPUT_H, OPT_INPUT_W));
    // profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB

#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int h_scale, int w_scale)
{
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    context.setBindingDimensions(inputIndex, Dims3(3, h_scale, w_scale));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * h_scale * w_scale * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], 8 * 16 * 16 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * h_scale * w_scale * sizeof(float), cudaMemcpyHostToDevice, stream));
    // context.enqueueV2(buffers, stream, nullptr);
    context.enqueue(1, buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], 8 * 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void build_engine()
{
    IHostMemory *modelStream{nullptr};
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p("wpod.engine", std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();
}
int main(int argc, char **argv)
{
    auto opt = std::string(argv[1]);
    if (opt[0] == 'c')
        build_engine();
    else if (opt[0] == 't')
    {
        std::string file_name = "/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/wpod/build/output/out_0.jpg";
        cv::Mat pr_img = cv::imread(file_name);
        assert(!pr_img.empty());
        cv::Mat out_im = cv::Mat(cv::Size(280*2,100), CV_8UC3);
        pr_img(cv::Rect(0,100,280,100)).copyTo(out_im(cv::Rect(0,0,280,100)));
        pr_img(cv::Rect(0,100,280,100)).copyTo(out_im(cv::Rect(280,0,280,100)));
        cv::imwrite("concat.jpg",out_im);
        
    }
    else
    {
        std::string file_name = opt;
        cv::Mat pr_img = cv::imread(file_name);
        assert(!pr_img.empty());
        cv::Mat src_img;
        cv::Mat img;
        cv::cvtColor(pr_img, src_img, cv::COLOR_BGR2RGB);
        cv::resize(src_img, img, cv::Size(SHORT_INPUT, SHORT_INPUT), cv::INTER_NEAREST);
        float *data = new float[3 * img.rows * img.cols];
        auto start = std::chrono::system_clock::now();
        int i = 0;

        for (int i = 0; i < 256 * 256; i++)
        {
            data[i] = (float)img.at<cv::Vec3b>(i)[0] / 255.0;

            data[i + 256 * 256] = (float)img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * 256 * 256] = (float)img.at<cv::Vec3b>(i)[2] / 255.0;
        }
        auto end = std::chrono::system_clock::now();
        std::cout << "pre time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        char *trtModelStream{nullptr};
        size_t size{0};

        std::ifstream file("wpod.engine", std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        IRuntime *runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        IExecutionContext *context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        static float *prob = new float[16 * 16 * 8];
        // Run inference
        auto start1 = std::chrono::system_clock::now();
        doInference(*context, data, prob, SHORT_INPUT, SHORT_INPUT);
        std::vector<cv::Mat> output;
        post_process(img, output, prob, 256, 256);
        auto end1 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;
        for (int i = 0; i < output.size(); i++)
        {
            cv::imwrite("output/out_" + std::to_string(i) + ".jpg", output[i]);
        }

        delete[] data;
        delete[] prob;
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }
    return 0;
}