#ifndef NEURALNET_H
#define NEURALNET_H

#include <armadillo>

enum class LayerType { input, output, hidden, convolutional, pooling };
enum class PoolType { average, max };

struct Layer
{
	LayerType				layerType;

	arma::mat				inputs;
	arma::mat				outputs;

	arma::mat				inputDerivatives;
	arma::mat				outputDerivatives;

	arma::mat				biases;
	arma::mat				weights;

	arma::mat				biasDerivatives;
	arma::mat				weightDerivatives;

	bool					twoDimInputs = true;

	int						kernelSize = 3;
	arma::mat				filter;
	arma::mat				filterDerivatives;

	PoolType				poolType;
};

class NeuralNet
{
public:
	static constexpr double	LearningRate = 0.1;

	static constexpr int	InputCount = 784;
	static constexpr int	OutputCount = 10;
	static constexpr int	HiddenNeuronCount = 48;											// currently arbitrary (# per layer)
	static constexpr int	HiddenLayerCount = 3;											// currently arbitrary
	static constexpr int	LayerCount = (2 + HiddenLayerCount);							// only somewhat arbitrary

private:
	Layer					layers[LayerCount];

public:
	std::vector<double>		Forward(std::vector<double> inputs);							// Forward propagation
	void					Back(std::vector<double> costs);								// Back propagation
	double					Train(std::vector<double> inputs, std::vector<double> targets);	// Training!
	void					ClearInOutputs();												// Duh

	NeuralNet(std::vector<LayerType> hiddenLayerTypes, std::vector<PoolType> poolTypes);
};

#endif