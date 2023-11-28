#include "neuralnet.h"

#include <cstdlib>
#include <time.h>

double Logistic(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double LogisticDerivative(double x)
{
	double fx = Logistic(x);
	if (fx == 0) return 0.5;
	return fx - (fx * fx);
}

double OutputError(double target, double output)
{
	double diff = target - output;
	return 0.5 * (diff * diff);
}

double Clamp(double input)
{
	if (input < 0.0) return 0.0;
	if (input > 1.0) return 1.0;
	return input;
}

std::vector<double> NeuralNet::Forward(std::vector<double> inputs)
{
	// First, we feed our inputs into the input layer of our
	// network.
	for (int i = 0; i < inputs.size(); i++)
	{
		layers[0].outputs(0, i) = inputs[i];
	}

	// Now, we go through each layer and propagate these forward,
	// handling different layer types as necessary.
	for (int i = 1; i < LayerCount; i++)
	{
		Layer* out = &layers[i - 1];
		Layer* in = &layers[i];

		// Convolutional layers are a pain, so we need to handle them
		// separately from the rest.
		if (in->layerType == LayerType::convolutional)
		{
			/*
				Okay, so, we have a convolutional layer.
				1D convolutional layers are nice and easy.
			*/

			in->inputs = arma::conv(out->outputs, in->filter, "same");
		}
		else if (in->layerType == LayerType::pooling)
		{
			/*
				Pooling layers apply some function across
				sections of the preceding layer.
			*/

			if (in->poolType == PoolType::average)
			{
				for (int j = 0; j < out->outputs.n_cols; j += 2)
				{
					double sum = out->outputs(0, j) + out->outputs(0, j + 1);
					in->inputs(0, j / 2) = (sum / 2.0);
				}
			}
			else // if (in->poolType == PoolType::max)
			{
				for (int j = 0; j < out->outputs.n_cols; j += 2)
				{
					double max = std::max(out->outputs(0, j), out->outputs(0, j + 1));
					in->inputs(0, j / 2) = max;
				}
			}
		}
		else
		{
			// Plain old layers are nice and easy.
			in->inputs = out->outputs * in->weights;
		}

		// Regardless of the type of layer, we go through
		// the inputs and transfer them to the outputs.
		for (int j = 0; j < in->inputs.n_cols; j++)
		{
			in->inputs(0, j) += in->biases(0, j);
			in->outputs(0, j) = Clamp(Logistic(in->inputs(0, j)));
		}
	}

	// Then, because for some awful reason I decided to use
	// an excess of vectors, we need to pull the outputs out
	// of the final layer and return them.
	std::vector<double> outputs;
	for (int i = 0; i < layers[LayerCount - 1].outputs.n_cols; i++) outputs.push_back(layers[LayerCount - 1].outputs(0, i));
	// std::cout << outputs[0] << "\n";
	return outputs;
}

void NeuralNet::Back(std::vector<double> costs)
{
	/*
		We go back through each layer computing the derivatives
		until we reach the second layer.

		First, we'll start with the last layer.
	*/

	Layer* last = &layers[LayerCount - 1];

	// First, we'll get the input and output derivatives.
	for (int i = 0; i < last->outputs.n_cols; i++)
	{
		last->outputDerivatives(0, i) = costs[i];
		double z = last->inputs(0, i);
		last->inputDerivatives(0, i) = LogisticDerivative(z);
	}

	// Then we'll do our biases.
	for (int i = 0; i < last->biases.n_cols; i++)
	{
		double outputDeriv = last->outputDerivatives(0, i);
		double inputDeriv = last->inputDerivatives(0, i);

		last->biasDerivatives(0, i) = outputDeriv * inputDeriv;
	}

	// And finally our weights.
	for (int i = 0; i < last->weights.n_rows; i++)
	{
		for (int j = 0; j < last->weights.n_cols; j++)
		{
			double outputDeriv = last->outputDerivatives(0, j);
			double inputDeriv = last->inputDerivatives(0, j);
			double weightDeriv = layers[LayerCount - 2].outputs(0, i);

			last->weightDerivatives(i, j) = outputDeriv * inputDeriv * weightDeriv;
		}
	}

	// And now we can go through and do the rest of our layers.
	for (int i = LayerCount - 2; i > 0; i--)
	{
		Layer* prev = &layers[i - 1];
		Layer* current = &layers[i];
		Layer* next = &layers[i + 1];

		/*
			The derivative of the error with respect to a neuron's output
			is the sum of the derivatives of the error with respect to the
			outputs of the neurons in the next layer times the derivatives of
			these neurons' outputs with respect to the output of the neuron in
			question.

			The derivative of a neuron's output (which we'll call neuron OUT) with
			respect to the output of a preceding neuron (which will be neuron IN)
			is the derivative of the logistic function with regard to the output of
			OUT times the derivative of the output of OUT with respect to the output
			of IN (which is the weight associated with the connection between IN and
			OUT).

			In other words: what we'll just call the "derivative" of
			a neuron is the sum of the "derivatives" of the neurons in the next
			layer times the derivatives of their outputs with respect to the
			neuron in question, which is just the LogisticDerivative() of the
			next-layer neuron times the weight between the two neurons.
		*/

		if (next->layerType != LayerType::convolutional &&
			next->layerType != LayerType::pooling)
		{
			for (int j = 0; j < current->outputs.n_cols; j++)
			{
				double sum = 0.0;

				for (int k = 0; k < next->weights.n_cols; k++)
				{
					double outputDeriv = next->outputDerivatives(0, k);
					double inputDeriv = next->inputDerivatives(0, k);
					double weight = next->weights(j, k);

					sum += outputDeriv * inputDeriv * weight;
				}

				current->outputDerivatives(0, j) = sum;
				current->inputDerivatives(0, j) = LogisticDerivative(current->inputs(0, j));
			}
		}
		else if (next->layerType == LayerType::convolutional)
		{
			/*
				Calculating the derivatives of layers preceding convolutional
				layers is not fun. It isn't hard, just annoying.

				Each node in the convolutional layer is the sum of the values
				of the kernel after multiplying with a section of the preceding
				layer's outputs. This means that each node in the preceding layer
				affects upwards of 9 nodes in the convolutional layer. However,
				other nodes in the preceding layer affect only one node in the
				convolutional layer. This is mildly annoying.

				The derivative of the error with regard to a node in a layer
				preceding a convolutional layer is the sum of the derivatives
				of the error with regard to the nodes in the convolutional layer
				that are affected by the node in question.

				First, we'll calculate the output derivatives of the current
				layer by preparing a 2d matrix of the next layer's derivatives
				and then convoluting it w/ the filter. This should yield a
				matrix of the same dimensions which has the derivative of the
				error with respect to the outputs of the current layer.
			*/

			arma::mat nextDerivatives;

			nextDerivatives = arma::mat(1, next->outputDerivatives.n_cols);
			for (int j = 0; j < next->outputDerivatives.n_cols; j++) nextDerivatives(0, j) = next->outputDerivatives(0, j) * next->inputDerivatives(0, j);

			current->outputDerivatives = arma::conv(nextDerivatives, next->filter, "same");

			for (int j = 0; j < current->outputs.n_cols; j++)
			{
				nextDerivatives(0, j) *= current->outputs(0, j);
			}

			/*
				Now we have to calculate the derivatives of the
				filter which is more complex. The derivative of the error
				with respect to each node in the filter is equal to the
				derivative of the error with respect to the outputs of the
				next layer times the derivative of those outputs with respect
				to the inputs times the derivative of those inputs with respect
				to the node in the filter. The derivative of an input with respect
				to a particular node in the filter is equal to the sum of the
				outputs of the neurons in the preceding layer that are multiplied
				by that node in the filter and go into the specific input in the
				next layer.
			*/

			next->filterDerivatives = arma::conv(next->filter, nextDerivatives, "same");

			// And now we need to throw together the input derivatives.
			for (int j = 0; j < current->inputs.n_cols; j++)
			{
				current->inputDerivatives(0, j) = LogisticDerivative(current->inputs(0, j));
			}
		}
		else // if (next->layerType == LayerType::pooling)
		{
			/*
				Calculating the derivatives of a layer preceding a pooling layer isn't
				as elegant as the preceding layer types.
			*/

			for (int j = 0; j < next->inputs.n_cols; j++)
			{
				double outputDeriv = next->outputDerivatives(0, j);
				double inputDeriv = next->inputDerivatives(0, j);
				double funcDerivA = 0.5;
				double funcDerivB = 0.5;

				if (next->poolType == PoolType::max)
				{
					/*
						So, whichever has the higher output has a funcDeriv of 1.0,
						whereas the lower has the funcDeriv of 0.0.
					*/

					bool aHigherB = (current->outputs(0, j * 2) > current->outputs(0, (j * 2) + 1));

					funcDerivA = aHigherB ? 1.0 : 0.0;
					funcDerivB = !aHigherB ? 1.0 : 0.0;
				}

				current->outputDerivatives(0, j * 2) = outputDeriv * inputDeriv * funcDerivA;
				current->inputDerivatives(0, j * 2) = LogisticDerivative(current->inputs(0, j * 2));

				current->outputDerivatives(0, (j * 2) + 1) = outputDeriv * inputDeriv * funcDerivB;
				current->inputDerivatives(0, (j * 2) + 1) = LogisticDerivative(current->inputs(0, (j * 2) + 1));
			}
		}

		/*
			And now we can calculate our bias derivatives. These are
			considerably easier.
		*/
		for (int j = 0; j < current->biases.n_cols; j++)
		{
			double outputDeriv = current->outputDerivatives(0, j);
			double inputDeriv = current->inputDerivatives(0, j);

			current->biasDerivatives(0, j) = outputDeriv * inputDeriv;
		}

		// We don't need to calculate weight derivatives for
		// convolutional layers.

		if (current->layerType == LayerType::convolutional ||
			current->layerType == LayerType::pooling) continue;

		/*
			Let's say that a weight takes the outputs from
			a neuron, IN, and feeds them into the inputs of a
			neuron, OUT, which we might represent as:
			IN -> weight -> OUT.

			Then, the derivative of the error with respect to
			that weight is the derivative of the error with
			respect to OUT's output times the derivative of
			OUT's output with respect to the weight.

			This is in turn calculated as the derivative of OUT's
			output with respect to OUT's input times the derivative
			of OUT's input with respect to the weight.

			The derivative of OUT's output with respect to OUT's input
			is calculated with LogisticDerivative(), while the derivative
			of OUT's input with respect to the weight is equal to
			IN's output.

			Putting this all together, we take the pre-calculated derivative
			for OUT, multiply it be the LogisticDerivative() of OUT's input,
			and in turn multiply this by IN's output.
		*/
		for (int j = 0; j < current->weights.n_rows; j++)
		{
			for (int k = 0; k < current->weights.n_cols; k++)
			{
				double outputDeriv = current->outputDerivatives(0, k);
				double inputDeriv = current->inputDerivatives(0, k);
				double weightDeriv = prev->outputs(0, j);

				current->weightDerivatives(j, k) = outputDeriv * inputDeriv * weightDeriv;
			}
		}
	}

	/*
		Because of the way that previous loop is written, I actually
		need to check if the second layer is a convolutional layer and update its
		derivatives if so. Luckily, this mostly involves reiterating code we've
		already written.
	*/

	if (layers[1].layerType == LayerType::convolutional)
	{
		Layer* current = &layers[0];
		Layer* next = &layers[1];

		arma::mat nextDerivatives = arma::mat(1, next->outputDerivatives.n_cols);
		for (int j = 0; j < next->outputDerivatives.n_cols; j++)
		{
			nextDerivatives(0, j) = next->outputDerivatives(0, j) * next->inputDerivatives(0, j);
		}

		current->outputDerivatives = arma::conv(nextDerivatives, next->filter, "same");

		for (int j = 0; j < current->outputs.n_cols; j++)
		{
			nextDerivatives(0, j) *= current->outputs(0, j);
		}

		next->filterDerivatives = arma::conv(next->filter, nextDerivatives, "same");
	}

	/*
		Now, we'll work forward, updating all our weights
		and biases.
	*/
	for (int i = 1; i < LayerCount; i++)
	{
		Layer* layer = &layers[i];

		layer->weights -= layer->weightDerivatives * LearningRate;
		layer->biases -= layer->biasDerivatives * LearningRate;
		layer->filter -= layer->filterDerivatives * LearningRate;

		// And now we clear all of the derivatives.
		layer->outputDerivatives.zeros();
		layer->inputDerivatives.zeros();
		layer->weightDerivatives.zeros();
		layer->biasDerivatives.zeros();
		layer->filterDerivatives.zeros();
	}
}

void NeuralNet::ClearInOutputs()
{
	for (int i = 0; i < LayerCount; i++)
	{
		layers[i].inputs.zeros();
		layers[i].outputs.zeros();
	}
}

double NeuralNet::Train(std::vector<double> inputs, std::vector<double> targets)
{
	std::vector<double> output = Forward(inputs);

	double loss = 0.0;
	for (int i = 0; i < output.size(); i++)
	{
		output[i] -= targets[i];
		loss += output[i] * output[i];
	}

	Back(output);

	ClearInOutputs();

	return loss;
}

NeuralNet::NeuralNet(std::vector<LayerType> hiddenLayerTypes, std::vector<PoolType> poolTypes)
{
	// Seed arma random
	arma::arma_rng::set_seed_random();
	int pools = 0;

	/*
		First, let's instantiate our layers.

		We need to do our inputs and outputs first.
	*/

	// Inputs
	layers[0].layerType = LayerType::input;
	layers[0].outputs = arma::mat(1, InputCount);

	// Outputs
	layers[LayerCount - 1].layerType = LayerType::output;
	layers[LayerCount - 1].inputs = arma::mat(1, OutputCount);
	layers[LayerCount - 1].outputs = arma::mat(1, OutputCount);

	layers[LayerCount - 1].inputDerivatives = arma::mat(1, OutputCount);
	layers[LayerCount - 1].outputDerivatives = arma::mat(1, OutputCount);

	layers[LayerCount - 1].biases = arma::mat(1, OutputCount, arma::fill::randu);
	layers[LayerCount - 1].biasDerivatives = arma::mat(1, OutputCount);
	
	// Now, we go through and instantiate each internal layer's
	// various features.
	for (int i = 1; i < LayerCount - 1; i++)
	{
		layers[i].layerType = hiddenLayerTypes[i - 1];

		// If this is a convolutional layer, we need to set up
		// its filter.
		// Right now, all filters are 3x3, but we will change that
		// once we move away from testing the network on visual data.
		if (layers[i].layerType == LayerType::convolutional)
		{
			layers[i].filter = arma::mat(1, layers[i].kernelSize, arma::fill::randu);

			layers[i].filterDerivatives = arma::mat(layers[i].filter.n_rows, layers[i].filter.n_cols);

			layers[i].inputs = arma::mat(1, layers[i - 1].outputs.n_cols);
			layers[i].outputs = arma::mat(1, layers[i].inputs.n_cols);
		}
		else if (layers[i].layerType == LayerType::pooling)
		{
			layers[i].poolType = poolTypes[pools];

			layers[i].inputs = arma::mat(1, layers[i - 1].outputs.n_cols / 2);
			layers[i].outputs = arma::mat(1, layers[i].inputs.n_cols);

			pools++;
		}
		else
		{
			// The only other option is that this is a plain old
			// fully-connected hidden layer.
			layers[i].inputs = arma::mat(1, HiddenNeuronCount);
			layers[i].outputs = arma::mat(1, HiddenNeuronCount);
		}

		// Regardless of the type of layer, the number of input, output, and bias
		// derivatives will always match.
		layers[i].inputDerivatives = arma::mat(1, layers[i].inputs.n_cols);
		layers[i].outputDerivatives = arma::mat(1, layers[i].outputs.n_cols);

		layers[i].biases = arma::mat(1, layers[i].inputs.n_cols, arma::fill::randu);
		layers[i].biasDerivatives = arma::mat(1, layers[i].biases.n_cols);
	}

	// Now, we need to activate the weights.
	// Each layer includes the weights that feed *into* it.
	// This is, in part, because convolutional layers need to be handled
	// somewhat differently than other layers.
	// We handle the weights after going through all the layers once since
	// we need to know the size of the preceding layer in order to create
	// the right number of weights.
	// Since the input layer has no weights, we can safely ignore it.
	// But we need to go all the way through to the output layer.
	for (int i = 1; i < LayerCount; i++)
	{
		// Convolutional layers are strange in that they apply filters rather than weights.
		// We already instantiated the filters so we can ignore them.
		if (layers[i].layerType == LayerType::convolutional ||
			layers[i].layerType == LayerType::pooling) continue;

		// The number of columns in the weights must be equal to the number of outputs of the
		// preceding layer, while the number of rows must equal the number of inputs of this layer.
		layers[i].weights = arma::mat(layers[i - 1].outputs.n_cols, layers[i].inputs.n_cols, arma::fill::randu);
		layers[i].weightDerivatives = arma::mat(layers[i].weights.n_rows, layers[i].weights.n_cols);

		for (int j = 0; j < layers[i].weights.n_rows; j++)
		{
			for (int k = 0; k < layers[i].weights.n_cols; k++)
			{
				layers[i].weights(j, k) *= 0.1;
			}
		}
	}

	// And now we should be done.
}