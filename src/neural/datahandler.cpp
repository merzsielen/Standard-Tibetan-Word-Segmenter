#include "datahandler.h"
#include "../utility/filehandling.h"

#include <queue>

//double WCharToDouble(wchar_t c)
//{
//	if (c == L' ') return DataHandler::wcharOffset - 0.001;
//	return DataHandler::wcharOffset + (0.001 * ((int)c - DataHandler::blockStart));
//}
//
//wchar_t DoubleToWChar(double d)
//{
//	if (d < DataHandler::wcharOffset) return L' ';
//	return (wchar_t)(((d - DataHandler::wcharOffset) * 1000) + DataHandler::blockStart);
//}

double WCharToDouble(wchar_t c)
{
	if (c == L' ') return 0.0;
	return (((int)c + 1) - DataHandler::blockStart) / (double)DataHandler::blockLength;
}

wchar_t DoubleToWChar(double d)
{
	if (d == 0.0) return L' ';
	return (wchar_t)((int)(d * DataHandler::blockLength) + DataHandler::blockStart - 1);
}

void DataHandler::Train()
{
	/*
		First, we need to break down our tokenized data into two sets:
		the input and the target. The input is all the data without spaces;
		the target, with. To do this, we iterate through the tokenized
		corpus word-by-word. We fit as many words as we can into the input,
		then hand the neural network the input and target data.
	*/

	int charIter = NeuralNet::InputCount;
	int trainIter = 0;
	int m = NeuralNet::InputCount / 2;

	std::vector<std::pair<double, bool>> corpus;

	for (int i = 0; i < tokenizedCorpus.size() / 8; i++)
	{
		wchar_t c = tokenizedCorpus[i];
		double d = WCharToDouble(c);

		if (c == L' ') corpus[corpus.size() - 1].second = true;
		else corpus.push_back(std::pair<double, bool>(d, false));
	}

	std::vector<double> inputs;
	std::vector<double> targets;

	// First, let's load up our inputs.
	inputs.push_back(0.0);
	for (int i = 1; i < NeuralNet::InputCount; i++)
	{
		inputs.push_back(corpus[i].first);
	}

	while (true)
	{
		// Have we reached the end of the corpus?
		if (charIter > corpus.size() - 1) break;

		// Pop the front of the inputs.
		std::vector<double> temp;
		for (int i = 1; i < inputs.size(); i++) temp.push_back(inputs[i]);
		inputs = temp;

		// Add the new back of the inputs.
		inputs.push_back(corpus[charIter].first);

		// Get the current "center" of the inputs.
		targets.push_back(corpus[(long long)charIter - m].second);

		charIter++;

		network->Train(inputs, targets, false);
		targets.clear();
		trainIter++;
	}

	std::cout << "Did " << trainIter << " iterations of training.\n";
	std::cout << "Now, we move on to testing.\n";

	/*
		Now, we're going to test it quickly.
	*/
	int testIter = 100;
	charIter = 0;// rand() % (corpus.size() - (2 * NeuralNet::InputCount));

	std::vector<double> out;
	std::vector<double> targs;
	inputs.clear();
	targets.clear();

	std::wstring inputText;
	std::wstring targetText;
	std::wstring outputText;

	// First, let's load up our inputs.
	inputs.push_back(0.0);
	for (int i = charIter; i < charIter + NeuralNet::InputCount; i++)
	{
		inputs.push_back(corpus[i].first);
	}

	charIter += NeuralNet::InputCount / 2;

	while (true)
	{
		if (testIter <= 0) break;

		// Pop the front of the inputs.
		std::vector<double> temp;
		for (int i = 1; i < inputs.size(); i++) temp.push_back(inputs[i]);
		inputs = temp;

		// Add the new back of the inputs.
		inputs.push_back(corpus[charIter].first);

		// Get the current "center" of the inputs.
		wchar_t c = DoubleToWChar(corpus[(long long)charIter - m].first);
		inputText += c;
		targetText += c;
		outputText += c;

		double isSpace = corpus[(long long)charIter - m].second;
		targets.push_back(isSpace);
		targs.push_back(isSpace);
		if (isSpace) targetText += L" ";

		charIter++;

		std::vector<double> o = network->Forward(inputs, false);
		out.push_back(o[0]);
		if (o[0]) outputText += L" ";

		targets.clear();
		testIter--;
	}

	double ssr = 0.0;
	double tss = 0.0;

	double mean = 0.0;
	for (int i = 0; i < targs.size(); i++) mean += targs[i];
	mean /= targs.size();

	for (int j = 0; j < out.size(); j++)
	{
		ssr += (targs[j] - out[j]) * (targs[j] - out[j]);
		tss += (targs[j] - mean) * (targs[j] - mean);
	}

	std::cout << "R^2: " << (1.0 - (ssr / tss)) << std::endl;
	std::wstring test = inputText + L"\n" + targetText + L"\n" + outputText;
	WriteFile("datasets/output/test_tokens.txt", test);
	std::cout << "Job done.\n";
}

DataHandler::DataHandler(NeuralNet* net, std::wstring nontoken, std::wstring token)
{
	this->network = net;
	this->nontokenizedCorpus = nontoken;
	this->tokenizedCorpus = token;
}