#include "datahandler.h"
#include "../utility/filehandling.h"

double WCharToDouble(wchar_t c)
{
	if (c == L' ') return DataHandler::wcharOffset - 0.001;
	return DataHandler::wcharOffset + (0.001 * ((int)c - DataHandler::blockStart));
}

wchar_t DoubleToWChar(double d)
{
	if (d < DataHandler::wcharOffset) return L' ';
	return (wchar_t)(((d - DataHandler::wcharOffset) * 1000) + DataHandler::blockStart);
}

//double WCharToDouble(wchar_t c)
//{
//	if (c == L' ') return 0.0;
//	return (((int)c + 1) - DataHandler::blockStart) / (double)DataHandler::blockLength;
//}
//
//wchar_t DoubleToWChar(double d)
//{
//	if (d == 0.0) return L' ';
//	return (wchar_t)((int)(d * DataHandler::blockLength) + 1 + DataHandler::blockStart);
//}

void DataHandler::Train()
{
	/*
		First, we need to break down our tokenized data into two sets:
		the input and the target. The input is all the data without spaces;
		the target, with. To do this, we iterate through the tokenized
		corpus word-by-word. We fit as many words as we can into the input,
		then hand the neural network the input and target data.
	*/

	int charIter = 0;
	int trainIter = 0;

	std::vector<double> inputs;
	std::vector<double> targets;

	while (true)
	{
		// We've reached the end of the corpus and done the
		// required number of training iterations (which is currently
		// arbitrary and doesn't really seem to matter).
		/*if (trainIter >= 500000) break;
		if (charIter > tokenizedCorpus.size() - 1) charIter = 0;*/
		if (charIter > tokenizedCorpus.size() - 1) break;

		// Otherwise, keep chugging along.
		wchar_t c = tokenizedCorpus[charIter];
		double d = WCharToDouble(c);

		// We only add spaces to the target values.
		if (c != L' ')
		{
			inputs.push_back(d);
			targets.push_back(0.0);
		}
		else if (targets.size() > 0)
		{
			targets[targets.size() - 1] = 1.0;
		}

		charIter++;

		// If we've got enough inputs, send it along
		// to the neural network.
		if (inputs.size() && inputs.size() % NeuralNet::InputCount == 0)
		{
			network->Train(inputs, targets, false);
			inputs.clear();
			targets.clear();
			trainIter++;
		}
	}

	std::cout << "Did " << trainIter << " iterations of training.\n";
	std::cout << "Now, we move on to testing.\n";

	/*
		Then, we're going to test it quickly on
		the first ten characters from the
		nontokenized corpus.
	*/
	/*inputs.clear();
	targets.clear();
	std::wstring inputText;
	std::wstring targetText;
	std::wstring outputText;

	std::wstring testPhrase = L"བཀྲ་ཤིས་བདེ་ལེགསབཀྲ་ཤིས་བདེ་ལེགསབཀྲ་ཤིས་བདེ་ལེགསབཀྲ་ཤིས་བདེ་ལེགསབཀྲ་ཤིས་བདེ་ལེགསབཀྲ་ཤིས་བདེ་ལེགསབཀྲ";

	charIter = 0;
	while (true)
	{
		wchar_t c = testPhrase[charIter];
		double d = WCharToDouble(c);

		inputText += c;
		if (c != L' ')
		{
			inputs.push_back(d);
		}

		charIter++;
		if (inputs.size() && inputs.size() % NeuralNet::InputCount == 0) break;
	}

	std::vector out = network->Forward(inputs, false);

	for (int i = 0; i < out.size(); i++)
	{
		outputText += inputText[i];
		if (out[i]) outputText += L" ";
	}

	WriteFile("datasets/output/test_nontokens.txt", outputText);
	std::cout << "Job done.\n";*/

	inputs.clear();
	targets.clear();
	std::wstring inputText;
	std::wstring targetText;

	charIter = rand() % 10000;
	while (true)
	{
		wchar_t c = tokenizedCorpus[charIter];
		double d = WCharToDouble(c);

		targetText += c;
		if (c != L' ')
		{
			inputText += c;
			inputs.push_back(d);
			targets.push_back(0.0);
		}
		else if (targets.size() > 0)
		{
			targets[targets.size() - 1] = 1.0;
		}

		charIter++;
		if (inputs.size() && inputs.size() % NeuralNet::InputCount == 0) break;
	}

	std::vector out = network->Forward(inputs, false);

	double ssr = 0.0;
	double tss = 0.0;

	double mean = 0.0;
	for (int i = 0; i < targets.size(); i++) mean += targets[i];
	mean /= targets.size();

	for (int j = 0; j < out.size(); j++)
	{
		ssr += (targets[j] - out[j]) * (targets[j] - out[j]);
		tss += (targets[j] - mean) * (targets[j] - mean);
	}

	std::cout << "R^2: " << (1.0 - (ssr / tss)) << std::endl;
	std::wstring test = inputText + L"\n" + targetText + L"\n";

	for (int i = 0; i < out.size(); i++)
	{
		test += inputText[i];
		if (out[i]) test += L" ";
	}

	for (int i = 0; i < out.size(); i++) std::cout << out[i] << " / ";
	std::cout << std::endl;
	for (int i = 0; i < targets.size(); i++) std::cout << targets[i] << " / ";
	std::cout << std::endl;

	WriteFile("datasets/output/test_tokens.txt", test);
	std::cout << "Job done.\n";
}

DataHandler::DataHandler(NeuralNet* net, std::wstring nontoken, std::wstring token)
{
	this->network = net;
	this->nontokenizedCorpus = nontoken;
	this->tokenizedCorpus = token;
}