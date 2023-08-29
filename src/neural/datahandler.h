#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include "neuralnet.h"

double WCharToDouble(wchar_t c);

wchar_t DoubleToWChar(double d);

class DataHandler
{
public:
	static constexpr int		blockStart = (int)L'༠';
	static constexpr int		blockLength = ((int)L'྾' - 1) - (int)L'༠';
	static constexpr double		wcharOffset = 0.2;

private:
	NeuralNet*		network;

	std::wstring	tokenizedCorpus;
	std::wstring	nontokenizedCorpus;

public:
	void			Train();

	DataHandler(NeuralNet* net, std::wstring nontoken, std::wstring token);
};

#endif