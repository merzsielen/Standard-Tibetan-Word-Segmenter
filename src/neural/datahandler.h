#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include <unordered_map>

#include "neuralnet.h"

struct Embedding
{
	unsigned int	index;
	std::wstring	syllable;
	double			vector[100];
};

class DataHandler
{
private:
	unsigned int							targetCursor;
	unsigned int							inputCursor;

	NeuralNet*								network;

	std::wstring							tokenizedCorpus;
	std::wstring							syllabizedTokenizedCorpus;
	std::wstring							nontokenizedCorpus;

	std::unordered_map<std::wstring, int>	freqMap;
	std::unordered_map<std::wstring, int>	indexMap;
	std::unordered_map<int, std::wstring>	tempMap;
	std::unordered_map<int, std::wstring>	sylMap;
	std::unordered_map<std::wstring, Embedding>		embMap;

	unsigned int							nUniqueSyllables;
	unsigned int							nTotalSyllables;

	void									ReadEmbeddings(std::string path);

	std::wstring							NextTargetSyllable();
	std::wstring							NextInputSyllable();
	void									Syllabize();

public:
	void									Train();

	DataHandler(std::wstring nontoken, std::wstring token, std::vector<LayerType> hiddenLayerTypes, std::vector<PoolType> poolTypes);
	~DataHandler();
};

#endif