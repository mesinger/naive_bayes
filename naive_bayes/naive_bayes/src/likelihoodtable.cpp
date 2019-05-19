#include "likelihoodtable.hpp"

void normalizeLikelihoodTable(LikelihoodTable& table, uint16_t pixel)
{
	float sum = 0.f;

	for (const auto& [digitClassification, totalNumber] : table[pixel]) {

		sum += totalNumber;
	}

	for (const auto& [digitClassification, totalNumber] : table[pixel]) {

		table[pixel][digitClassification] /= sum;
	}
}
