#pragma once

#include <map>
#include <string>
#include <memory>

// saves only bright pixels, pixels with almost 0 are ignored
// { pixel (int) : { digits (int, range(0, 10)) : float }}
typedef std::map<uint16_t, std::map<uint8_t, float>> LikelihoodTable;
//typedef std::shared_ptr<__LikelihoodTable> LikelihoodTable;

void normalizeLikelihoodTable(LikelihoodTable& table, uint16_t pixel);
