#pragma once

#include "likelihoodtable.hpp"
#include <vector>

class naive_bayes_mnist_classifier {

public:

	void buildLikelihoodTableFromMnistTrainingSet(const std::vector<std::vector<uint8_t>>& training_images, const std::vector<uint8_t>& training_lables, uint8_t pixelObserveThreshold);

	uint8_t classify(const std::vector<uint8_t>& image, uint8_t pixelObserveThreshold);

private:

	bool prepareClassifier(const std::vector<std::vector<uint8_t>>& training_images, const std::vector<uint8_t>& training_lables);

private:

	LikelihoodTable likelihoodtable;

	int numberOfTrainingImages;
	int sizeOfImageInPixel;

	std::map<uint8_t, uint32_t> numberOfDigitsInTrainingImages;
	std::map<uint8_t, float> rateOfDigitsInTrainingImages;

	class naive_bayes_util {

	public:

		uint8_t maxPixel(const std::map<uint8_t, float>& pixels) const;
	};

	naive_bayes_util utilManager;


};


