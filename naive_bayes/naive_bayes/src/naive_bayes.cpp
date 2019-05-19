#include "naive_bayes.hpp"
#include <iostream>

void naive_bayes_mnist_classifier::buildLikelihoodTableFromMnistTrainingSet(const std::vector<std::vector<uint8_t>>& training_images, const std::vector<uint8_t>& training_lables, uint8_t pixelObserveThreshold)
{
	if (!prepareClassifier(training_images, training_lables)) {

		std::cerr << "Invalid MNIST data input" << std::endl;
		return;
	}

	int imageProcessedCounter = 0;

	for (const auto& image : training_images) {

		uint8_t actualLabel = training_lables[imageProcessedCounter];

		for(size_t pixelNumber = 0; pixelNumber < sizeOfImageInPixel; pixelNumber++){

			uint16_t pixelColor = image[pixelNumber];

			if (pixelColor > pixelObserveThreshold) {

				likelihoodtable[pixelNumber][actualLabel]++;
			}
		}

		imageProcessedCounter++;
	}

	for (int pixelNumber = 0; pixelNumber < sizeOfImageInPixel; pixelNumber++) {
		normalizeLikelihoodTable(likelihoodtable, pixelNumber);
	}
}

uint8_t naive_bayes_mnist_classifier::classify(const std::vector<uint8_t>& image, uint8_t pixelObserveThreshold)
{
	std::map<uint8_t, float> resultingClassifications;

	for (size_t digit = 0; digit <= 9; digit++) {

		resultingClassifications[digit] = rateOfDigitsInTrainingImages[digit];

		for (size_t pixelNumber = 0; pixelNumber < image.size(); pixelNumber++) {

			uint8_t pixel = image[pixelNumber];

			if (pixel > pixelObserveThreshold) {

				float p_x_c = likelihoodtable[pixelNumber][digit] > 0.f ? likelihoodtable[pixelNumber][digit] : 1.f;
				resultingClassifications[digit] *= p_x_c * 7;

				//std::cout << "Digit " << digit << ": " << resultingClassifications[digit] << std::endl;
			}
		}
	}

	return utilManager.maxPixel(resultingClassifications);
}

bool naive_bayes_mnist_classifier::prepareClassifier(const std::vector<std::vector<uint8_t>>& training_images, const std::vector<uint8_t>& training_lables)
{
	if (training_images.size() != training_lables.size() || training_images.size() == 0) return false;

	numberOfTrainingImages = training_images.size();

	sizeOfImageInPixel = training_images[1].size();

	for (uint8_t label : training_lables) {

		numberOfDigitsInTrainingImages[label]++;
	}

	for (uint8_t label : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {

		rateOfDigitsInTrainingImages[label] = (float) numberOfDigitsInTrainingImages[label] / numberOfTrainingImages;
	}

	return true;
}

uint8_t naive_bayes_mnist_classifier::naive_bayes_util::maxPixel(const std::map<uint8_t, float>& pixels) const
{
	float maxValue = -FLT_MAX;
	uint8_t maxDigit = 0;

	for (const auto [digit, value] : pixels) {

		if (value > maxValue) {

			maxValue = value;
			maxDigit = digit;
		}
	}

	return maxDigit;
}
