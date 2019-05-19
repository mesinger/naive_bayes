#include "mnist/mnist_reader.hpp"
#include "naive_bayes.hpp"
#include "statistics.hpp"

int main(int argc, char** argv) {

	std::cout << "Naive Bayes MNIST Handwritten Digits classifier" << std::endl << std::endl;

	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data");

	naive_bayes_mnist_classifier classifier;

	classifier.buildLikelihoodTableFromMnistTrainingSet(dataset.training_images, dataset.training_labels, 200);

	statistics::ConfusionMatrix matrix;
	statistics::initMatrix(matrix, 10, 10);

	int correctPredicted = 0;

	for (size_t imageNumber = 0; imageNumber < dataset.test_images.size(); imageNumber++) {

		uint8_t actual = dataset.test_labels[imageNumber];
		uint8_t classified = classifier.classify(dataset.test_images[imageNumber], 200);

		matrix[actual][classified]++;
	}

	statistics::printConfusionMatrix(matrix);

	float avgAcc = 0.f;
	for (const uint8_t digit : { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }) {

		statistics::ConfusionMetrics metrices;
		statistics::getConfusionMetrices(matrix, digit, metrices);
		avgAcc += statistics::getAcc(metrices);
	}

	avgAcc /= 10;

	std::cout << "\nAccuracy = " << avgAcc * 100 << "%" << std::endl;

 	return 0;
}