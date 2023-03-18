# mira.py
# -------


# Mira implementation

import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            # this is the data-structure you should use
            self.weights[label] = util.Counter()

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):

        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        self.features = trainingData[0].keys()

        newWeights = self.weights.copy()
        bestAccuracy = 0
        correct = 0
        for c in Cgrid:

            self.weights = newWeights.copy()
            for iteration in range(self.max_iterations):

                print("Starting iteration ", iteration, "...")

                for datum, trueLabel in zip(trainingData, trainingLabels):
                    scores = util.Counter()

                    for label in self.legalLabels:
                        scores[label] = self.weights[label] * datum

                    predictedLabel = scores.argMax()

                    if predictedLabel != trueLabel:
                        tau = self.getTau(trueLabel, predictedLabel, datum, c)
                        self.updateWeights(
                            trueLabel, predictedLabel, datum, tau)
        #             else:
        #                 correct += 1
        #     accuracy = correct / len(validationData)
        #     if accuracy > bestAccuracy:
        #         bestAccuracy = accuracy
        #         bestWeight = self.weights

        # self.weights = bestWeight

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """

        guesses = []
        for i in range(len(data)):
            datum = data[i]
            scores = util.Counter()
            for label in self.legalLabels:
                scores[label] = self.weights[label] * datum
            predictedLabel = scores.argMax()
            guesses.append(predictedLabel)
        return guesses

    # Funciones auxiliares para hacer el código más legible.

    def getTau(self, trueLabel, predictedLabel, datum, c):
        denominador = 0
        nominador = 0
        substract = self.weights[trueLabel] - \
            self.weights[predictedLabel]
        for key, value in datum.items():
            nominador += substract[key] * value
            denominador += value**2

        tau = min(c, (nominador + 1.0) / (denominador * 2.0))

        return tau

    def updateWeights(self, trueLabel, predictedLabel, datum, tau):
        for key, value in datum.items():
            self.weights[trueLabel][key] += tau * value
            self.weights[predictedLabel][key] -= tau * value
