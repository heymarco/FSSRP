import random
import numpy as np
from river import base, drift, utils, tree
from river.metrics.base import Metric

class ROSE(base.Classifier, base.MultiClassifier):

    def __init__(self, tree_learner=tree.HoeffdingTreeClassifier, ensemble_size=10,
                 lambda_=6.0, drift_detection_method=None, warning_detection_method=None,
                 feature_space=1, percentage_features_mean=0.7, theta=0.99, window_size=500):

        self.tree_learner = tree_learner
        self.ensemble_size = ensemble_size
        self.lambda_ = lambda_
        self.drift_detection_method = drift_detection_method if drift_detection_method is not None else drift.ADWIN()
        self.warning_detection_method = warning_detection_method if warning_detection_method is not None else drift.ADWIN()
        self.feature_space = feature_space
        self.percentage_features_mean = percentage_features_mean
        self.theta = theta
        self.window_size = window_size

        self.ensemble = None
        self.ensemble_background = None
        self.instances_class = None
        self.class_size = None
        self.instances_seen = 0
        self.first_warning_on = 0
        self.warning_detected = True
        self.evaluator = None

    def reset(self):
        self.warning_detected = True
        self.first_warning_on = 0
        self.instances_class = None
        self.ensemble = None
        self.ensemble_background = None
        self.class_size = None
        self.instances_seen = 0
        self.evaluator = utils.Metric()

    def train_one(self, x, y):

        self.instances_seen += 1

        if self.instances_class is None:
            self._init_ensemble(x)

        class_val = int(y)
        self.instances_class[class_val].add((x, y), self.instances_seen)

        for i in range(len(self.class_size)):
            self.class_size[i] = self.theta * self.class_size[i] + (1 - self.theta) * (1 if int(y) == i else 0)

        lambda_val = self.lambda_ + self.lambda_ * np.log(self.class_size[np.argmax(self.class_size)] / self.class_size[class_val])

        for i in range(len(self.ensemble)):
            votes = self.ensemble[i].predict_proba_one(x)
            k = np.random.poisson(lambda_val)
            if k > 0:
                self.ensemble[i].learn_many([(x, y)] * k)

        if not self.warning_detected:
            for i in range(len(self.ensemble)):
                if self.ensemble[i].warning_detected:
                    self.warning_detected = True
                    self.first_warning_on = self.instances_seen
                    break

            if self.warning_detected:
                # Create a new background ensemble
                self.ensemble_background = [self._create_background_learner() for _ in range(self.ensemble_size)]
                index_class = [0] * len(self.instances_class)
                oldest_timestamps = [self.instances_class[i].get_timestamp(index_class[i]) for i in range(len(self.instances_class))]

                while True:
                    oldest_timestamp = float('inf')
                    next_class = -1

                    for c in range(len(self.instances_class)):
                        if oldest_timestamps[c] is not None and oldest_timestamps[c] < oldest_timestamp:
                            oldest_timestamp = oldest_timestamps[c]
                            next_class = c

                    if next_class == -1:
                        break

                    window_instance, _ = self.instances_class[next_class].get_instance(index_class[next_class])
                    for i in range(len(self.ensemble_background)):
                        votes = self.ensemble_background[i].predict_proba_one(window_instance)
                        k = np.random.poisson(self.lambda_)
                        if k > 0:
                            self.ensemble_background[i].learn_many([(window_instance, next_class)] * k)

                    index_class[next_class] += 1

        if self.warning_detected:
            for i in range(len(self.ensemble_background)):
                votes = self.ensemble_background[i].predict_proba_one(x)
                k = np.random.poisson(lambda_val)
                if k > 0:
                    self.ensemble_background[i].learn_many([(x, y)] * k)

            if self.instances_seen - self.first_warning_on == self.window_size:
                # Compare the ensemble and the background ensemble. Select the best components

                classifiers = []
                selection = []
                kappas = []
                accuracies = []

                for i in range(len(self.ensemble)):
                    classifiers.append(self.ensemble[i])
                    kappas.append(self.ensemble[i].evaluator.kappa())
                    accuracies.append(self.ensemble[i].evaluator.accuracy())

                for i in range(len(self.ensemble_background)):
                    classifiers.append(self.ensemble_background[i])
                    kappas.append(self.ensemble_background[i].evaluator.kappa())
                    accuracies.append(self.ensemble_background[i].evaluator.accuracy())

                for i in range(len(self.ensemble)):
                    max_kappa_accuracy = -1
                    max_kappa_accuracy_classifier = -1

                    for j in range(len(self.ensemble) + len(self.ensemble_background) - i):
                        if kappas[j] * accuracies[j] >= max_kappa_accuracy:
                            max_kappa_accuracy = kappas[j] * accuracies[j]
                            max_kappa_accuracy_classifier = j

                    selection.append(classifiers[max_kappa_accuracy_classifier])

                    classifiers.pop(max_kappa_accuracy_classifier)
                    kappas.pop(max_kappa_accuracy_classifier)
                    accuracies.pop(max_kappa_accuracy_classifier)

                self.ensemble = selection
                self.ensemble_background = None
                self.warning_detected = False

    def predict_proba_one(self, x):
        if self.ensemble is None:
            return []

        combined_vote = [0.0] * len(self.instances_class)

        for i in range(len(self.ensemble)):
            votes = self.ensemble[i].predict_proba_one(x)
            combined_vote = [a + b for a, b in zip(combined_vote, votes)]

        if sum(combined_vote) > 0.0:
            combined_vote = [x / sum(combined_vote) for x in combined_vote]

        return combined_vote

    def _init_ensemble(self, x):
        self.ensemble = [self.tree_learner() for _ in range(self.ensemble_size)]
        self.instances_class = [utils.WindowImbalancedClassificationPerformanceEvaluator() for _ in range(self.feature_space)]
        self.class_size = [0] * len(self.instances_class)
        self.instances_seen = 0
        self.first_warning_on = 0
        self.warning_detected = True
        self.evaluator = Metric()

    def _create_background_learner(self):
        learner = self.tree_learner()
        learner.reset()
        return learner
