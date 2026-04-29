import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict, Counter


np.random.seed(42)

iris_data = np.loadtxt("iris-data.csv", delimiter=",", usecols=(0, 1, 2, 3))
iris_label = np.loadtxt("iris-labels.csv", delimiter=",")

iris_data = iris_data / np.max(iris_data)


class SelfOrganisingMap:
    def __init__(self):
        self.weights = np.random.uniform(0, 1, (40, 40, 4))
        self.learning_rate = 0.1
        self.sigma = 10
        self.learning_rate_decay = 0.01
        self.sigma_decay = 0.05

    def find_bmu(self, input_vector):
        diff = self.weights - input_vector
        dist = np.linalg.norm(diff, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dist), dist.shape)
        return bmu_idx

    def neighbourhood_function(self, r_i, r_i_0):
        diff = r_i - r_i_0
        return np.exp(-np.sum(diff**2) / (2 * self.sigma**2))

    def update_weights(self, input_vector):
        r_i_0 = self.find_bmu(input_vector)

        for x in range(self.weights.shape[0]):
            for y in range(self.weights.shape[1]):
                h = self.neighbourhood_function(np.array([x, y]), r_i_0)
                self.weights[x, y] += (
                    self.learning_rate * h * (input_vector - self.weights[x, y])
                )

    def train(self, data, num_epochs):
        num_patterns = data.shape[0]

        for epoch in range(num_epochs):
            for _ in range(num_patterns):
                idx = np.random.randint(0, num_patterns)
                self.update_weights(data[idx])

            self.learning_rate *= np.exp(-self.learning_rate_decay * epoch)
            self.sigma *= np.exp(-self.sigma_decay * epoch)


def quantization_error(som, data):
    errors = []

    for x in data:
        bmu = som.find_bmu(x)
        bmu_weight = som.weights[bmu]
        errors.append(np.linalg.norm(x - bmu_weight))

    return np.mean(errors)


def som_majority_vote_accuracy(som, data, labels):
    neuron_labels = defaultdict(list)

    for x, label in zip(data, labels):
        bmu = som.find_bmu(x)
        neuron_labels[bmu].append(int(label))

    neuron_majority = {
        neuron: Counter(labels).most_common(1)[0][0]
        for neuron, labels in neuron_labels.items()
    }

    correct = 0

    for x, label in zip(data, labels):
        bmu = som.find_bmu(x)
        predicted_label = neuron_majority[bmu]

        if predicted_label == int(label):
            correct += 1

    return correct / len(labels)


def average_same_class_bmu_distance(som, data, labels):
    bmu_positions = np.array([som.find_bmu(x) for x in data])
    distances = []

    for label in np.unique(labels):
        class_positions = bmu_positions[labels == label]

        for i in range(len(class_positions)):
            for j in range(i + 1, len(class_positions)):
                distances.append(np.linalg.norm(class_positions[i] - class_positions[j]))

    return np.mean(distances)


def average_different_class_bmu_distance(som, data, labels):
    bmu_positions = np.array([som.find_bmu(x) for x in data])
    distances = []

    for i in range(len(bmu_positions)):
        for j in range(i + 1, len(bmu_positions)):
            if labels[i] != labels[j]:
                distances.append(np.linalg.norm(bmu_positions[i] - bmu_positions[j]))

    return np.mean(distances)


SOM = SelfOrganisingMap()
initial_map = SelfOrganisingMap()

initial_map.weights = SOM.weights.copy()

initial_qe = quantization_error(initial_map, iris_data)
initial_acc = som_majority_vote_accuracy(initial_map, iris_data, iris_label)

initial_same_class_distance = average_same_class_bmu_distance(
    initial_map, iris_data, iris_label
)
initial_diff_class_distance = average_different_class_bmu_distance(
    initial_map, iris_data, iris_label
)

SOM.train(iris_data, 10)

final_qe = quantization_error(SOM, iris_data)
final_acc = som_majority_vote_accuracy(SOM, iris_data, iris_label)

final_same_class_distance = average_same_class_bmu_distance(
    SOM, iris_data, iris_label
)
final_diff_class_distance = average_different_class_bmu_distance(
    SOM, iris_data, iris_label
)

qe_reduction = 100 * (initial_qe - final_qe) / initial_qe
accuracy_improvement = 100 * (final_acc - initial_acc)

print("Initial quantization error:", initial_qe)
print("Final quantization error:", final_qe)
print("Quantization error reduction (%):", qe_reduction)

print("Initial majority-vote accuracy:", initial_acc)
print("Final majority-vote accuracy:", final_acc)
print("Accuracy improvement percentage points:", accuracy_improvement)

print("Initial same-class BMU distance:", initial_same_class_distance)
print("Final same-class BMU distance:", final_same_class_distance)

print("Initial different-class BMU distance:", initial_diff_class_distance)
print("Final different-class BMU distance:", final_diff_class_distance)

initial_bmu_positions = np.array([initial_map.find_bmu(x) for x in iris_data])
final_bmu_positions = np.array([SOM.find_bmu(x) for x in iris_data])

plt.figure(figsize=(12, 5))
colormap = ListedColormap(["r", "g", "b"])

jitter_strength = 0.4

initial_jitter = np.random.uniform(
    -jitter_strength,
    jitter_strength,
    initial_bmu_positions.shape
)

final_jitter = np.random.uniform(
    -jitter_strength,
    jitter_strength,
    final_bmu_positions.shape
)

plt.subplot(1, 2, 1)
plt.scatter(
    initial_bmu_positions[:, 0] + initial_jitter[:, 0],
    initial_bmu_positions[:, 1] + initial_jitter[:, 1],
    c=iris_label,
    cmap=colormap,
    alpha=0.7,
    s=40,
    edgecolors="k",
    linewidths=0.3,
)
plt.title("Initial SOM")
plt.xlim(0, 40)
plt.ylim(0, 40)

plt.subplot(1, 2, 2)
plt.scatter(
    final_bmu_positions[:, 0] + final_jitter[:, 0],
    final_bmu_positions[:, 1] + final_jitter[:, 1],
    c=iris_label,
    cmap=colormap,
    alpha=0.7,
    s=40,
    edgecolors="k",
    linewidths=0.3,
)
plt.title("Trained SOM")
plt.xlim(0, 40)
plt.ylim(0, 40)

plt.legend(
    handles=[
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="r",
            markersize=8,
            label="Class 0",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="g",
            markersize=8,
            label="Class 1",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="b",
            markersize=8,
            label="Class 2",
        ),
    ],
    loc="upper right",
)

plt.suptitle("Self-Organising Map on Iris Data: Initial vs Trained", fontsize=14)
plt.tight_layout()
plt.show()
