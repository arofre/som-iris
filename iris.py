import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris_data = np.loadtxt('iris-data.csv', delimiter=',', usecols=(0, 1, 2, 3))
iris_label = np.loadtxt('iris-labels.csv', delimiter=',')

iris_data = iris_data / np.max(iris_data)

class SelfOrganisingMap:
    def __init__(self):
        self.weights = np.random.uniform(0, 1, (40, 40, 4))
        self.learning_rate = 0.1
        self.sigma = 10
        self.learning_rate_decay = 0.01
        self.sigma_decay = 0.05

    def find_bmu(self, input_vector):
        bmu_idx = (0, 0)
        min_dist = float('inf')
        
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
                h = self.neighbourhood_function(np.array([x,y]), r_i_0)
                
                self.weights[x, y] += self.learning_rate * h * (input_vector - self.weights[x, y])

    def train(self, data, num_epochs):
        num_patterns = iris_data.shape[0]
        for epoch in range(num_epochs):
            for i in range(num_patterns):
                idx = np.random.randint(0, num_patterns)
                self.update_weights(data[idx])
            self.learning_rate *= np.exp(-self.learning_rate_decay * epoch)
            self.sigma *= np.exp(-self.sigma_decay * epoch)

SOM = SelfOrganisingMap() 
initial_map = SelfOrganisingMap()

initial_map.weights = SOM.weights.copy()

SOM.train(iris_data, 10)

initial_bmu_positions = np.array([initial_map.find_bmu(x) for x in iris_data])

final_bmu_positions = np.array([SOM.find_bmu(x) for x in iris_data])

plt.figure(figsize=(12, 5))
colormap = ListedColormap(["r", "g", "b"])

jitter_strength = 0.4
initial_jitter = np.random.uniform(-jitter_strength, jitter_strength, initial_bmu_positions.shape)
final_jitter = np.random.uniform(-jitter_strength, jitter_strength, final_bmu_positions.shape)

plt.subplot(1, 2, 1)
plt.scatter(initial_bmu_positions[:, 0] + initial_jitter[:, 0],
            initial_bmu_positions[:, 1] + initial_jitter[:, 1],
            c=iris_label, cmap=colormap, alpha=0.7, s=40, edgecolors='k', linewidths=0.3)
plt.title("Initial SOM")
plt.xlim(0, 40)
plt.ylim(0, 40)

plt.subplot(1, 2, 2)
plt.scatter(final_bmu_positions[:, 0] + final_jitter[:, 0],
            final_bmu_positions[:, 1] + final_jitter[:, 1],
            c=iris_label, cmap=colormap, alpha=0.7, s=40, edgecolors='k', linewidths=0.3)
plt.title("Trained SOM")
plt.xlim(0, 40)
plt.ylim(0, 40)

# Legend
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Class 0'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Class 1'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Class 2')
], loc='upper right')

plt.suptitle("Self-Organising Map on Iris Data: Initial vs Trained", fontsize=14)
plt.tight_layout()
plt.show()
