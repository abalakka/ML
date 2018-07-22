from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]] }
new_features = [5,7]

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1], s=100, color=i)
plt.scatter(new_features[0], new_features[1], s=99, color='c')

def k_nrst_nbr(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value let than total voting groups')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_dist = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_dist, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

result = k_nrst_nbr(dataset, new_features)
print(result)
plt.show()














































































