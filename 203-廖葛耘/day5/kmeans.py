import numpy as np
import random
import sys

class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__cosine_distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        if np.array_equal(self.points, new_center):
            sum_similarity = self.__sum_similarity(result)
            return result, self.points, sum_similarity
        self.points = np.array(new_center)
        return self.cluster()

    def __sum_similarity(self, result):
        sum_similarity = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum_similarity += self.__cosine_similarity(result[i][j], self.points[i])
        return sum_similarity

    def __center(self, lst):
        return np.array(lst).mean(axis=0)

    def __cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

    def __cosine_distance(self, vec1, vec2):
        return 1 - self.__cosine_similarity(vec1, vec2)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, similarities = kmeans.cluster()
print(result)
print(centers)
print(similarities)
