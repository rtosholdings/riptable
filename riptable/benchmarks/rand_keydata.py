import itertools
from typing import Callable, Mapping, Union

import numpy as np
import numba as nb

from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation


MetricFunction = Callable[[np.ndarray], float]
"""Type hint describing a function which accepts an ndarray and calculates some metric about it."""

class ArrayMetric:

    @staticmethod
    @nb.njit(cache=True)
    def inversions(arr: np.ndarray) -> float:
        """
        Calculates the _total_ number of 'inversions' within an array,
        then normalizes the value based on the array length.

        Parameters
        ----------
        arr

        Returns
        -------

        """
        l = arr.shape[0]
        if l < 2:
            return 0
        inv = 0
        for i in range(l):
            for j in range(i + 1, l):
                if arr[i] > arr[j]:
                    inv += 1
        total_inv = l * (l - 1) / 2

        return 1 - (inv / total_inv)

    @staticmethod
    def adjacent_unsorted(arr: np.ndarray) -> float:
        diffs = np.diff(arr)
        asc_sorted_count = diffs[diffs > 0].sum()
        desc_sorted_count = diffs[diffs < 0].sum()

        # We want this metric to be symmetric, so it doesn't matter
        # whether the array is sorted in ascending or descending order.
        # To achieve that, take the larger of the number of ASC and DESC
        # sorted counts.
        sorted_count = max(asc_sorted_count, desc_sorted_count)

        # Normalize over the number of unique elements in the array
        # to account for multiple occurrences of a given element; this is because
        # if we sorted the array, any single instance of an element occurring
        # multiple times could be a range of indices, all of which would be correct.
        # TODO: Need to somehow normalize this over the whole [0.0, 1.0] range,
        #       currently it'd only (it seems) cover from [0.5, 1.0]
        return sorted_count / len(np.unique(arr))

    @staticmethod
    def mean_displacement(arr: np.ndarray) -> float:
        """
        Calculates the mean 'displacement' of the array elements from the position they'd
        be in if the array were to be sorted.

        Notes
        -----
        This function accounts for multiple occurrences of an element, and is also symmetric
        -- reversing an array should result in the same score (to make it so that it's irrelevant
        whether the array is sorted by ascending or descending values).
        """
        # We're dealing with integer key arrays (integers from 0..N), if we called 'unique' on the input
        # we should get the natural numbers up to N back. This allows us to better handle multiple occurrences
        # of a given key -- instead of saying the key could occur at indices X..Y (where Y-X is the number of
        # occurrences of that specific key), we just treat those indices X..Y as the same value. And by the logic
        # above, that means that elements 0..N would occur at indices 0..N, so each element represents the index
        # it'd fall at in the sorted array.
        # Argsort the array, then subtract the value of each element (in the argsort result) from it's index;
        # this gives the "displacement" of each element from it's index if the array was fully sorted.
        # then compute the mean over the differences and return that.
        pass

    @staticmethod
    @nb.njit(cache=True)
    def lnds(arr: np.ndarray) -> float:
        len = arr.shape[0]
        L = np.zeros(len)
        L[0] = 1
        for i in range(1, len):
            for j in range(i):
                if arr[j] <= arr[i]:
                    L[i] = max(L[i], L[j] + 1)

        return np.max(L) / len

    @staticmethod
    @nb.njit(cache=True)
    def pw_diff_sum(arr: np.ndarray) -> float:
        l = arr.shape[0]
        if l < 2:
            return 0
        n = np.min(arr)
        x = np.max(arr)
        diff_sums = np.array([np.sum(np.abs(arr - a)) for a in arr])
        max_sum = (x - n) * l * (l - 1) / 2
        return np.sum(diff_sums) / max_sum

    @staticmethod
    @nb.njit(cache=True)
    def mean_seq_diff(arr: np.ndarray) -> float:
        diffs = np.diff(arr)
        m = np.mean(np.abs(diffs))
        x = np.max(diffs)
        return 1 if (m == 0 or x == 0) else 1 - m / x

    @staticmethod
    @nb.njit(cache=True)
    def mean_abs_dev(arr: np.ndarray) -> float:
        l = arr.shape[0]
        if l < 2:
            return 0
        m = np.mean(arr)
        n = np.min(arr)
        x = np.max(arr)
        abs_devs = np.abs(arr - m)
        dev = np.mean(abs_devs)
        max_dev = ((m - n) + (x - m)) / 2
        return 0 if max_dev == 0 else dev / max_dev

    @staticmethod
    @nb.njit(cache=True)
    def var_ratio(arr: np.ndarray) -> float:
        l = arr.shape[0]
        if l < 2: return 0
        return np.var(arr) * 4 / np.square(np.max(arr) - np.min(arr))

    @staticmethod
    @nb.njit(cache=True)
    def davies_bouldin(arr: np.ndarray) -> float:
        """
        Calculates the Davies-Bouldin index of a key array. This provides a measurement of "clustering".

        Parameters
        ----------
        arr : np.ndarray
            The key array to calculate the Davies-Bouldin index over.

        Returns
        -------
        float
            The calculated Davies-Bouldin index of `arr`.

        Notes
        -----
        * https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation
        * https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
        * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score

        TODO: This could be implemented more efficiently if the input were a riptable Grouping object
        instead of a raw key array.

        See Also
        --------
        ArrayMetric.dunn
        ArrayMetric.silhouette_coeff
        """
        if len(arr) == 0:
            return np.nan

        # Get the max value in the array -- this tells us how many keys there are.
        key_count = arr.max() + 1

        # Arrays to hold the sum of the indices where each key is found, and the number of occurrences of that key.
        key_index_sums = np.zeros(key_count, dtype=np.uint64)
        key_counts = np.zeros(key_count, dtype=np.uint64)

        for i in range(len(arr)):
            key = arr[i]
            key_index_sums[key] += i
            key_counts[key] += 1

        # Compute the centroid (mean index) for each key (representing a cluster).
        centroids = key_index_sums.astype(np.float64) / key_counts.astype(np.float64)
        del key_index_sums

        # For each key/cluster, determine the average (mean) distance of the points (in our case,
        # indices of the occurrences of the key) to the centroid for that key.
        # We actually just compute the total distance from each point to the centroid (per key); since we
        # know how many times each key occurs we can easily compute the average later when needed, and this
        # avoids an extra array allocation / unnecessary memory usage.
        total_dist_to_centroid = np.zeros(key_count, dtype=np.float64)
        for i in range(len(arr)):
            key = arr[i]
            # abs() is used here since we're computing a Euclidian distance (L2 norm) over 1D.
            total_dist_to_centroid[key] += abs(centroids[key] - i)

        # The summation value (added to for each key).
        summation = 0.

        # Iterate over pairs of keys (direction doesn't matter, so this is a "triangular" nested loop).
        for i in range(key_count - 1):
            at_least_one_pair_seen = False
            # The max value of the intra-vs-intra-cluster distance metric for all j values given this i value.
            max_ij_distance_ratio = np.nan

            sigma_i = total_dist_to_centroid[i] / key_counts[i]

            for j in range(i + 1, key_count):
                sigma_j = total_dist_to_centroid[j] / key_counts[j]

                # Distance over 1D (array indices) between centroids for key/cluster i and j.
                # abs() is used to calculate Euclidian distance (L2 norm) over 1D.
                ij_centroid_dist = abs(centroids[i] - centroids[j])

                # Calculate the intra-vs-inter cluster metric for (i, j).
                ij_distance_ratio = (sigma_i + sigma_j) / ij_centroid_dist

                # If this is the first trip through the loop, set the max distance ratio.
                # Otherwise, if the ratio for this (i, j) pair is larger than the max value
                # we've seen for all other j's (given the current value of i), update the max value.
                if not at_least_one_pair_seen:
                    max_ij_distance_ratio = ij_distance_ratio
                    at_least_one_pair_seen = True
                elif ij_distance_ratio > max_ij_distance_ratio:
                    max_ij_distance_ratio = ij_distance_ratio

            if at_least_one_pair_seen:
                summation += max_ij_distance_ratio

        # Return the average of the summed max-ratios.
        return summation / key_count

    @staticmethod
    @nb.njit(cache=True)
    def dunn(arr: np.ndarray) -> float:
        """
        Calculates the Dunn index of a key array. This provides a measurement of "clustering".

        Parameters
        ----------
        arr : np.ndarray
            The key array to calculate the Dunn index for.

        Returns
        -------
        float
            The calculated Dunn index of `arr`.

        Notes
        -----
        * https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation
        * https://en.wikipedia.org/wiki/Dunn_index

        TODO: This could be implemented more efficiently if the input were a riptable Grouping object
        instead of a raw key array.

        See Also
        --------
        ArrayMetric.davies_bouldin
        ArrayMetric.silhouette_coeff
        """
        if len(arr) == 0:
            return np.nan

        # Get the max value in the array -- this tells us how many keys there are.
        key_count = arr.max() + 1

        # Arrays to hold the sum of the indices where each key is found, and the number of occurrences of that key.
        key_index_sums = np.zeros(key_count, dtype=np.uint64)
        key_counts = np.zeros(key_count, dtype=np.uint64)

        for i in range(len(arr)):
            key = arr[i]
            key_index_sums[key] += i
            key_counts[key] += 1

        # Compute the minimum inter-cluster distance, where the distance is defined
        # as the Euclidian distance between the centroids of the clusters.
        min_intercluster_dist = 1e50
        for i in range(key_count - 1):
            centroid_i = np.float64(key_index_sums[i]) / key_counts[i]
            for j in range(i + 1, key_count):
                centroid_j = np.float64(key_index_sums[j]) / key_counts[j]

                # Euclidian distance over 1D between the two centroids.
                ij_centroid_dist = abs(centroid_i - centroid_j)

                # If the distance between the centroids for keys/clusters i and j
                # is smaller than the current min, update the min.
                if ij_centroid_dist < min_intercluster_dist:
                    min_intercluster_dist = ij_centroid_dist

        del key_index_sums

        # Compute the maximum intra-cluster distance.
        # There are various ways to define intra-cluster distance, and some may provide "better" results than others.
        # E.g. mean of pairwise distances between all points in a cluster.
        # See the article (linked in docstring) for details.
        # Here we use the maximum distance between any two points in the cluster;
        # we're working in 1D, so the extremes for each key/cluster are the first and
        # last occurrences of each key within the input array.
        first_key_index = np.full(key_count, len(arr), dtype=np.int64)
        last_key_index = np.zeros(key_count, dtype=np.int64)

        for i in range(len(arr)):
            key = arr[i]
            if first_key_index[key] > i:
                # This is the first occurrence of this key, so save the array index.
                first_key_index[key] = i

            # Always update the last (greatest) index at which we saw a given key.
            last_key_index[key] = i

        max_intracluster_dist = 0
        for i in range(key_count):
            intracluster_dist = last_key_index[i] - first_key_index[i]

            # Update the max intracluster distance if this cluster's distance is larger
            # than any we've seen so far.
            if intracluster_dist > max_intracluster_dist:
                max_intracluster_dist = intracluster_dist

        # The Dunn index is defined as the ratio between the minimal inter-cluster distance
        # and the maximal intra-cluster distance.
        return min_intercluster_dist / max_intracluster_dist

    @staticmethod
    def silhouette_coeff(arr: np.ndarray) -> float:
        """
        Calculates the silhouette coefficient of a key array. This provides a measurement of "clustering".

        Parameters
        ----------
        arr : np.ndarray
            The key array to calculate the silhouette coefficient for.

        Returns
        -------
        float
            The calculated silhouette coefficient of `arr`.

        Notes
        -----
        * https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation
        * https://en.wikipedia.org/wiki/Silhouette_(clustering)
        * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score

        TODO: This could be implemented more efficiently if the input were a riptable Grouping object
        instead of a raw key array.

        See Also
        --------
        ArrayMetric.davies_bouldin
        ArrayMetric.dunn
        """
        # TODO: Instead of implementing this ourselves, can just call the scikit-learn implementation (link in docstring).
        pass


def calculate_metrics(arr: np.ndarray, metrics: Mapping[str, MetricFunction]) -> Mapping[str, float]:
    """
    Given an array and a table of metric/scoring functions, calculates the metrics for the array.
    """
    return {k: f(arr) for k, f in metrics}


_metrics: Mapping[str, MetricFunction] = {
    "inversions": ArrayMetric.inversions,
    "adjacent_unsorted": ArrayMetric.adjacent_unsorted,
    "lnds": ArrayMetric.lnds,
    "pw_diff_sum": ArrayMetric.pw_diff_sum,
    "mean_seq_diff": ArrayMetric.mean_seq_diff,
    "mean_abs_dev": ArrayMetric.mean_abs_dev,
    "var_ratio": ArrayMetric.var_ratio,
    "davies_bouldin": ArrayMetric.davies_bouldin,
    "dunn_index": ArrayMetric.dunn,
    #"silhouette_coeff": ArrayMetric.silhouette_coeff
}

def score_array(arr: np.ndarray) -> Mapping[str, float]:
    """Scores an array based on a pre-defined set of metric functions."""
    return calculate_metrics(arr, _metrics)


def batch_index_gen_error(
    lo: Union[int, np.int32, np.int64],
    hi: Union[int, np.int32, np.int64],
    size: Union[int, np.int32, np.int64],
    s_metric: MetricFunction,
    c_metric: MetricFunction,
    d_metric: MetricFunction
) -> None:
    ss = np.linspace(0, 1, 6)
    cs = np.linspace(0, 1, 6)
    ds = np.linspace(0, 1, 6)
    mets = itertools.product(ss, cs, ds)
    index_count = 0
    s_error = 0
    c_error = 0
    d_error = 0
    for (s, c, d,) in mets:
        indexes = gen_index_MOO(lo, hi, size, s_metric, c_metric, d_metric, sort=s, clustered=c, dispersed=d)
        index_count += indexes.shape[0]
        s_error_curr = 0
        c_error_curr = 0
        d_error_curr = 0
        for index in indexes:
            s_exp = s
            c_exp = c
            d_exp = d

            s_obs = s_metric(index)
            c_obs = c_metric(index)
            d_obs = d_metric(index)

            s_error_curr += abs(s_exp - s_obs)
            c_error_curr += abs(c_exp - c_obs)
            d_error_curr += abs(d_exp - d_obs)

            s_error += s_error_curr
            c_error += c_error_curr
            d_error += d_error_curr

        print (f"batch errors -  s: {s_error_curr/indexes.shape[0]}    c: {c_error_curr/indexes.shape[0]}   d: {d_error_curr/indexes.shape[0]}    # indexes: {indexes.shape[0]}")

    s_error /= index_count
    c_error /= index_count
    d_error /= index_count

    print (f"s_metric avg error: {s_error}    c_metric avg error: {c_error}     d_metric avg error: {d_error}")

class SCDArrayGen(Problem):
    def __init__(self, lo, hi, index_size, s_metric, c_metric, d_metric, s, c, d):
        self.lo = lo
        self.hi = hi
        self.index_size = index_size
        self.s_metric = s_metric
        self.c_metric = c_metric
        self.d_metric = d_metric
        self.s = s
        self.c = c
        self.d = d

        xl = np.full(index_size, self.lo)
        xu = np.full(index_size, self.hi)

        super().__init__(n_var=self.index_size, n_constr=2, xl=xl, xu=xu, type_var=np.int)

    def s_error(self, arr):
        return abs(self.s_metric(arr) - self.s)

    def c_error(self, arr):
        return abs(self.c_metric(arr) - self.c)

    def d_error(self, arr):
        return abs(self.d_metric(arr) - self.d)

    def _evaluate(self, x, out, *args, **kwargs):

        #objectives
        se = np.apply_along_axis(self.s_error, 1, x)
        ce = np.apply_along_axis(self.c_error, 1, x)
        de = np.apply_along_axis(self.d_error, 1, x)

        #constraints
        g1 = self.lo - x
        g2 = x - self.hi

        out["F"] = np.column_stack([se, ce, de])
        out["G"] = np.column_stack([g1, g2])


# Evolutionary, Black-Box, Multi-Objective Optimization Algorithm
def gen_index_MOO(lo, hi, size, s_metric ,c_metric, d_metric, dist=None, sort=0.0, clustered=0.0, dispersed=0.0):
    max_error = 0.25
    prob = SCDArrayGen(lo, hi, size, s_metric, c_metric, d_metric, sort, clustered, dispersed)
    alg = NSGA2(
        pop_size=100,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_two_point"),
        mutation=get_mutation("int_pm"),
    )
    res = minimize(prob, alg, seed=3)

    #optionally screen out solutions that still deviated from the desired values by too much
    # mask = res.F < max_error
    # masksum = np.sum(mask, axis=1) == 3
    return res.X

if __name__ == "__main__":
    # test
    #print(gen_index_MOO(0, 150, 500, ArrayMetric.inversions, ArrayMetric.pw_diff_sum, ArrayMetric.mean_abs_dev, sort=0.5, clustered=0.2, dispersed=0.8))
    lo = 0
    hi = 10000
    size=100
    s_metric = ArrayMetric.inversions
    c_metric = ArrayMetric.pw_diff_sum
    d_metric = ArrayMetric.mean_abs_dev
    batch_index_gen_error(lo, hi, size, s_metric, c_metric, d_metric)



