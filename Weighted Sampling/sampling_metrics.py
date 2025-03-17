# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Define functions for computing variance and KL divergence
def compute_variance(category_distribution):
    """
    Computes the variance of the category proportions.
    
    Parameters:
    - category_distribution (dict): {category: proportion}
    
    Returns:
    - float: Variance of proportions
    """
    proportions = np.array(list(category_distribution.values()))
    mean_p = np.mean(proportions)
    variance = np.mean((proportions - mean_p) ** 2)
    return variance

def compute_kl_divergence(category_distribution):
    """
    Computes KL divergence from a uniform distribution.
    
    Parameters:
    - category_distribution (dict): {category: proportion}
    
    Returns:
    - float: KL divergence value
    """
    proportions = np.array(list(category_distribution.values()))
    num_categories = len(category_distribution)
    uniform_prob = 1 / num_categories

    # Compute KL divergence (ignoring zero probabilities to avoid log(0))
    kl_div = np.sum(proportions * np.log(proportions / uniform_prob), where=(proportions > 0))
    return kl_div

# Re-define weight assignment functions
def assign_weights_least_frequency(data, categories):
    num_samples = data.shape[0]
    category_counts = {cat: 0 for cat in categories}

    # Count occurrences of each category in the dataset
    for row in data:
        for cat in row:
            category_counts[cat] += 1

    # Compute weights using the least frequent category in each sample
    weights = np.zeros(num_samples)
    for i, row in enumerate(data):
        min_frequency = min(category_counts[cat] for cat in row)
        weights[i] = 1 / min_frequency

    # Normalize weights
    weights /= np.sum(weights)
    return weights

def assign_weights_sum_inverse_frequency(data, categories):
    num_samples = data.shape[0]
    category_counts = {cat: 0 for cat in categories}

    # Count occurrences of each category in the dataset
    for row in data:
        for cat in row:
            category_counts[cat] += 1

    # Compute weights using the sum of inverse frequencies of categories in each row
    weights = np.zeros(num_samples)
    for i, row in enumerate(data):
        row_weight = sum(1 / category_counts[cat] for cat in row)
        weights[i] = row_weight
    
    # Normalize weights
    weights /= np.sum(weights)
    return weights

def weighted_sampling(data, weights, sample_size, replacement):
    """
    Performs weighted random sampling on the dataset.

    Parameters:
    - data (np.ndarray): The dataset where each row contains multiple categories.
    - weights (np.ndarray): Weights for each sample.
    - sample_size (int): Number of samples to select.
    - replacement (bool): Sampling with or without replacement.

    Returns:
    - np.ndarray: Sampled subset of the dataset.
    """
    indices = np.random.choice(len(data), size=sample_size, replace=replacement, p=weights)
    return data[indices]

# Generate synthetic dataset with 100 samples and 10 unique categories
np.random.seed(42)
categories = [f"Cat_{i}" for i in range(10)]
dataset = np.array([np.random.choice(categories, size=np.random.randint(2, 4), replace=False) for _ in range(100)], dtype=object)

# Compute original category distribution
original_counts = Counter(cat for row in dataset for cat in row)
total_original = sum(original_counts.values())
original_distribution = {cat: original_counts[cat] / total_original for cat in categories}

# Compute weights using both methods
weights_least = assign_weights_least_frequency(dataset, categories)
weights_sum = assign_weights_sum_inverse_frequency(dataset, categories)

# Sample data using both methods (with replacement)
sample_size = 50
num_trials = 1000

sampled_counts_least = Counter()
sampled_counts_sum = Counter()

for _ in range(num_trials):
    sampled_data_least = weighted_sampling(dataset, weights_least, sample_size, replacement=True)
    sampled_data_sum = weighted_sampling(dataset, weights_sum, sample_size, replacement=True)

    for row in sampled_data_least:
        sampled_counts_least.update(row)
    for row in sampled_data_sum:
        sampled_counts_sum.update(row)

# Normalize sampled distributions
total_sampled_least = sum(sampled_counts_least.values())
total_sampled_sum = sum(sampled_counts_sum.values())

sampled_distribution_least = {cat: sampled_counts_least[cat] / total_sampled_least for cat in categories}
sampled_distribution_sum = {cat: sampled_counts_sum[cat] / total_sampled_sum for cat in categories}

# Compute variance and KL divergence for original and sampled distributions
variance_original = compute_variance(original_distribution)
variance_least = compute_variance(sampled_distribution_least)
variance_sum = compute_variance(sampled_distribution_sum)

kl_original = compute_kl_divergence(original_distribution)
kl_least = compute_kl_divergence(sampled_distribution_least)
kl_sum = compute_kl_divergence(sampled_distribution_sum)

# Display results
{
    "Variance": {
        "Original": variance_original,
        "Least-Frequency Sampling": variance_least,
        "Sum-Inverse-Frequency Sampling": variance_sum
    },
    "KL Divergence": {
        "Original": kl_original,
        "Least-Frequency Sampling": kl_least,
        "Sum-Inverse-Frequency Sampling": kl_sum
    }
}
