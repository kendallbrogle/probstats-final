# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Set up plot style
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Number one
# PART 1a

# Simulating a Bernoulli random variable
p = 0.5  
n = 1000  
bernoulli_data = np.random.choice([0, 1], size=n, p=[1-p, p])

plt.hist(bernoulli_data, bins=[0, 1, 2], align='left', edgecolor='#7E00A7',linewidth=1.2, rwidth=0.8, density=True, alpha=0.5, color = '#DCB7FF')
plt.xticks([0, 1])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Bernoulli Random Variable')
plt.show()

#Simulating a binomial distribution
# Define parameters
n = 10 # number of trials
p = 0.5 # probability of success

# Simulate binomial distribution

# Define parameters
n = 15 # number of trials
p = 0.5 # probability of success

samples = np.random.binomial(n, p, size=10000)

# Plot histogram of samples
plt.hist(samples, bins=np.arange(n+2)-0.5, density=True, alpha=0.5,
         edgecolor='#00BFC1', linewidth=1.2, color='#B7FEFF')
plt.xlabel('Number of successes', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.title('Binomial Distribution with n=10 and p=0.5', fontsize=16)
plt.xticks(np.arange(n+1))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(axis='y', alpha=0.75)
plt.show()

#Simulate geometric distribution
# Define parameters
p = 0.3 # probability of success


samples = np.random.geometric(p, size=10000)

# Plot histogram of samples
plt.hist(samples, bins=np.arange(np.max(samples))+0.5, density=True, alpha=0.5,
         edgecolor='#00A25A', linewidth=1.2, color='#86bf91')
plt.xlabel('Number of trials until first success', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.title('Geometric Distribution with p=0.3', fontsize=16)
plt.xticks(np.arange(np.max(samples)+1))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(axis='y', alpha=0.75)

# Add text annotation for expected value and standard deviation
mean = 1/p
std = np.sqrt((1-p)/(p**2))
plt.text(0.98, 0.95, f'E(X) = {mean:.2f}\nSD = {std:.2f}',
         transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='#7fb3d5', alpha=0.5))

plt.show()

# Simulate Poisson distribution
# Define parameter
lam = 10 # rate parameter
samples = np.random.poisson(lam, size=10000)

# Plot histogram of samples
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(samples, bins=np.arange(16)-0.5, density=True, alpha=0.5,
         edgecolor='#D700C1', linewidth=1.2, color='#FF6DF0')
ax.set_xlabel('Number of events')
ax.set_ylabel('Probability')
ax.set_title(f'Poisson Distribution with lambda={lam}')
ax.set_xticks(np.arange(16))
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(axis='y', alpha=0.75)

plt.show()


# PART 1B
# Simulating a Gaussian random variable
mu = 0  # Mean
sigma = 1  # Standard deviation
n = 1000  # Number of samples
gaussian_data = np.random.normal(mu, sigma, n)

# Plotting a histogram and PDF for the Gaussian random variable
plt.hist(gaussian_data, bins=30, density=True, alpha=0.7, color = '#EC985E',edgecolor='#A25C2E',  linewidth=1.2, label='Histogram')
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r', label='PDF', color = '#800080')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Gaussian Random Variable Visualization')
plt.legend()
plt.show()

# Number Two

# Simulating a sequence of coin tosses
p = 0.5  # Probability of heads
n = 1000  # Number of trials
coin_tosses = np.random.choice([0, 1], size=n, p=[1-p, p])

# Calculating the cumulative mean after each trial
cumulative_mean = np.cumsum(coin_tosses) / np.arange(1, n+1)

# Plotting the graph of sample means
plt.plot(np.arange(1, n+1), cumulative_mean)
plt.axhline(p, color='#9D5B76', linestyle='--', label='Expected Value')
plt.xlabel('Number of Trials')
plt.ylabel('Sample Mean')
plt.title('Law of Large Numbers Visualization')
plt.legend()
plt.show()

#Number Three
# Define the population distribution
pop_mean = 100  # population mean
pop_std = 20    # population standard deviation
population = np.random.normal(pop_mean, pop_std, size=100000)

# Define the sample size and number of samples
sample_size = 50
num_samples = 10000

# Simulate the sample means
sample_means = np.mean(np.random.choice(population, size=(num_samples, sample_size)), axis=1)

# Compute the mean and standard deviation of the sample means
sample_mean = np.mean(sample_means)
sample_std = np.std(sample_means, ddof=1)

# Compute the normal approximation
norm_mean = pop_mean
norm_std = pop_std / np.sqrt(sample_size)
norm = norm.pdf(np.linspace(np.min(sample_means), np.max(sample_means), num=num_samples), loc=norm_mean, scale=norm_std)

# Create the figure and plot the histograms
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(sample_means, bins=50, alpha=0.5, density=True, edgecolor='#7C3E36',
        linewidth=1.2, color='#D85342', label='Sample Means')
ax.plot(np.linspace(np.min(sample_means), np.max(sample_means), num=num_samples), norm, linewidth=3,
        color='#f9c74f', label='Normal Approximation')

# Add labels and legend
ax.set_xlabel('Sample Mean', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title(f'n = {sample_size}\nMean = {sample_mean:.2f}, Std = {sample_std:.2f}', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()

# Set the title of the figure
fig.suptitle('Central Limit Theorem', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()




