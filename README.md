# Collaborative Filtering with Netflix Movie Ratings

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/NumPy-1.19+-green.svg" alt="NumPy 1.19+"/>
  <img src="https://img.shields.io/badge/Pandas-1.2+-orange.svg" alt="Pandas 1.2+"/>
  <img src="https://img.shields.io/badge/SciPy-1.6+-yellow.svg" alt="SciPy 1.6+"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.3+-red.svg" alt="Matplotlib 3.3+"/>
</div>

## 🎬 Introduction

This project implements an advanced recommendation system using Collaborative Filtering techniques to predict Netflix movie ratings. By leveraging Matrix Factorization and Gaussian Mixture Models (GMMs), we are able to accurately predict how users would rate movies they haven't watched yet, effectively solving the core problem behind recommendation systems.

The project addresses a simplified version of the famous Netflix Prize challenge, where the goal is to fill a partially complete user-movie rating matrix by predicting missing ratings through sophisticated machine learning models.

## 📊 Problem Statement

In a recommendation system, we typically have:
- A large number of users
- A large catalog of items (movies in this case)
- A sparse matrix of ratings (most users rate only a small subset of movies)

Our task is to predict the missing entries in this sparse matrix to enable movie recommendations for users.

## 🔍 Models Implemented

### 1. Matrix Factorization (MF)

Matrix Factorization decomposes the ratings matrix X into the product of two lower-dimensional matrices:

```
X ≈ U × V^T
```

Where:
- U: User latent factor matrix (n × k)
- V: Movie latent factor matrix (d × k)
- k: Number of latent factors (much smaller than n or d)

The optimization objective is to minimize:

```
minimize ||X - UV^T||^2 + λ(||U||^2 + ||V||^2)
```

Where λ is a regularization parameter to prevent overfitting.

### 2. Gaussian Mixture Models (GMMs)

GMMs model the probability distribution of user ratings as a mixture of Gaussian distributions:

```
P(x|θ) = ∑(j=1 to K) πj𝒩(x; μ^(j), σj^2I)
```

Where:
- θ: All parameters in the mixture
- μ^(j): Mean of the jth Gaussian component
- σj^2: Variance of the jth Gaussian component
- πj: Mixing coefficient of the jth component

The parameters are estimated using the Expectation-Maximization (EM) algorithm, which iteratively:
1. Computes the probability of each user belonging to each cluster (E-step)
2. Updates the parameters of each Gaussian component (M-step)

## 📈 Results & Comparison

Both models were evaluated using Root Mean Squared Error (RMSE) on a test dataset:

| Model | RMSE |
|-------|------|
| Matrix Factorization | 0.5026 |
| Gaussian Mixture Model | 0.5012 |

Key observations:
- Both models achieved similar performance
- MF is simpler to implement and computationally more efficient
- GMM provides a probabilistic interpretation but requires more computational resources
- Both models converge to local optima, hence multiple random initializations are recommended

## 🛠️ Technical Implementation

The implementation consists of:

1. **Data Preprocessing**:
   - Loading the sparse rating matrix
   - Analyzing sparsity patterns
   - Splitting data for training and validation

2. **Matrix Factorization**:
   - Implementing alternating optimization for U and V
   - Using gradient descent with regularization
   - Monitoring convergence through loss values

3. **Gaussian Mixture Model**:
   - Implementing the EM algorithm
   - Calculating log-likelihoods and posteriors
   - Determining optimal number of clusters

4. **Rating Prediction**:
   - Filling missing values in the rating matrix
   - Rounding predictions to the nearest integer (1-5 scale)
   - Comparing with true values

## 📂 Project Structure

```
Netflix/
│
├── data/
│   ├── netflix_incomplete.txt    # Sparse ratings matrix
│   └── netflix_complete.txt      # Complete matrix for validation
│
├── netflix.ipynb                 # Main implementation notebook
└── README.md                     # This documentation
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Netflix.git
   cd Netflix
   ```

2. Install required dependencies:
   ```bash
   pip install numpy pandas scipy matplotlib
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run `netflix.ipynb`

## 💡 Key Insights

- Collaborative filtering can effectively predict user preferences even with sparse data
- Matrix Factorization provides an efficient approach with good performance
- The choice between MF and GMM depends on the specific application requirements:
  - MF is preferred for simpler implementations and better interpretability
  - GMM is suitable when probabilistic interpretation is required


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
