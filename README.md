# Collaborative Filtering for Netflix Movies Ratings

The goal of this project is to build a mixture model for collaborative filtering. The project uses a data matrix containing movie ratings made by users, where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies, so the data matrix is only partially filled. The goal is to predict all the remaining entries in the matrix.

This project will employ two models for collaborative filtering: Matrix Factorization and Gaussian Mixture Models (GMMs).

To evaluate the performance of these models, we will use the Root Mean Squared Error (RMSE) metric. RMSE is a standard measure for evaluating the accuracy of predicted ratings compared to the actual ratings, and it is particularly useful for identifying the magnitude of prediction errors in the context of collaborative filtering.
