# PRODIGY_ML_02

# Customer Segmentation using K-means Clustering

This repository contains code to perform customer segmentation using K-means clustering algorithm. The code is implemented in a Google Colab notebook and uses the Mall Customers dataset.

## Introduction

Customer segmentation is the process of dividing customers into groups based on common characteristics. K-means clustering is a popular unsupervised machine learning algorithm used for segmentation tasks. In this project, we use K-means clustering to segment customers based on their annual income and spending score.

## Dataset

The dataset used in this project is the Mall Customers dataset, which contains information about customers including their annual income and spending score. The dataset is available on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

## Code

The code is implemented in a Google Colab notebook (`Customer_Segmentation_Kmeans.ipynb`). It performs the following steps:

1. Load the dataset from Google Drive.
2. Preprocess the data (if necessary).
3. Scale the features using StandardScaler.
4. Determine the optimal number of clusters using the Elbow Method.
5. Apply K-means clustering with the optimal number of clusters.
6. Visualize the clusters.

## Usage

To run the code:

1. Upload the dataset file (`Mall_Customers.csv`) to your Google Drive.
2. Open the Colab notebook (`Customer_Segmentation_Kmeans.ipynb`) in Google Colab.
3. Mount your Google Drive and specify the path to the dataset file.
4. Run the notebook cells sequentially.

## Dependencies

The code requires the following Python libraries:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

These dependencies can be installed using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
