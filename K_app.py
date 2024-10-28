# kmeans_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title and description
st.title("K-Means Clustering App")
st.write("This app uses K-Means Clustering to group customers based on their annual income and spending score.")

# Load dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(dataset.head())
    
    # Select features for clustering
    X = dataset.iloc[:, [3, 4]].values  # Adjust indices if necessary based on uploaded file structure
    
    # Elbow method to find optimal clusters
    st.write("Using the Elbow Method to determine optimal number of clusters...")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Plot WCSS
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)
    
    # Get number of clusters from the user
    n_clusters = st.slider("Select the number of clusters", min_value=1, max_value=10, value=5)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    
    # Add cluster data to DataFrame
    dataset['Cluster'] = y_kmeans
    st.write("Data with Cluster Labels:")
    st.write(dataset.head())
    
    # Plot clusters
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'pink']
    for i in range(n_clusters):
        ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    ax.set_title('Clusters of Customers')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    st.pyplot(fig)
