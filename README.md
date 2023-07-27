# Realtime Sentiment Analysis using Spark

This repository contains the implementation of a study on sentiment analysis using Spark Streaming. The goal is to explore the performance of multiple classification and clustering online/incremental learning ML models on a streaming corpus of tweets and their corresponding sentiment labels. The models are designed to learn on batches of data streamed over time, and the repository provides detailed analysis and evaluation of the models based on various performance metrics.

## About the Dataset
The dataset used for this study consists of two CSV files: one for training (1520k records) and another for testing (80k records). Each record in the dataset contains two columns: one for the sentiment label (0 for negative, 4 for positive) and the other for the tweet text.

## Task Workflow
The repository follows the following workflow for sentiment analysis with Spark Streaming:

1. **Streaming the Data**: Spark Streaming is used to ingest the streaming corpus of tweets and their corresponding sentiment labels.

2. **Cleaning and Preprocessing**: Each RDD (Resilient Distributed Dataset) of input data is cleaned and preprocessed to remove any noise, irrelevant information, or unwanted characters from the tweet text.

3. **Online/Incremental Training of Classification Models**: The repository includes implementations of various online/incremental learning classification models. These models are trained on batches of data as they are streamed in real-time.

4. **Online/Incremental Training of Clustering Models**: In addition to classification, the repository also explores online/incremental learning clustering models. These models are trained on the streaming data and can dynamically adapt to new clusters as more data arrives.

5. **Testing the Models**: The trained classification and clustering models are tested against cleaned and preprocessed RDDs of the test data stream. This allows us to evaluate the models' performance on unseen data and assess their effectiveness in real-time scenarios.

6. **Plotting Graphs**: To facilitate analysis and evaluation, the repository includes code to generate graphs and visualizations of the models' performance metrics. These graphs illustrate trends and insights based on varying hyperparameter and streaming batch size values.
