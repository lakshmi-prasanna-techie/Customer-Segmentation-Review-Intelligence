
# Customer Segmentation & Review Intelligence

## About the Project

An intelligent customer analytics system that groups customers based on their purchasing behaviour and reviews. It uses K-Means clustering to identify customer segments, silhouette scores to measure clustering quality, and PCA to visualize the groups. XGBoost helps analyse customer patterns and preferences, enabling businesses to improve customer experience and make better data-driven decisions.

## Objectives

- Identify customer segments based on purchasing behaviour.
- Analyse customer reviews and sentiment patterns.
- Evaluate clustering performance using silhouette scores.
- Visualize customer segments using PCA.
- Use XGBoost for customer pattern analysis.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- K-Means Clustering
- PCA
- XGBoost
- TF-IDF
- NLP

## Project Workflow

1. Data Loading
2. Data Preprocessing
3. Review Feature Extraction
4. Customer Segmentation using K-Means
5. Clustering Evaluation using Silhouette Score
6. PCA-based Visualization
7. Customer Analysis using XGBoost

## Project Structure

```text
Customer-Segmentation-Review-Intelligence/
│
├── customer_segmentation/
│   ├── classifier.py
│   ├── clustering.py
│   ├── config.py
│   ├── data_loader.py
│   ├── main.py
│   ├── nlp_features.py
│   ├── pipeline.py
│   ├── score.py
│   └── viz.py
│
├── README.md
└── requirements.txt
```

## Results

Model evaluation results and visualizations will be added after implementation.

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Run the main Python file.

```bash
pip install -r requirements.txt
python -m customer_segmentation.main
```
