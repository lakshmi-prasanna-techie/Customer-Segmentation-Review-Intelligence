# Customer Segmentation & Review Intelligence

Modular machine learning pipeline combining **RFM behavioral metrics**, **review text NLP** (lexicon sentiment, complaint rates, and TF-IDF + TruncatedSVD topics), **K-Means clustering** (validated via Silhouette and Davies-Bouldin scores), **2D PCA** centroid visualization, and **XGBoost** segment classification/scoring.

## Project Structure
- config.py — hyperparameters, paths, and NLP/feature definitions
- data_loader.py — CSV loader and multi-persona synthetic dataset generator
- nlp_features.py — lexicon sentiment, complaint rate, and TF-IDF + TruncatedSVD features
- clustering.py — scaling, K-Means validation, and business persona mapping
- classifier.py — stratified split, 5-fold cross-validation, and XGBoost training
- viz.py — cluster validation curve, 2D PCA centroid plot, and XGBoost feature importance
- score.py — loads saved bundle to score new incoming customers
- main.py — runs end-to-end training, evaluation, and artifact export

## How to Run
    pip install -r requirements.txt
    python main.py
    python score.py

## Generated Outputs (artifacts/)
- pca_segments_2d.png — 2D PCA projection with cluster centroids and persona labels
- cluster_validation.png — Silhouette and Davies-Bouldin scores across cluster counts
- xgb_feature_importance.png — feature importance chart for segment classification
- pipeline_metrics.json — summary metrics and persona action playbooks
- segmented_customers.csv & segmentation_bundle.joblib — segmented data and saved pipeline bundle
