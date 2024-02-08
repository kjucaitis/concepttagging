# Concepts2Metrics
A data science internship at a market research company, throughout which I collaborated with Igli Kristo on 'Concepts2Metrics,' where we extracted data-driven insights from product concepts.
# Table of Contents
# Introduction
The project relies on a dataset characterized by a restricted data volume, comprising merely 717 rows. Dealing with small datasets can pose challenges, including issues such as overfitting and restricted generalizability. To prepare the data for the project, we did a bit of cleaning.
## Dataset
![Source](https://github.com/kjucaitis/concepttagging/assets/142523963/9282066d-0a17-4705-a363-45dfe11df68e)
## Cleaning
![Cleaning](https://github.com/kjucaitis/concepttagging/assets/142523963/c6aaada1-27ab-4499-8ab0-a9cdf4ddd201)
# Analysis
We conducted initial analysis, which allowed us to see that unlike the other metrics, 'Distinctiveness' had a negative correlation and thus had to be treated differently.
## Seaborn Plot
![EDA](https://github.com/kjucaitis/concepttagging/assets/142523963/aa736d24-a7e3-4279-b916-4e313b8fb085)
## Correlation Heatmap
![corr heatmap](https://github.com/kjucaitis/concepttagging/assets/142523963/012069f3-0aec-468a-a710-07b6a41c5ca8)
## Principal Component Analysis
![PCA analysis](https://github.com/kjucaitis/concepttagging/assets/142523963/c6299ce3-7a5c-4f9c-803a-617bee83d791)
## Feature Engineering
### Feature engineering encompasses dimensionality reduction through the application of Principal Component Analysis (PCA). The objective of the project is to eliminate noise and irrelevant features, potentially enhancing the accuracy of identified clusters.
![FeatureEngineering](https://github.com/kjucaitis/concepttagging/assets/142523963/aa978519-f6ae-4d0c-b0d1-71046fcd43b0)
# Modeling
## XGBoost
### For our project, we used XGBoost, which is a powerful ensemble learning algorithm, widely used for classification and regression tasks due to its exceptional performance and ability to handle complex relationships. It utilizes a gradient boosting framework and regularization techniques to improve accuracy, prevent overfitting, and enhance generalization.
![Creation](https://github.com/kjucaitis/concepttagging/assets/142523963/a2c58bc8-cfc1-4a5c-8f17-e553cba20a0f)
# Model Results
![Results1](https://github.com/kjucaitis/concepttagging/assets/142523963/eb2f8bcb-52f1-41a6-ac65-6ab42466ab6d)
# Reflections 
## Struggles
* Challenges in mastering XGBoost and machine learning
* Dealing with a small dataset prone to overfitting
* Too much focus on unnecessary details
* Inclusion of duplicate tags in features
## Successes
* Effective task distribution
* Clear and efficient communication within the team
* Learning and improvement throughout the project
## Future Improvements
* Continue refining the feature engineering process
* Consider data augmentation techniques to expand the dataset and improve results
