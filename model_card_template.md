# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The project implements a supervised machine learning model to predict whether an individual's income exceeds $50,000 per year based on demographic and employment-related features. The model is a Random Forest classifier implemented using the scikit-learn library. 

## Intended Use
The intended use of this model is to serve as a predictive tool for income classification based on census style demographic data. This model was created for use in course from Udacity. 

## Training Data
The model was trained on the UCI Census Income dataset, which contains demographic and employment information such as age, education, occupation, marital status, and hours worked per week. 

Cateogorical features were processed using one-hot encoding. The target variable was named salary and split into two classes: <=50K and >50K. The dataset was split into training and testing subsets using an 80/20 split.

## Evaluation Data
The evaluation data consists of the 20% holdout test set created using a random train-test split with a fixed random state for reproducibility. The same preprocessing steps applied to the training data were reused for the test data using the fitted encoder and label binarizer. 

In addition to overall evaluation, model performance was assessed across slices of the data defined by categorical feature values to better understand subgroup performance. 

## Metrics
The model was evaluated using the following metrics:
Precision: The proportion of positive correct predictions
Recall: The proportion of actual positives that are correctly identified
F1 Score: The harmonic mean of precision and recall

Results
Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

## Ethical Considerations
This model is trained based on many sensitive attributes and protected classes. For example; race, sex, and nationality all fall in one of those categories. The use of this model could learn the biases of those classes and perpetuate them futher. 

The slice-based evaluation showed that there was variability in performance across different subgroups so care should be taken to evaluate and mitigate any bias before deploying the model in a real world scenario. 


## Caveats and Recommendations
The model has some limitations. The dataset itself is dated, performance varying across categorical slices could be a major issues, and cross-validation was not used. That could add a fuller evaluation. 

Recommendations for this model would be to conduct a deeper analysis of underperforming slices, use more recent and diverse datasets, and implement cross-validation. 
