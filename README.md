## Project: Resume-Job Description Matching Engine 🤖
Introduction
This project is a data analysis and machine learning pipeline that builds a model to predict the compatibility score between a resume and a job description. The goal is to demonstrate the end-to-end process of a data science project, from exploratory data analysis (EDA) and text preprocessing to model building and evaluation. The final model serves as a proof-of-concept for an automated recruitment screening tool.

Dataset
The dataset used in this project is the Resume vs. Job Description Matching Dataset from Kaggle. It is a synthetic dataset containing 10,000 entries with the following columns:

job_description: A text description of a job posting.

resume: A text summary of a candidate's profile.

match_score: A numerical score (1-5) indicating the quality of the match.

Methodology
The project followed a standard data science workflow:

Data Loading & EDA: The data was loaded using pandas. Initial exploration confirmed the dataset had 10,000 entries with no missing values. The match_score distribution was found to be skewed towards higher scores (3 and 4), and the character counts of both resumes and job descriptions followed a consistent normal distribution.

Text Preprocessing: The raw text data was cleaned to make it suitable for a machine learning model. This involved:

Lowercasing the text.

Removing punctuation and special characters.

Removing common stopwords (e.g., "the", "a", "is").

Stemming words to their root form (e.g., "needed" became "need").

Feature Engineering: The preprocessed text was converted into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique assigns a numerical weight to each word, representing its importance in a document relative to the entire dataset. The resulting feature matrix had a shape of (20000, 869).

Modeling: A Random Forest Regressor model was trained to predict the match_score based on the TF-IDF features. The data was split into training and testing sets to ensure a robust evaluation.

Results
The model's performance was evaluated using two key regression metrics:

Mean Absolute Error (MAE): 0.99

R-squared (R 
2
 ): -0.02

The results indicate that the current model is not highly accurate. An MAE of 0.99 means the model's predictions are, on average, off by almost a full point on a 1-5 scale. The negative R 
2
  score suggests the model performs worse than simply predicting the average score for all data points.

Future Work & Improvements
The model's performance highlights several opportunities for improvement, which are excellent avenues for further development:

Advanced NLP: Use more sophisticated techniques like Word Embeddings (Word2Vec) or Sentence Transformers (BERT) to capture the semantic meaning of the text, rather than just word frequency.

Hyperparameter Tuning: Optimize the RandomForestRegressor by tuning its parameters (e.g., n_estimators, max_depth).

Additional Features: Engineer new features based on the number of overlapping skills or keywords between the resume and job description.

Explore Alternative Models: Test other regression models, such as XGBoost or a simple neural network, to see if they can better capture the relationship in the data.

Technologies Used
Python

Pandas for data manipulation.

NLTK for text preprocessing.

Scikit-learn for TF-IDF vectorization and machine learning models.

Matplotlib & Seaborn for data visualization.

Jupyter Notebook for the interactive workflow.






