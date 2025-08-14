## CS5293sp23 – Project3

## Name: Likhitha Reddy Gundre

## Project Description:
This project aims to process PDF files provided via the command line. It extracts the raw text from each page of the PDF files and applies cleaning procedures to eliminate punctuation and identify and exclude any blank or corrupted PDFs. The cleaned text is then transformed into vectors, taking into account the corresponding city information. The project subsequently carries out K-Means, Hierarchical, and DBSCAN clustering algorithms on the vectorized data, computing silhouette, Davies, and Calinski values for each clustering approach. To determine the most suitable value of k for each algorithm, a range of k values from 2 to 51 is considered. Finally, the project generates a TSV file containing the obtained results, organized in a tab-separated format.

## How to install:
pipenv installation: sudo -H pip install -U pipenv
pypdf installation: pipenv install pypdf
pandas installation: pipenv install pandas
pytest installation: pipenv install pytest
sklearn installation: pipenv install scikit-learn
nltk installation: pipenv install nltk


## How to run:
To run the project: pipenv run python project3.py --document "CA FRENSO.pdf"
To run the pytests: pipenv run python -m pytest


# Video:


https://user-images.githubusercontent.com/114453047/236927616-abb129ee-a9cf-4f38-b6f8-26f083585960.mp4




## Files
# project3.ipynb
This file initially reads all the PDF files located in the "smartcity" folder. It extracts the city names and raw text from each PDF and stores them in a dataframe.

Next, the data is cleaned by removing numbers, punctuations, and stopwords. Tokenization and lemmatization techniques are applied to the words, and the cleaned text is added to the existing dataframe. Additionally, any corrupted files are removed from the dataframe.

The indexes of the dataframe are then reset to ensure consistent indexing.

On the cleaned data, the project performs K-Means, Hierarchical, and DBSCAN clustering. It calculates the Calinski, Silhouette, and Davies scores for the K-Means and Hierarchical algorithms using predefined k values of 9, 18, and 36.

The project proceeds to determine the optimal k value for each clustering algorithm by iterating over a range of k values and evaluating their corresponding Calinski, Silhouette, and Davies scores.

The resulting model is saved as "model.pkl".

Finally, the output data is saved as a file named "smartcity_eda.tsv".



# project3.py
## Functions

input_files(args): This function receives the args parameter from the command line, specifically the files specified with the --document tag. It retrieves all the PDF files listed in the "smartcity" folder and returns those files.

create_Dataframe(files): This function takes the files parameter and reads each document page by page. It creates a dataframe based on the extracted information and returns that dataframe.

cleanPDF(df): This function takes the dataframe df returned by the previous method and performs data cleaning on the PDF data. It removes punctuations, and applies tokenization and lemmatization to the words, resulting in cleaned text. The cleaned text is then appended to the existing dataframe. The function also identifies and removes cities with corrupted data, such as those containing an excessive number of images and tables that the pypdf library cannot process. Rows with blank cleaned text are eliminated, and the index of the dataframe is reset to ensure consistency. Finally, the cleaned dataframe is returned as the output of the function

performClustering(cleaned_df): The function performClustering(cleaned_df) accepts the cleaned dataframe cleaned_df as input. It proceeds to perform K-Means and Hierarchical clustering using predefined k values of 8, 16, and 36. Silhouette, Davies, and Calinski scores are computed for both clustering algorithms. The function then iterates over a range of k values from 2 to 51 in order to identify the optimal k values for K-Means, Hierarchical, and DBSCAN algorithms. For each optimal k value, it calculates the corresponding silhouette, Davies, and Calinski scores for all three algorithms and stores the results in a dictionary. Finally, the function returns the dictionary containing the computed scores.

calculateClusterId(optimal_values, cleaned_df): This function takes the optimal_values dictionary and the cleaned dataframe cleaned_df as input. It finds the cluster ID by comparing the silhouette values of the optimal k values for all three algorithms. The cluster ID is determined based on the algorithm with the highest silhouette value. The cluster ID is added to the cleaned dataframe, and the updated dataframe is returned.

create_TsvFile(result_df): This function accepts the final updated dataframe result_df and generates a TSV (Tab-Separated Values) file.


## Bugs And Assumptions
# Bugs:
* It is not giving any value for k=36.
# Assumptions:
- All files are pdf files and the files are present in smartcity folder.

