import argparse
import os
import sys
import pypdf
import glob
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def input_files(args):
    folder_path = 'smartcity/'
    files = []

    if not args.document:
        print("File not present", file=sys.stderr)
        sys.exit(0)
    else:
        file_path = os.path.join(folder_path, args.document[0].strip("'"))
        print(file_path)
        files = glob.glob(file_path)
        print(files)

    if not files:
        print("Text file not present", file=sys.stderr)
        sys.exit(0)

    return files

def create_dataframe (files):
    docText = []
    city_names = []
    for filename in files:
        if filename.endswith('.pdf'):
            pdf_file = open(filename, 'rb')
            pdf_reader = pypdf.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader._get_page(page_num)
                text = page.extract_text()
                docText.append(text)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                city_name = base_name.split('_')[0]
                city_names.append(city_name)
    df = pd.DataFrame({'City': city_names, 'raw_text': docText})
    return df

def cleanPDF(df):
    sw = set(stopwords.words('english')).union(['city', 'cities', 'state', 'states', 'page', 'smart'])
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for text in df['raw_text']:
        text = re.sub(r'[\d\W]+', ' ', text)  # Remove numbers and punctuation
        words = word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word not in sw]
        cleaned_text.append(' '.join(words))
    df['Cleaned_Text'] = cleaned_text
    cities_to_remove = ['OH Toledo', 'CA Moreno Valley', 'TX Lubbock', 'NV Reno', 'FL Tallahassee', 
                        'NY Mt Vernon Yonkers New Rochelle', 'VA Newport News']
    df = df[~df['City'].isin(cities_to_remove)]
    df = df[df['Cleaned_Text'] != '']
    df.reset_index(drop=True, inplace=True)
    return df


def performClustering(cleaned_df):
    k_values = [9, 18, 36]
    k_range = range(2, 50)
    kmeans_val = []
    hierarchical_val = []
    optimal_score=[]

    optimal_k_kmeans = 0
    optimal_kmeans = -1
    optimal_kmeans_calinski = -1
    optimal_kmeans_devis = -1

    optimal_k_hierarchical = 0
    optimal_hierarchical = -1
    optimal_hierarchical_calinski = -1
    optimal_hierarchical_devis = -1

    optimal_k_dbscan = 0
    optimal_dbscan = -1
    optimal_dbscan_calinski = -1
    optimal_dbscan_davis = -1

    for city in cleaned_df['City'].unique():
        city_df = cleaned_df[cleaned_df['City'] == city]
        vectorizer = TfidfVectorizer()
        vectorized_text = vectorizer.fit_transform(city_df['Cleaned_Text'])
        for k in k_values:
            if(vectorized_text.shape[0]>=k):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
                labels = kmeans.fit_predict(vectorized_text)
                silhouette = silhouette_score(vectorized_text, labels)
                calinski = calinski_harabasz_score(vectorized_text.toarray(), labels)
                davies = davies_bouldin_score(vectorized_text.toarray(), labels)
                kmeans_val.append({'k':k,'city':city,'silhouette':silhouette, 'calinski':calinski, 'davies':davies})
                
                hierarchical = AgglomerativeClustering(n_clusters=k)
                hierarchical_labels = hierarchical.fit_predict(vectorized_text.toarray())
                hierarchical_silhouette = silhouette_score(vectorized_text, hierarchical_labels)
                hierarchical_calinski = calinski_harabasz_score(vectorized_text.toarray(), hierarchical_labels)
                hierarchical_davies = davies_bouldin_score(vectorized_text.toarray(), hierarchical_labels)
                hierarchical_val.append({'k':k,'city':city,'silhouette':hierarchical_silhouette, 'calinski':hierarchical_calinski, 'davies':hierarchical_davies})
            else:
                kmeans_val.append({'k':k,'city':city,'silhouette':0, 'calinski':0, 'davies':0})
                hierarchical_val.append({'k':k,'city':city,'silhouette':0, 'calinski':0, 'davies':0})
            
        for k in k_range:
            if(vectorized_text.shape[0]>k):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
                kmeans_labels = kmeans.fit_predict(vectorized_text)
                k_means_silhouette = silhouette_score(vectorized_text, kmeans_labels)
                k_means_calinski = calinski_harabasz_score(vectorized_text.toarray(), kmeans_labels)
                k_means_davies = davies_bouldin_score(vectorized_text.toarray(), kmeans_labels)
                if k_means_silhouette > optimal_kmeans:
                    optimal_k_kmeans = k
                    optimal_kmeans = k_means_silhouette
                    optimal_kmeans_calinski = k_means_calinski
                    optimal_kmeans_davies = k_means_davies

                hierarchical = AgglomerativeClustering(n_clusters=k)
                hierarchical_labels = hierarchical.fit_predict(vectorized_text.toarray())
                hierarchical_silhouette = silhouette_score(vectorized_text, hierarchical_labels)
                hierarchical_calinski = calinski_harabasz_score(vectorized_text.toarray(), hierarchical_labels)
                hierarchical_davies = davies_bouldin_score(vectorized_text.toarray(), hierarchical_labels)
                if hierarchical_silhouette > optimal_hierarchical:
                    optimal_k_hierarchical = k
                    optimal_hierarchical = hierarchical_silhouette
                    optimal_hierarchical_calinski = hierarchical_calinski
                    optimal_hierarchical_davies = hierarchical_davies
                    
                dbscan = DBSCAN(eps=0.5, min_samples=k)
                dbscan_labels = dbscan.fit_predict(vectorized_text)
                if len(np.unique(dbscan_labels)) > 1: 
                    dbscan_silhouette = silhouette_score(vectorized_text, dbscan_labels)
                    dbscan_calinski = calinski_harabasz_score(vectorized_text.toarray(), dbscan_labels)
                    dbscan_davies = davies_bouldin_score(vectorized_text.toarray(), dbscan_labels)
                    if dbscan_silhouette > optimal_dbscan:
                        optimal_k_dbscan = k
                        optimal_dbscan = dbscan_silhouette
                        optimal_dbscan_calinski = dbscan_calinski
                        optimal_dbscan_davis = dbscan_davies
                        
        
                    
        optimal_score.append({'City':city, 'optimal_k_kmeans':optimal_k_kmeans, 'optimal_k_hierarchical':optimal_k_hierarchical, 'optimal_k_dbscan':optimal_k_dbscan,
                            'k_means_silhouette':optimal_kmeans,'k_means_calinski':optimal_kmeans_calinski, 'k_means_davies': optimal_kmeans_davies, 
                            'hierarchical_silhouette':optimal_hierarchical, 'hierarchical_calinski':optimal_hierarchical_calinski, 'hierarchical_davies':optimal_hierarchical_davies,
                            'dbscan_silhouette': optimal_dbscan, 'dbscan_calinski': optimal_dbscan_calinski, 'dbscan_davies':optimal_dbscan_davis})

    optimal_values = [{'City':d['City'],'optimal-k_means':d['optimal_k_kmeans'], 'optimal_k_hierarchical': d['optimal_k_hierarchical'],'optimal_k_dbscan': d['optimal_k_dbscan'], 
                    'k_means_optimal_score':[d['k_means_silhouette'], d['k_means_calinski'], d['k_means_davies']], 
                    'Hierarchical_optimal_score':[d['hierarchical_silhouette'],d['hierarchical_calinski'], d['hierarchical_davies']],
                    'DBSCAN_optimal_score':[d['dbscan_silhouette'], d['dbscan_calinski'], d['dbscan_davies']]}for d in optimal_score]
    return optimal_values

def calculateClusterId(optimal_values, cleaned_df):
    opt_i = []
    for i in optimal_values:
        if i['k_means_optimal_score'][0]>=i['Hierarchical_optimal_score'][0] and i['k_means_optimal_score'][0]>=i['DBSCAN_optimal_score'][0]:
            opt = i['optimal-k_means']
            opt_i.append({'City':i['City'], 'clusterid': opt})
        elif i['Hierarchical_optimal_score'][0] >= i['k_means_optimal_score'][0] and i['Hierarchical_optimal_score'][0] >= i['DBSCAN_optimal_score'][0]:
            opt = i['optimal_k_hierarchical']
            opt_i.append({'City':i['City'], 'clusterid': opt})
        else:
            opt = i['optimal_k_dbscan']
            opt_i.append({'City':i['City'], 'clusterid': opt})        
    clusterid_df = pd.DataFrame(opt_i)
    cleaned_df = pd.merge(cleaned_df, clusterid_df, on="City")
    print(cleaned_df)
    return cleaned_df

def create_TsvFile(result_df):
    result_df.to_csv('smartcity_predict.tsv', sep='\t')

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--document", type=str, required=True, nargs='*')
    args=parser.parse_args()
    pd.set_option('display.max_columns', None)
    files = input_files (args)
    df = create_dataframe(files)
    cleaned_df = cleanPDF(df)
    optimal_values = performClustering(cleaned_df)
    final_df = calculateClusterId(optimal_values, cleaned_df)
    create_TsvFile(final_df)