import argparse
import os
import sys
import pypdf
import glob
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
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
        files = glob.glob(file_path)

    if not files:
        print("Text file not present", file=sys.stderr)
        sys.exit(0)

    return files

def create_dataframe(files):
    doc_text = []
    city_names = []
    for filename in files:
        if filename.endswith('.pdf'):
            base_name = os.path.splitext(os.path.basename(filename))[0]
            city_name = base_name.split('_')[0]
            try:
                with open(filename, 'rb') as pdf_file:
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text = page.extract_text() or ""
                        doc_text.append(text)
                        city_names.append(city_name)
            except Exception as exc:
                print(f"Failed to read {filename}: {exc}", file=sys.stderr)
    df = pd.DataFrame({'City': city_names, 'raw_text': doc_text})
    return df

def cleanPDF(df):
    cleaned_text: list[str] = []
    pattern = re.compile(r'[\d\W]+')
    for text in df['raw_text'].fillna(''):
        text = pattern.sub(' ', text).lower().strip()
        cleaned_text.append(text)
    df['Cleaned_Text'] = cleaned_text
    cities_to_remove = ['OH Toledo', 'CA Moreno Valley', 'TX Lubbock', 'NV Reno', 'FL Tallahassee',
                        'NY Mt Vernon Yonkers New Rochelle', 'VA Newport News']
    df = df[~df['City'].isin(cities_to_remove)]
    df = df[df['Cleaned_Text'].astype(bool)]
    df.reset_index(drop=True, inplace=True)
    return df


def performClustering(cleaned_df):
    k_values = [9, 18, 36]
    kmeans_val = []
    hierarchical_val = []
    optimal_score = []

    for city in cleaned_df['City'].unique():
        city_df = cleaned_df[cleaned_df['City'] == city]

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
        vectorized_text = vectorizer.fit_transform(city_df['Cleaned_Text'])

        n_samples, n_features = vectorized_text.shape
        if n_samples < 2 or n_features == 0:
            # Not enough data to cluster
            optimal_score.append({
                'City': city,
                'optimal_k_kmeans': 0,
                'optimal_k_hierarchical': 0,
                'optimal_k_dbscan': 0,
                'k_means_silhouette': -1,
                'k_means_calinski': -1,
                'k_means_davies': -1,
                'hierarchical_silhouette': -1,
                'hierarchical_calinski': -1,
                'hierarchical_davies': -1,
                'dbscan_silhouette': -1,
                'dbscan_calinski': -1,
                'dbscan_davies': -1,
            })
            continue

        # Dimensionality reduction to avoid dense conversions and speed up clustering/metrics
        n_components = max(2, min(100, n_samples - 1, n_features - 1)) if n_features > 1 else 2
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        normalizer = Normalizer(copy=False)
        X_reduced = normalizer.fit_transform(svd.fit_transform(vectorized_text))

        # Evaluate fixed k values when feasible
        for k in k_values:
            if n_samples >= k and k >= 2:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=5)
                labels = kmeans.fit_predict(X_reduced)
                silhouette = silhouette_score(X_reduced, labels, metric='euclidean')
                calinski = calinski_harabasz_score(X_reduced, labels)
                davies = davies_bouldin_score(X_reduced, labels)
                kmeans_val.append({'k': k, 'city': city, 'silhouette': silhouette, 'calinski': calinski, 'davies': davies})

                hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
                hierarchical_labels = hierarchical.fit_predict(X_reduced)
                hierarchical_silhouette = silhouette_score(X_reduced, hierarchical_labels, metric='euclidean')
                hierarchical_calinski = calinski_harabasz_score(X_reduced, hierarchical_labels)
                hierarchical_davies = davies_bouldin_score(X_reduced, hierarchical_labels)
                hierarchical_val.append({'k': k, 'city': city, 'silhouette': hierarchical_silhouette, 'calinski': hierarchical_calinski, 'davies': hierarchical_davies})
            else:
                kmeans_val.append({'k': k, 'city': city, 'silhouette': 0, 'calinski': 0, 'davies': 0})
                hierarchical_val.append({'k': k, 'city': city, 'silhouette': 0, 'calinski': 0, 'davies': 0})

        # Search for optimal k in a bounded range
        max_k = max(2, min(20, n_samples - 1))
        optimal_k_kmeans = 0
        optimal_kmeans = -1
        optimal_kmeans_calinski = -1
        optimal_kmeans_davies = -1

        optimal_k_hierarchical = 0
        optimal_hierarchical = -1
        optimal_hierarchical_calinski = -1
        optimal_hierarchical_davies = -1

        for k in range(2, max_k + 1):
            if n_samples > k:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=5)
                kmeans_labels = kmeans.fit_predict(X_reduced)
                k_means_silhouette = silhouette_score(X_reduced, kmeans_labels, metric='euclidean')
                k_means_calinski = calinski_harabasz_score(X_reduced, kmeans_labels)
                k_means_davies = davies_bouldin_score(X_reduced, kmeans_labels)
                if k_means_silhouette > optimal_kmeans:
                    optimal_k_kmeans = k
                    optimal_kmeans = k_means_silhouette
                    optimal_kmeans_calinski = k_means_calinski
                    optimal_kmeans_davies = k_means_davies

                hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
                hierarchical_labels = hierarchical.fit_predict(X_reduced)
                hierarchical_silhouette = silhouette_score(X_reduced, hierarchical_labels, metric='euclidean')
                hierarchical_calinski = calinski_harabasz_score(X_reduced, hierarchical_labels)
                hierarchical_davies = davies_bouldin_score(X_reduced, hierarchical_labels)
                if hierarchical_silhouette > optimal_hierarchical:
                    optimal_k_hierarchical = k
                    optimal_hierarchical = hierarchical_silhouette
                    optimal_hierarchical_calinski = hierarchical_calinski
                    optimal_hierarchical_davies = hierarchical_davies

        # DBSCAN grid search with small parameter grid
        optimal_k_dbscan = 0  # we will store the best min_samples as a proxy
        optimal_dbscan = -1
        optimal_dbscan_calinski = -1
        optimal_dbscan_davis = -1

        eps_values = [0.3, 0.5, 0.7]
        min_samples_values = sorted({3, 5, max(2, int(np.sqrt(n_samples)))})
        for min_samples in min_samples_values:
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X_reduced)
                if len(np.unique(dbscan_labels)) > 1:
                    dbscan_silhouette = silhouette_score(X_reduced, dbscan_labels, metric='euclidean')
                    dbscan_calinski = calinski_harabasz_score(X_reduced, dbscan_labels)
                    dbscan_davies = davies_bouldin_score(X_reduced, dbscan_labels)
                    if dbscan_silhouette > optimal_dbscan:
                        optimal_k_dbscan = min_samples
                        optimal_dbscan = dbscan_silhouette
                        optimal_dbscan_calinski = dbscan_calinski
                        optimal_dbscan_davis = dbscan_davies

        optimal_score.append({
            'City': city,
            'optimal_k_kmeans': optimal_k_kmeans,
            'optimal_k_hierarchical': optimal_k_hierarchical,
            'optimal_k_dbscan': optimal_k_dbscan,
            'k_means_silhouette': optimal_kmeans,
            'k_means_calinski': optimal_kmeans_calinski,
            'k_means_davies': optimal_kmeans_davies,
            'hierarchical_silhouette': optimal_hierarchical,
            'hierarchical_calinski': optimal_hierarchical_calinski,
            'hierarchical_davies': optimal_hierarchical_davies,
            'dbscan_silhouette': optimal_dbscan,
            'dbscan_calinski': optimal_dbscan_calinski,
            'dbscan_davies': optimal_dbscan_davis,
        })

    optimal_values = [{
        'City': d['City'],
        'optimal-k_means': d['optimal_k_kmeans'],
        'optimal_k_hierarchical': d['optimal_k_hierarchical'],
        'optimal_k_dbscan': d['optimal_k_dbscan'],
        'k_means_optimal_score': [d['k_means_silhouette'], d['k_means_calinski'], d['k_means_davies']],
        'Hierarchical_optimal_score': [d['hierarchical_silhouette'], d['hierarchical_calinski'], d['hierarchical_davies']],
        'DBSCAN_optimal_score': [d['dbscan_silhouette'], d['dbscan_calinski'], d['dbscan_davies']]
    } for d in optimal_score]
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