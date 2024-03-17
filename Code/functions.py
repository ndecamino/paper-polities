def pairs(df, type, by_period):
    
    # Modules: 
    import pandas as pd

    # Culture pairs dataframe for all periods:
    if by_period==False:
        
        # Obtain culture pairs in all periods:
        cultures = df.culture_cl.unique().tolist()
        culture_pairs = [(a, b) for idx, a in enumerate(cultures) for b in cultures[idx:]]
        df_pairs = pd.DataFrame(culture_pairs, columns=['culture_1', 'culture_2'])

        # Add descriptions for each culture in each pair:
        df_pairs[[f'{type}_1', f'{type}_2']] = None
        for i in range(len(df_pairs)):
            for j in range(1,3):
                culture = df_pairs.iloc[i][f'culture_{j}']
                culture_df = df[df['culture_cl']==culture]
                df_pairs.at[i, f'{type}_{j}'] = culture_df[f'{type}'].tolist()

    # Culture pairs dataframe by periods:
    elif by_period == True:

        # Obtain culture pairs in each period:
        periods = df.cronology_time.unique().tolist()
        culture_pairs = []
        for period in periods:
            period_df = df[df['cronology_time']==period]
            period_cultures = period_df.culture_cl.unique().tolist()
            period_culture_pairs = [(a, b) for idx, a in enumerate(period_cultures) for b in period_cultures[idx:]]
            for pair in period_culture_pairs:
                culture_pairs.append((*pair, period))
        df_pairs = pd.DataFrame(culture_pairs, columns=['culture1', 'culture2', 'period'])

        # Add descriptions (as lists of sets of words or shingles) for each culture in each pair:
        df_pairs[[f'{type}_1', f'{type}_2']] = None
        for i in range(len(df_pairs)):
            period = df_pairs.iloc[i]['period']
            for j in range(1,3):
                culture = df_pairs.iloc[i][f'culture_{j}']
                period_culture_df = df[df['culture_cl']==culture & df['cronology_time']==period]
                df_pairs.at[i, f'{type}_{j}'] = period_culture_df[f'{type}'].tolist()
                df_pairs.at[i, f'{type}_{j}'] = period_culture_df[f'{type}'].tolist()
    
    # Return culture pairs dataframe:
    return df_pairs

def jaccard_similarity(set_1, set_2):
    intersection_size = len(set_1.intersection(set_2))
    union_size = len(set_1.union(set_2))
    similarity = intersection_size / union_size if union_size != 0 else 0
    return similarity

def mean_distance(row, type):

    # Modules:
    import numpy as np

    # Obtain cultures' names:
    culture_1, culture_2 = row['culture_1'], row['culture_2']
    
    # Return distance == 0 if the culture is the same:
    if culture_1 == culture_2:
        return 0.0

    # Get descriptions (as lists of sets of words or shingles) for each culture:
    descriptions_1, descriptions_2 = row[f'lemmatized_description'], row[f'{type}_2']

    # Calculate the mean distance between any pair of descriptions:
    distances = []
    if type == 'words' or type == 'shingles':
        for description_1 in descriptions_1:
            for description_2 in descriptions_2:
                similarity = jaccard_similarity(description_1, description_2)
                distance = 1 - similarity
                distances.append(distance)
    elif type == 'tf-idf':
        distance = None
        distances.append(distance)
    elif type == 'bert_embeddings':
        distance = None
        distances.append(distance)

    # Return the mean distance:
    mean_distance = np.mean(distances)
    return mean_distance

def cultures_distances(df, type, length_filter, by_period):    
    
    # Modules:
    import pandarallel

    # Filter by length of the description:
    df = df.loc[df['lem_desc_length'] > length_filter]

    # Obtain culture pairs dataframe:
    df_pairs = pairs(df, by_period=by_period)  
    
    # Calculate the mean distance of each culture pair:
    pandarallel.initialize(verbose=0, progress_bar=False)
    df_pairs['mean_distance'] = df_pairs.parallel_apply(
                lambda row: mean_distance(row, type=type), axis=1
        )

    # Return culture pairs dataframe:
    return df_pairs

def distance_matrix(mean_distances_df):
    
    # If the 'period' column doesn't exist print one mean distance matrix:
    if 'period' not in mean_distances_df.columns:
        mean_distances_matrix = mean_distances_df.pivot(index='culture_1', columns='culture_2', values='mean_distance')
        mean_distances_matrix = mean_distances_matrix.fillna(mean_distances_matrix.T)
        headers = [''] + mean_distances_matrix.columns.tolist()
        print(tabulate(mean_distances_matrix, headers=headers, floatfmt=".4f"))

    # If the 'period' column exists print one mean distance matrix for each period:
    else:
        periods = mean_distances_df.period.unique().tolist()
        for period in periods:
            mean_distances_matrix = mean_distances_df[mean_distances_df.period == period].pivot(index='culture_1', columns='culture_2', values='mean_distance')
            mean_distances_matrix = mean_distances_matrix.fillna(mean_distances_matrix.T)
            headers = [''] + mean_distances_matrix.columns.tolist()
            print(f"Periodo: {period}")
            print(tabulate(mean_distances_matrix, headers=headers, floatfmt=".4f"))
            print()