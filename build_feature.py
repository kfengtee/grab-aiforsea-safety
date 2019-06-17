import pandas as pd
import numpy as np

def dispersion(s):
    # calculate range
    return s.max() - s.min()

def build_feature(tele, tele_iqr, win_mean_std, pca_gyro, cluster_win):
    """
    Build 3 type of features from raw telematics data. 
        Type 1: Statistical summary of telematics data, including mean, 
                median and standard deviation. 
        Type 2: Count outlying driving behaviours based on telematics 
                readings. For example, the number of times a driver exceed 
                speed limit on highway (110 kmph). 
        Type 3: Sliding windows aggregated features. We slide over the 
                telematics using window size of 8 and compute the 
                corresponding statistical summary for these windows. Then, 
                we clusters these windows using K-means clustering algorithm. 
                Each of these clusters represent certain driving behaviour, such as 
                harsh braking and hard acceleration. The number of occurence 
                for each of these clusters (behaviour) is then used as
                feature for the trip.
    
    Prior to building features, we perform data cleaning and data transformation.
        Data Cleaning: 
            1. Remove observations with inaccurate GPS data, as suggested by 
               'Accuracy' feature.
            2. Remove observations with Speed = -1.
        Data Transformation:
            1. Transform triaxial accelerometer readings into one by finding
               the magnitue. 
                   (Magnitude = sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
            2. Transform the gyroscopre readings into its first principal 
               component using Principal Component Analysis (PCA).
    
    Parameters
    ----------
    tele: DataFrame, columns = ['bookingID', 'Accuracy', 'Bearing', 
                                'acceleration_x','acceleration_y', 
                                'acceleration_z', 'gyro_x', 'gyro_y',
                                'gyro_z', 'second', 'Speed']
        Telematics data in raw format (same as given by GRAB).
        
    tele_iqr: dictionary, keys = ['acceleration_z', 'acceleration_x', 
                                  'acceleration_y', 'gyro_y', 'Speed', 
                                  'second', 'gyro_z', 'gyro_x']
        The 25th and 75th percentile of telematics data. 
        Calculated from training data.
    
    win_mean_std: dictionary, keys = ['acceleration_std', 'Speed_median', 
                                      'acceleration_mean', 'gyro_median', 
                                      'acceleration_median', 'Speed_std', 
                                      'gyro_std', 'Speed_mean', 'gyro_mean']
        The mean and standard deviation of sliding window aggregated features. 
        Calculated from training data.
    
    pca_gyro: sklearn.decomposition.PCA model
        Pre-trained PCA model to transform triaxial gyroscope readings into 
        first principal component. Trained using training data.
        
    """
    # required column names
    COL_ACCE = ('acceleration_x', 'acceleration_y', 'acceleration_z')
    COL_GYRO = ('gyro_x', 'gyro_y', 'gyro_z')
    COL_TELE = ('bookingID', 'Accuracy', 'Bearing', 'second', 'Speed', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z')
    
    bid = tele.bookingID.unique()
    
    #### STAGE 0: Data Validation ####
    if not sorted(tele.columns) == sorted(COL_TELE):
        raise Exception('Input columns mismatched! Expected: \n {}'.format(COL_TELE))
     
    # sort according to bookingID & seconds
    tele = tele.sort_values(['bookingID', 'second']).reset_index(drop=True)
    
    
    #### STAGE 1: Data Cleaning ####
    
    print("... (1/5) cleaning data ... ")
    # filter out inaccurate GPS data and speed = -1
    tele = tele.loc[(tele.Accuracy <= 16) & (tele.Speed != -1)]
    
    # drop 'Accuracy' & 'Bearing' to save memory. we don't need these anymore. 
    tele.drop(['Accuracy', 'Bearing'], axis=1, inplace=True)
    
    
    #### STAGE 2: Data Transformation ####
    
    print("... (2/5) transforming data ... ")
    # calculate magnitude of acceleration sqrt(acc_x^2 + acc_y^2 + acc_z^2)
    tele['acceleration'] = np.sqrt((tele.loc[:, COL_ACCE] ** 2).sum(axis=1))
    
    # transform triaxial gyro readings into its first principal components
    tele['gyro'] = pca_gyro.transform(tele.loc[:, COL_GYRO])
    
    
    #### STAGE 3A: Generating Feature (Type 1: Statistical Description) ####
    
    print("... (3/5) generating feature (Type 1: Statistical Description) ... ")
    feature1 = tele.groupby('bookingID')['acceleration', 'gyro', 'Speed', 'second'].agg(['mean', 'median', 'std', dispersion]).fillna(0)
    feature1.columns = ['_'.join(col) for col in feature1.columns] # rename columns
    feature1.reset_index(inplace=True)
    
    
    #### STAGE 3B: Generating Feature (Type 2: Detecting Outlying Behaviours) ####
    
    print("... (4/5) generating feature (Type 2: Counting Outlying Behaviours) ... ")
    feature2 = pd.DataFrame()
    
    # use 75th percentile only
    feature2['over_Speed'] = tele.groupby('bookingID')['Speed'].apply(lambda x: sum(x > tele_iqr['Speed'][1]))
    feature2['over_second'] = tele.groupby('bookingID')['second'].apply(lambda x: sum(x > tele_iqr['second'][1]))
    
    # use 25th and 75th percentile
    for col in (COL_ACCE + COL_GYRO):
        feature2['over_{}'.format(col)] = tele.groupby('bookingID')[col].apply(lambda x: sum((x < tele_iqr[col][0]) | (x > tele_iqr[col][1])))
    
    feature2.reset_index(inplace=True)
    
    
    #### STAGE 3C: Generating Feature (Type 3: Sliding Window)
    
    print("... (5/5) generating feature (Type 3: Sliding Window) ... ")
    print("    (WARNING! This process may take up many RAM memory. Please allocate enough memory.)")
    print('    Side Note: This may take awhile, please be patient. :)')
    
    # groupby object
    agg_win_feat = tele.loc[:, ['bookingID', 'Speed', 'acceleration', 'gyro']].groupby('bookingID')
    
    # calculate aggregate features for rolling windows of size 8, overlapped 50% 
    agg_win_feat = agg_win_feat.rolling(8).agg(['mean', 'median', 'std']).dropna()[::4]
    
    # minor adjustments towards output rows and columns
    agg_win_feat = agg_win_feat.drop('bookingID', axis=1)

    agg_win_feat.columns = ['_'.join(col) for col in agg_win_feat.columns]
    
    # standardize the data before clustering algorithm
    agg_win_feat = agg_win_feat.apply(lambda x: (x - win_mean_std[x.name][0]) / (win_mean_std[x.name][1]))
    
    # cluster into different groups (different driving behaviour, e.g: harsh braking, hard acceleration)
    agg_win_feat['cluster'] = cluster_win.predict(agg_win_feat)
    agg_win_feat = agg_win_feat.droplevel(1).reset_index()
    
    # count the occurrence of each actions during a trip
    feature3 = pd.crosstab(agg_win_feat.bookingID, agg_win_feat.cluster)
    feature3.reset_index(inplace=True)
    
    # handle missing clusters
    exp_clust = set(range(cluster_win.n_clusters)) # expected clusters
    out_clust = set(feature3.columns) # outputed clusters
    for col in exp_clust - out_clust:
        feature3[col] = 0

    feature3.columns = feature3.columns.astype(str)
    
    # join all 3 features
    output = pd.DataFrame(bid, columns=['bookingID'])
    output = output.merge(feature1, how='left', on='bookingID')
    output = output.merge(feature2, how='left', on='bookingID')
    output = output.merge(feature3, how='left', on='bookingID')
    output = output.fillna(0)
    
    print('Done!')
    
    return output