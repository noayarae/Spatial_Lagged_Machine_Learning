

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pyproj import Proj, transform
from scipy.spatial import distance_matrix
from libpysal.weights import DistanceBand
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


df = pd.read_csv('D:/work/research_t/corn_field/ml_modeling/rgb_7_s.csv')

# Re-projecting M1
'''proj_wgs84 = Proj(init='epsg:4326')  # WGS84 Latitude/Longitude
proj_utm = Proj(init='epsg:32614')   # WGS84 UTM Zone 14N
df['East'], df['North'] = transform(proj_wgs84, proj_utm, df['LON'].values, df['LAT'].values)
print(df) #'''

# Re-projecting M2. Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LON'], df['LAT']), crs="EPSG:4326")
gdf_utm = gdf.to_crs("EPSG:32614")  # Project to UTM Zone 14N
df['East2'] = gdf_utm.geometry.x
df['North2'] = gdf_utm.geometry.y


#-------------------- TEST
import libpysal
from esda.moran import Moran

# Step 3.1: Define the spatial weights matrix (e.g., using K-nearest neighbors)
coords1 = df[['East2', 'North2']].values
knn_weights1 = libpysal.weights.KNN.from_array(coords1, k=8)  # using 8 nearest neighbors

# Step 3.2: Calculate Moran's I for the Yield column
moran_yield = Moran(df['Yield_Mg_h'], knn_weights1)
print(f"Moran's I for Yield: {moran_yield.I}")
print(f"p-value: {moran_yield.p_sim}")


# Step 3.1: Calculate Moran's I using a distance-based approach
distance_threshold = 20  # Set an appropriate distance threshold in the same units as East2 and North2
distance_weights1 = DistanceBand.from_array(coords1, threshold=distance_threshold, binary=True)
moran_yield = Moran(df['Yield_Mg_h'], distance_weights1)
print(f"Moran's I for Yield: {moran_yield.I}")
print(f"p-value: {moran_yield.p_sim}")


#------------------ END TEST


# Step 1: Define Spatial Weights Matrix W based on Euclidean distance threshold
distance_threshold = 7.0  # Define a reasonable threshold based on data scale            # <--- SET

# Compute pairwise distances based on coordinates
distances = distance_matrix(df[['East2', 'North2']], df[['East2', 'North2']])

# Create binary weight matrix W based on the distance threshold
W = (distances <= distance_threshold).astype(int)  # 1 if within threshold, 0 otherwise
np.fill_diagonal(W, 0)  # Remove self-loops (set diagonal to 0)
print(colored(250, 150, 0, ('Avge of neighbors: ', np.sum(W)/W.shape[0])))


# Step 2: Create spatially lagged predictors using the weights matrix W
#for col in ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']:
for col in ['Blue', 'Green', 'Red', 'RedEdge', 'NIR','NDVI','EVI','SAVI','GNDVI','NDRE','MSAVI','TVI','GCI','ARI','CCCI']:
    # Calculate spatial lag as weighted average of neighboring values
    df[f'{col}_lag'] = W.dot(df[col]) / W.sum(axis=1).clip(min=1)  # Normalize by sum of weights, avoid division by zero




# Step 1: Cluster the data into k clusters based on spatial coordinates
n_clusters = 10  # You can adjust this based on your data                                  # <--- SET
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['LON', 'LAT']])

# Save DataFrame to CSV
#df.to_csv('D:/work/research_t/corn_field/experiment_neighbor/df_clust_7.csv', index=False)



# Step 2: Split clusters into training and testing sets
train_clusters, test_clusters = train_test_split(df['Cluster'].unique(), test_size=0.15, random_state=42)
#train_clusters, test_clusters = [0, 2, 3, 4, 5], [1] # [0,1,2,3,5,6], [4] # 
train_clusters, test_clusters = [1,2,3,4,5,6,7,8,9], [0] # 10 clusters
train_data = df[df['Cluster'].isin(train_clusters)]
test_data = df[df['Cluster'].isin(test_clusters)]


# Step 3: Define features and target
features = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR'] #
#features = ['Blue','Green','Red','RedEdge','NIR', 'NDVI', 'EVI', 'SAVI', 'GNDVI', 'NDRE', 'MSAVI', 'TVI', 'GCI', 'ARI', 'CCCI']
features = ['Blue','Green','Red','RedEdge','NIR','NDVI','EVI','SAVI','GNDVI','NDRE','MSAVI','TVI','GCI','ARI','CCCI', 
            #'BGI','NGRDI','MCARI','CREI','PPR','GNDRE','TSAVI','PSRI','MCCI','GARI']
            'BGI','NGRDI','MCARI','CREI','PPR','GNDRE','TSAVI','PSRI','MCCI','NPCI'] #'''
#features = ['Blue','Green','Red','RedEdge','NIR','NDVI','EVI','SAVI','GNDVI','NDRE','MSAVI','TVI','GCI','ARI','CCCI', 
            #'BGI','NGRDI','MCARI','CREI','PPR','GNDRE','TSAVI','PSRI','MCCI','GARI',
            ##'BGI','NGRDI','MCARI','CREI','PPR','GNDRE','TSAVI','PSRI','MCCI','NPCI',
            #'Blue_lag','Green_lag','Red_lag','RedEdge_lag','NIR_lag'] #'''

#features = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR', 'Blue_lag', 'Green_lag', 'Red_lag', 'RedEdge_lag', 'NIR_lag']
#features = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR'#,
#            'NDVI','EVI','SAVI','GNDVI','NDRE',#'MSAVI','TVI','GCI','ARI','CCCI',
#            'Blue_lag', 'Green_lag', 'Red_lag', 'RedEdge_lag', 'NIR_lag',
#            'NDVI_lag', 'EVI_lag', 'SAVI_lag', 'GNDVI_lag', 'NDRE_lag'#, 
#            'MSAVI_lag', 'TVI_lag', 'GCI_lag', 'ARI_lag', 'CCCI_lag'
#            ]

features = ['Blue','Green','Red','RedEdge','NIR'
            ,'NDVI','EVI','SAVI','GNDVI','NDRE' # (1)
            ,'MSAVI','TVI','GCI','ARI','CCCI'   # (2)
            ,'BGI','NGRDI','MCARI','CREI','PPR' # (3)
            ,'GNDRE','TSAVI','PSRI','MCCI','NPCI'# (4)
            ] #'''

### Backward
'''features = ['Blue','Green','Red','RedEdge','NIR'
            ,'CREI','GCI','CCCI','GNDRE','NPCI'
            ,'ARI','MCARI','MCCI','MSAVI','PPR'
            ,'BGI','EVI','NDRE','NGRDI','TVI'
            ,'GNDVI','NDVI','PSRI','SAVI','TSAVI'
            ] #'''

# New set of VIs
features = ['Blue','Green','Red','RedEdge','NIR','Blue_lag','Green_lag','Red_lag','RedEdge_lag','NIR_lag'
            ,'GCI','CCCI','CREI','NPCI','TrVI'
            ,'ARI','MCARI','MCCI','MSAVI','PPR'
            ,'BGI','EVI','NDRE','NGRDI','TVI'
            ,'GNDVI','NDVI','PSRI','SAVI','TSAVI'
            #,'RDVI','MTVI2','NG','NDVIBlue','TCI'
            #,'GNDRE'
            ] #'''
# New VIs for Forward
'''features = ['Blue','Green','Red','RedEdge','NIR'
            ,'GNDVI','NDVI','PSRI','SAVI','TSAVI'
            #,'RDVI','MTVI2','NG','NDVIBlue','TCI'
            #,'ARI','MCARI','MCCI','MSAVI','PPR'
            #,'BGI','EVI','NDRE','NGRDI','TVI'
            #,'GCI','CCCI','CREI','NPCI','TrVI'
            ] #'''
target = 'Yield_Mg_h'


# Model hyperparameters
rf_best_params = {'max_depth': 21, 'max_features': 11, 'n_estimators': 500}
xg_best_params = {'colsample_bytree': 0.9, 'gamma': 0.30, 'learning_rate': 0.05, 'max_depth': 6, 
                  'n_estimators': 200,'n_jobs': 8, 'subsample': 0.7, 'verbosity': 1}
et_best_params = {'max_depth': 21, 'max_features': 51, 'n_estimators': 300}
dt_best_params = {'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 8}
gbr_best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.7}


# Step 4: Train the model using the training data
#ml_model = LinearRegression()
#ml_model = RandomForestRegressor(**rf_best_params, random_state=42)
#ml_model = xgb.XGBRegressor(**xg_best_params, random_state=42)
#ml_model = ExtraTreesRegressor(**et_best_params, random_state=42)
#ml_model = DecisionTreeRegressor(**dt_best_params, random_state=42)
#ml_model = GradientBoostingRegressor(**gbr_best_params, random_state=42)

choice = 'GBR' # 'LR', 'RF', 'XGB', 'ET', 'DT', 'GBR'
model_text = {'LR': 'Linear Regression', 'RF': 'Random Forest', 'XGB': 'XGBoost', 'ET': 'Extra Tree Regression', 
              'DT': 'Decision Tree Regression', 'GBR':'Gradient Boosting Regression'}

models = {'LR': LinearRegression(), 'RF': RandomForestRegressor(**rf_best_params, random_state=42),
          'XGB':xgb.XGBRegressor(**xg_best_params, random_state=42), 'ET':ExtraTreesRegressor(**et_best_params, random_state=42),
          'DT': DecisionTreeRegressor(**dt_best_params, random_state=42), 'GBR':GradientBoostingRegressor(**gbr_best_params, random_state=42)}
ml_model = models[choice]


ml_model.fit(train_data[features], train_data['Yield_Mg_h'])


# Step 5: Test the model on the spatially separate testing data
y_test_pred = ml_model.predict(test_data[features])
r2 = r2_score(test_data['Yield_Mg_h'], y_test_pred)
rmse = np.sqrt(mean_squared_error(test_data['Yield_Mg_h'], y_test_pred))
mae = mean_absolute_error(test_data['Yield_Mg_h'], y_test_pred)   # Mean Absolute Error
r_value, _ = pearsonr(test_data['Yield_Mg_h'], y_test_pred)

print(colored(250, 250, 0, (f"R2 on Test Set: {r2}")))
print(colored(250, 150, 0, (f"RMSE on Test Set: {rmse}")))
print(colored(250, 150, 0, (f"MAE on Test Set: {mae}")))
print(colored(100, 250, 0, (f"Pearson corr-coeff (r): {r_value}")))
print(r2, rmse)#, mae, r_value)



# Step 5.2: Calculate residuals
residuals = test_data['Yield_Mg_h'] - y_test_pred

#----- TEST
# Step 6: Calculate Moran's I for residuals
coords2 = test_data[['East2', 'North2']].values
knn_weights2 = libpysal.weights.KNN.from_array(coords2, k=8)

moran_res_1 = Moran(residuals, knn_weights2)
print(colored(0, 250, 255, (f"Moran's I for residuals: {moran_res_1.I}")))
print(colored(0, 250, 250, (f"p-value: {moran_res_1.p_sim}")))

# Step 3.1: Calculate Moran's I using a distance-based approach
dist_thresh_res = 7.0  # Set distance threshold
dist_weights2 = DistanceBand.from_array(coords2, threshold = dist_thresh_res, binary=True)
moran_res_2 = Moran(residuals, dist_weights2)
print(colored(0, 250, 255, (f"Moran's I for Yield: {moran_res_2.I}")))
print(colored(0, 250, 250, (f"p-value: {moran_res_2.p_sim}")))
#------------ END TEST




# Scatter Plot Random Forest
y_test, RF_pred = test_data['Yield_Mg_h'], y_test_pred
r2_RF, rmse_RF, mae_RF = r2, rmse, mae
plt.figure(figsize=(16, 4), dpi=500)
plt.subplot(1, 3, 1)
#plt.scatter(y_test, RF_pred, c=color_values, cmap='viridis', alpha=0.5)
plt.scatter(y_test, RF_pred, color ='#b3b3ff', edgecolor='#0000ff', alpha=0.5, s=10, linewidths=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=0.8)
plt.xlabel('Measured Yield')
plt.ylabel('Predicted Yield')
plt.title(model_text[choice]) # ('Linear Regression')
plt.text(0.05, 0.95, f'R2: {r2_RF:.2f}\nRMSE: {rmse_RF:.2f}\nMAE: {mae_RF:.2f}', # \nMean Y: {mean_y_test:.2f}
         transform=plt.gca().transAxes, fontsize=8, verticalalignment='top', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#plt.text(0.5, 0.95, 'Linera Regression', transform=plt.gca().transAxes, fontsize=8, 
#         horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='white'))
plt.grid(color='#cccccc', linestyle='--', linewidth = 0.4) 
plt.show()
#'''






## Important Features Random Forest
f_import_rf = 100 * ml_model.feature_importances_
sorted_idx = np.argsort(f_import_rf)

# Plotting the feature importances
fig = plt.figure(figsize=(8, 10), dpi=300)
plt.barh(range(len(sorted_idx)), f_import_rf[sorted_idx], align='center', color='#b3b3ff', edgecolor='#0000ff',
         linewidth=0.5, height=0.5)

# Set y-tick labels to the feature names sorted by importance
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx], fontsize=10)

plt.title('Feature Importance Random Forest')
plt.grid(axis="x")
plt.xlabel("Percentage %")
plt.ylabel("Parameter")
plt.show()

# Print feature importances sorted in descending order
import_df_rf = pd.DataFrame({
    'Feature': np.array(features)[sorted_idx][::-1],
    'Importance': f_import_rf[sorted_idx][::-1]
})
print("Importance in percentages:\n", import_df_rf)
#'''

























