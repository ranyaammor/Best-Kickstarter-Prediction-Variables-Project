#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Data handling
import pandas as pd
import numpy as np

# Model building
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


###1st Model:CLASSIFICATION


# In[ ]:


# Importing the data
df = pd.read_excel('C:/Users/ranya/OneDrive/Bureau/DATA ANALYTICS/DATA MINING/Kickstarter.xlsx')
df.head()


# In[ ]:


# Pre-Processing
df = df.dropna()
df = df.drop_duplicates()
display(df.isnull().sum())
df.info()


# In[ ]:


# Filtering out projects to include only "successful" or "failed" based on 'state'
df = df[df['state'].isin(['successful', 'failed'])]

# Dropping columns not useful for the model
columns_to_drop = [
    'id',  # Unique identifier, does not contribute to prediction
    'name',  # Textual data, complex to process without more complex techniques
    'blurb',  # Textual data
    'pledged','usd_pledged',#post launch data
    'backers_count',# #post launch data
    'spotlight',#post launch data
    'staff_pick',#post launch data
    'state_changed_at','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','state_changed_at_weekday',#post launch data
    'disable_communication',
    'currency',
    'launch_to_state_change_days',#post launch data
    'deadline','deadline_weekday','deadline_day',
    'created_at','created_at_day','created_at_hr','created_at_weekday',
    'launched_at','launched_at_weekday','launched_at_hr','launched_at_day',
    'name_len',
    'blurb_len',
    'deadline_hr',   
]

df = df.drop(columns=columns_to_drop, errors='ignore')

# Convert to USD and drop now irrelevant predictors
df['usd_goal'] = df['goal'] * df['static_usd_rate']
irrelevant_variables = ['goal', 'static_usd_rate']
df = df.drop(columns=irrelevant_variables)

# Verifying the columns that remain in df_cleaned
remaining_columns = df.columns.tolist()
remaining_columns


# In[ ]:


# Converting all categorical predictors to category for consistency / except month for heatmap
df.dtypes
df['country'] = df['country'].astype('category')
df['category'] = df['category'].astype('category')
df.dtypes


# In[ ]:


# Heatmap to visualize correlations
c= df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12,12))         # Sample figsize in inches
sns.heatmap(c,cmap="viridis", annot=True, linewidths=.5, ax=ax,fmt='.1f',linecolor ='black')


# In[ ]:


# Dropping highly correlated variables
variables_to_drop = ['deadline_month','created_at_month']
df = df.drop(columns=variables_to_drop)


# In[ ]:


# Converting launched_at_month into seasons
# A dictionary to map months to seasons directly
seasons_map = {
    1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
}

# Directly map the 'launched_at_month' to 'launch_at_season', then convert to category
df['launch_at_season'] = df['launched_at_month'].map(seasons_map).astype('category')

# Drop the 'launched_at_month' column as it's no longer needed
df.drop(columns=['launched_at_month'], inplace=True)

# Display the data types to verify the 'launch_at_season' is now a category
print(df.dtypes)


# In[ ]:


# Print value counts for 'country' and 'category' columns
print("Country Distribution:\n", df['country'].value_counts())
print("\nCategory Distribution:\n", df['category'].value_counts())

# Dummify all categorical predictors at once
#'drop_first=True' is used to avoid multicollinearity by dropping the first category
categorical_cols = df.select_dtypes(include='category').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df.info()


# In[ ]:


#converting state to binary
df['state'] = df['state'].map({'successful': 1, 'failed': 0})
df['state'] = df['state'].astype(int)
df.dtypes


# In[ ]:


# Splitting the dataset into features (X) and the target variable (y)
X = df.drop('state', axis=1)
y = df['state']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# In[ ]:


# Build the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)

# Extract feature importances
importances = random_forest.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout() 
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Making predictions on the test set
y_pred = random_forest.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# In[ ]:


###2nd Model:CLUSTERING


# In[3]:


# Importing the data
df = pd.read_excel('C:/Users/ranya/OneDrive/Bureau/DATA ANALYTICS/DATA MINING/Kickstarter.xlsx')
df.head()


# In[4]:


clustering_df = df
clustering_df.dtypes


# In[5]:


# Changing state and creating 'usd_goal'
clustering_df = clustering_df[(clustering_df['state'] == 'failed') | (clustering_df['state'] == 'successful')]
clustering_df['usd_goal']=clustering_df['goal']*clustering_df['static_usd_rate']


# In[6]:


# Creating 'launch_at_season'
def classify_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
clustering_df['launched_at_season'] = clustering_df['launched_at_month'].apply(classify_season)


# In[7]:


# Dropping variables ; keeping the relevant pre + post launch
to_drop = ['deadline','state_changed_at','created_at','launched_at','goal','static_usd_rate',
           'disable_communication','pledged','id','name','name_len','blurb_len',
           'deadline_hr','state_changed_at_hr','created_at_hr','launched_at_hr',
           'state_changed_at_month','created_at_month','deadline_month','state_changed_at_yr',
           'deadline_yr','created_at_yr','state_changed_at_day','created_at_day','deadline_day',
           'state_changed_at_weekday','created_at_weekday','launched_at_weekday',
           'launched_at_month','launched_at_day','launched_at_yr','deadline_weekday','currency']
clustering_df = clustering_df.drop(columns=to_drop)
clustering_df.dtypes


# In[8]:


# Dropping missing values
clustering_df=clustering_df.dropna()

# Categorical as type category
for column in clustering_df.columns:
    if clustering_df[column].dtype == 'object' or clustering_df[column].dtype == 'bool':
        clustering_df[column] = clustering_df[column].astype('category')
clustering_df['launched_at_season'] = clustering_df['launched_at_season'].astype('category')
clustering_df.dtypes


# In[9]:


from sklearn.preprocessing import StandardScaler

# Standardize numerical variables
numerical_cols = clustering_df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = clustering_df.select_dtypes(include=['category']).columns

categorical_df = clustering_df[categorical_cols].copy()


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_numerical = scaler.fit_transform(clustering_df[numerical_cols])
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_cols, index=clustering_df.index)


# In[11]:


scaled_df = pd.concat([scaled_numerical_df, categorical_df], axis=1)
scaled_df.dtypes

# Getting categorical column indices
categorical_indices = [scaled_df.columns.get_loc(col) for col in categorical_cols]


# In[12]:


get_ipython().system('pip install kmodes')
from kmodes.kprototypes import KPrototypes
# 'Huang' initialization method is chosen for categorical variables
# It's more refined than 'Cao' as it considers the mode for initialization
# Set the number of initializations to 5 for stability of results
# Enable verbose mode for algorithm progress information
# Set a random state for reproducibility
kmixed = KPrototypes(n_clusters=2, init='Huang', n_init=5, verbose=2, random_state=0)

# Fit the model to the data ('scaled_df') and simultaneously predict the clusters
# 'categorical' is a list of indices for categorical features in 'scaled_df'
cluster_labels = kmixed.fit_predict(scaled_df, categorical=categorical_indices)

# Count the occurrences of each cluster label to understand the distribution
cluster_distribution = pd.Series(cluster_labels).value_counts()

# Retrieve the centroids of the clusters from the fitted model
# The result will include both numerical and categorical feature centroids
cluster_centroids = kmixed.cluster_centroids_

# Create a DataFrame for the centroids with columns corresponding to 'scaled_df'
# This DataFrame will aid in interpreting the cluster centers
columns = list(scaled_df.columns) 
centroid_df = pd.DataFrame(cluster_centroids, columns=columns)

# Display the DataFrame containing the centroids for inspection
centroid_df


# In[13]:


kmixed.gamma # Optimal weighting (out of 5 runs) is more weight on categorical features (equal would be 0.5)

overall_dissimilarity = kmixed.cost_
overall_dissimilarity

scaled_df['Cluster'] = cluster_labels

# First, select only numeric columns
numeric_scaled_df = scaled_df.select_dtypes(include=[np.number])
# Then, perform the grouping and calculate the standard deviation on this filtered dataframe
std_within_cluster = numeric_scaled_df.groupby('Cluster').std()
std_within_cluster


# In[14]:


# Silhouette score = 0.67
# clusters are relatively well-separated
# Data points within each cluster on average,closer to their own cluster center than to other cluster centers. 
from sklearn.metrics import silhouette_score
s_score = silhouette_score(scaled_df.select_dtypes(include=[np.number]), cluster_labels,random_state=0)
print("Silhouette Score:", s_score)

# Model Gamma
model_gamma = kmixed.gamma
print("Gamma used by the model:", model_gamma)

