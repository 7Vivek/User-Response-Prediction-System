#!/usr/bin/env python
# coding: utf-8

# ## Load Libraries

# In[1]:


import numpy as np                    # Linear Algebra
import pandas as pd                   # Data processing 
import matplotlib.pyplot as plt       # Visualizations
import seaborn as sns                 # Visualizations
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix 
import warnings                       # Hide warning messages
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data

# In[2]:


# Reading the file 
df = pd.read_csv(r"C:\Users\Vivek 6666\Downloads\advertising.csv") 


# ## Examine the data

# In[3]:


df.head(10) # Checking the 1st 10 rows of the data


# ## Extracting Datetime Variables
# 
# - Utilizing ```timestamp``` feature to better understand the pattern when a user is clicking on a ad.

# In[4]:


# Extract datetime variables using timestamp column
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
# Converting timestamp column into datatime object in order to extract new features
df['Month'] = df['Timestamp'].dt.month 
# Creates a new column called Month
df['Day'] = df['Timestamp'].dt.day     
# Creates a new column called Day
df['Hour'] = df['Timestamp'].dt.hour   
# Creates a new column called Hour
df["Weekday"] = df['Timestamp'].dt.dayofweek 
# Creates a new column called Weekday with sunday as 6 and monday as 0
# Other way to create a weekday column
#df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday()) # Monday 0 .. sunday 6
# Dropping timestamp column to avoid redundancy
df = df.drop(['Timestamp'], axis=1) # deleting timestamp


# In[5]:


df.head() # verifying if the variables are added to our main data frame


# ## Basic model building based on the actual data

# In[6]:


# Importing train_test_split from sklearn.model_selection family
from sklearn.model_selection import train_test_split


# In[7]:


# Assigning Numerical columns to X & y only as model can only take numbers
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']


# In[8]:


# Splitting the data into train & test sets 
# test_size is % of data that we want to allocate & random_state ensures a specific set of random splits on our data because 
#this train test split is going to occur randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
# We dont have to use stratify method in train_tst_split to handle class distribution as its not imbalanced and does contain equal number of classes i.e 1's and 0's
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ## Building a Basic Model

# In[9]:


# Import LogisticRegression from sklearn.linear_model family
from sklearn.linear_model import LogisticRegression


# In[10]:


# Instantiate an instance of the linear regression model (Creating a linear regression object)
logreg = LogisticRegression()
# Fit the model on training data using a fit method
model = logreg.fit(X_train,y_train)
model


# ## Predictions

# In[11]:


# The predict method just takes X_test as a parameter, which means it just takes the features to draw predictions
predictions = logreg.predict(X_test)
# Below are the results of predicted click on Ads
predictions[0:20]


# ## Performance Metrics
# 
# - Now, we need to see how far our predictions met the actual test data **(y_test)** by performing evaluations using classification report & confusion matrix on the target variable and the projections.
# 
# 
# - **confusion matrix** is used to evaluate the model behaviour from a matrix.
# 
# 
# - **TP** - True Positive **TN** - True Negative **FP** - False Positive **FN** - False Negative.
# 
# - True Positive is the proportion of positives that are correctly identified.
# - Similarly, True Negative is the proportion of negatives that are correctly identified. 
# - False Positive is the condition where we predict a result that it doesn't fulfil. 
# - Similarly, False Negative is the condition where the prediction failed when it was successful.
# 
# 
# - If we want to calculate any specific value, we can do it from the confusion matrix directly.
# 
# 
# - **classification_report** will tell us the precision, recall value's accuracy, f1 score & support. This way, we don't have to read it ourselves from a confusion matrix.
# 
# 
# - **precision** is the fraction of retrieved values that are relevant to the data. The precision is the ratio of tp / (tp + FP).
# 
# 
# - **recall** is the fraction of successfully retrieved values that are relevant to the data. The recall is the ratio of tp / (tp + fn).
# 
# 
# - **f1-score** is the harmonic mean of precision and recall where a score reaches its best value at one and worst score at 0.
# 
# 
# - **support** is the number of occurrences of each class in y_test.

# In[12]:


# Importing classification_report from sklearn.metrics family
from sklearn.metrics import classification_report

# Printing classification_report to see the results
print(classification_report(y_test, predictions))


# In[13]:


# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))


# ## Results for Basic Model
# 
# - The results from the evaluation are as follows:
# 
# **Confusion Matrix:**
# 
# - The users that are predicted to click on commercials and the clicked users were **140**, the people who were expected not to click on the commercials and did not click on them were **129**.
# 
# - The people who were predicted to click on commercials and did not click on them are **6**, and the users who were not expected to click on the commercials and clicked on them are **25**.
# 
# - We have only a few mislabelled points that are not wrong from the given size of the dataset.
# 
# **Classification Report:**
# 
# - From the report obtained, the precision & recall are **0.90**, which depicts the predicted values are **90%** accurate. 
# The probability that the user can click on the commercial is **0.90**, which is a good precision value to get a good model.

# ## Feature Engineering

# In[14]:


new_df = df.copy() # just to keep the original dataframe unchanged


# In[15]:


# Creating pairplot to check effect of datetime variables on target variable (variables which were created)
pp = sns.pairplot(new_df, hue= 'Clicked on Ad', vars = ['Month', 'Day', 'Hour', 'Weekday'], palette = 'mako',height = 2,aspect=1.5)


# #### There don't seem to be any effect of the month, day, weekday and hour on the target variable.

# In[16]:


# Dummy encoding on Month column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Month'], prefix='Month')], axis=1) 
# Dummy encoding on weekday column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Weekday'], prefix='Weekday')], axis=1)


# In[17]:


# Creating buckets for hour columns based on EDA part
new_df['Hour_bins'] = pd.cut(new_df['Hour'], bins = [0, 5, 11, 17, 23], 
                        labels = ['Hour_0-5', 'Hour_6-11', 'Hour_12-17', 'Hour_18-23'], include_lowest= True)


# In[18]:


# Dummy encoding on Hour_bins column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Hour_bins'], prefix='Hour')], axis=1)


# In[19]:


# Feature engineering on Age column
plt.figure(figsize=(16,6))
sns.barplot(new_df['Age'],df['Clicked on Ad'], ci=None,palette = 'GnBu')
plt.xticks(rotation=90)


# In[20]:


# checking bins
limit_1 = 18
limit_2 = 35

x_limit_1 = np.size(df[df['Age'] < limit_1]['Age'].unique())
x_limit_2 = np.size(df[df['Age'] < limit_2]['Age'].unique())

plt.figure(figsize=(16,6))
#sns.barplot(df['age'],df['survival_7_years'], ci=None)
sns.countplot('Age',hue='Clicked on Ad',data=df)
plt.axvspan(-1, x_limit_1, alpha=0.25, color='green')
plt.axvspan(x_limit_1, x_limit_2, alpha=0.25, color='red')
plt.axvspan(x_limit_2, 50, alpha=0.25, color='yellow')

plt.xticks(rotation=90)


# In[21]:


# Creating Bins on Age column based on above plots
new_df['Age_bins'] = pd.cut(new_df['Age'], bins=[0, 18, 30, 45, 70], labels=['Young','Adult','Mid', 'Elder'])


# In[22]:


plt.figure(figsize=(16,6))
sns.countplot('Age_bins',hue='Clicked on Ad',data= new_df,palette = 'PuBu') # Verifying the bins by checking the count


# In[23]:


# Dummy encoding on Age column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Age_bins'], prefix='Age')], axis=1) 


# In[24]:


# Dummy encoding on Column column based on EDA
new_df = pd.concat([new_df, pd.get_dummies(new_df['Country'], prefix='Country')], axis=1)


# In[25]:


# Remove redundant and no predictive power features
new_df.drop(['Country', 'Ad Topic Line', 'City', 'Day', 'Month', 'Weekday', 
             'Hour', 'Hour_bins', 'Age', 'Age_bins'], axis = 1, inplace = True)
new_df.head() # Checking the final dataframe


# ## Building Logistic Regression Model

# In[26]:


X = new_df.drop(['Clicked on Ad'],1)
y = new_df['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[27]:


# Standarizing the features
from  sklearn.preprocessing  import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[28]:


import  statsmodels.api  as sm
from scipy import stats

X2   = sm.add_constant(X_train_std)
est  = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


# - We can see that the feature **Male(Gender)** does not contribute to the model (i.e., see x4), so we can remove that variable from our model. After removing the variable if the Adjusted R-squared has not changed from the previous model. Then we could conclude that the feature indeed was not contributing to the model. Looks like the contributing features for the model are:
# 
#  - Daily Time Spent on site
#  - Daily Internet Usage
#  - Age
#  - Country
#  - Area income

# In[29]:


# Applying logistic regression model to training data
lr = LogisticRegression(penalty="l2", C= 0.1, random_state=42)
lr.fit(X_train_std, y_train)
# Predict using model
lr_training_pred = lr.predict(X_train_std)
lr_training_prediction = accuracy_score(y_train, lr_training_pred)

print( "Accuracy of Logistic regression training set:",   round(lr_training_prediction,3))


# In[30]:


#Creating K fold Cross-validation 
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(lr, # model
                         X_train_std, # Feature matrix
                         y_train, # Target vector
                         cv=kf, # Cross-validation technique
                         scoring="accuracy", # Loss function
                         n_jobs=-1) # Use all CPU scores
print('10 fold CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[31]:


from sklearn.model_selection import cross_val_predict
fig = plt.figure(figsize = (18,6))
print('The cross validated score for Logistic Regression Classifier is:',round(scores.mean()*100,2))
y_pred = cross_val_predict(lr,X_train_std,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="tab20")
plt.title('Confusion_matrix', y=1.05, size=15)


# ## Modelling with Random Forests

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


rf = RandomForestClassifier(criterion='gini', n_estimators=400,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=42,n_jobs=-1)
rf.fit(X_train_std,y_train)
# Predict using model
rf_training_pred = rf.predict(X_train_std)
rf_training_prediction = accuracy_score(y_train, rf_training_pred)
x`
print("Accuracy of Random Forest training set:",   round(rf_training_prediction,3))


# In[39]:


kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, # model
                         X_train_std, # Feature matrix
                         y_train, # Target vector
                         cv=kf, # Cross-validation technique
                         scoring="accuracy", # Loss function
                         n_jobs=-1) # Use all CPU scores
print('10 fold CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[40]:


from sklearn.model_selection import cross_val_predict
fig = plt.figure(figsize = (18,6))
print('The cross validated score for Random Forest Classifier is:',round(scores.mean()*100,2))
y_pred = cross_val_predict(rf,X_train_std,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="tab20b")
plt.title('Confusion_matrix', y=1.05, size=15)


# ## Random Forest Feature Importances

# In[41]:


columns = X.columns
train = pd.DataFrame(np.atleast_2d(X_train_std), columns=columns) # Converting numpy array list into dataframes


# In[42]:


# Get Feature Importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances.head(10)


# In[43]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the Feature Importance
sns.barplot(x="importance", y='index', data=feature_importances[0:10],label="Total", color="b",palette='Paired')


# ## Test Models Performance

# In[44]:


print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, lr.predict(X_test_std))

print ("Logistic Regression AUC = %2.2f" % lr_auc)
print(classification_report(y_test, lr.predict(X_test_std)))

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test_std))

print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test_std)))


# #### We can observe that random forest has higher accuracy compared to logistic regression model in both test and train data sets.

# #### Fit the model on the entire data and save the predictions on csv file

# In[46]:


final_predictions = lr.predict(X)


# In[47]:


predic = pd.DataFrame({'Predict_Click on Ad': final_predictions})
output=pd.concat([X,predic], axis=1)
output.head()


# In[ ]:


#output.to_csv('Ad_predictions.csv', index=False)
#print("Your output was successfully saved!")


# ## ROC Graph

# In[45]:


# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test_std)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test_std)[:,1])

fig = plt.figure(figsize = (16,6))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

