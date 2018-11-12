import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap


# Functions provided during the course
def make_meshgrid(ax, h=.02):
    # x_min, x_max = x.min() - 1, x.max() + 1
    # y_min, y_max = y.min() - 1, y.max() + 1
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):

    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)


# Create your df here:
df = pd.read_csv("profiles.csv")

###############################################################################
###############################################################################
'''Visualize data'''
# Age distribution on web page
'''plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()'''

# Distinguish between male and female.
# df_female = df.loc[(df['sex'] == 'f')]
# df_male = df.loc[(df['sex'] == 'm')]
# print(len(df_female))
# print(len(df_male))

# Male and Female comparison - jobs
'''frame_counts = pd.DataFrame(df_female['job'].value_counts(normalize=True))
frame_counts['Male'] = pd.DataFrame(df_male['job']
                                    .value_counts(normalize=True))
frame_counts.columns = ['Female', 'Male']

frame_counts.plot(kind='bar', width=0.9, color=['r', 'b'])
plt.title('Male and Female Job Comparison')
plt.xlabel("Job")
plt.ylabel("Frequency - Normalized")
plt.show()'''

# Male and Female comparison - body type
'''frame_counts = pd.DataFrame(df_female['body_type'].value_counts(normalize=True))
frame_counts['Male'] = pd.DataFrame(df_male['body_type']
                                    .value_counts(normalize=True))
frame_counts.columns = ['Female', 'Male']

frame_counts.plot(kind='bar', width=0.9, color=['r', 'b'])
plt.title('Male and Female Body Type Comparison')
plt.xlabel("Body Type")
plt.ylabel("Frequency - Normalized")
plt.show()'''

# Male and Female comparison - Income
'''frame_counts = pd.DataFrame(df_female['income'].value_counts(normalize=True))
frame_counts['Male'] = pd.DataFrame(df_male['income']
                                    .value_counts(normalize=True))
frame_counts.columns = ['Female', 'Male']

frame_counts.plot(kind='bar', width=0.9, color=['r', 'b'])
plt.title('Male and Female Income Comparison')
plt.xlabel("Income")
plt.ylabel("Frequency - Normalized")
plt.show()'''

# Histograms comparing height distribution for male and female
'''hist = df_male['height'].hist(bins = 100, alpha = 0.5,
                                 label='Male', density=True)
hist = df_female['height'].hist(bins = 100, alpha = 0.5,
                                label='Female', density=True)
plt.legend()
plt.title('Difference in Height Males Compared to Females - histogram')
plt.xlim(55, 80)
plt.xlabel("Height [feet]")
plt.ylabel("Normalized frequency")'''

###############################################################################
###############################################################################
''' Questions for Classification:

As seen from the figures, differences between man and woman are significant.

For instance, females dominates over males at jobs of administrative/clerical
or medicine/health type. Men on the other hand completely dominates in
military, computer/harware/software industry and in construction/craftmanship.

Differences can be also seen in income, as women's wages seem to be generally
lower.

Final figure shows height distribution for men and women. Males are generally
considered as higher.

Main question of whole project is: Can be sex determined from the data set?

Question 1: Can sex be determined just by income and height?

Question 2:
How will the predictive power of models improve, if job and body_type
variables are introduced?

'''
###############################################################################
###############################################################################
'''Augment data - for classification part of project'''
# print(df['job'].value_counts())
# print(df['income'].value_counts())
# print(df['sex'].value_counts())

sex_mapping = {"m": 0, "f": 1}
job_mapping = {"other": 0, "student": 1, "science / tech / engineering": 2,
               "computer / hardware / software": 3,
               "artistic / musical / writer": 4,
               "sales / marketing / biz dev": 5, "medicine / health": 6,
               "education / academia": 7, "executive / management": 8,
               "banking / financial / real estate": 9,
               "entertainment / media": 10, "law / legal services": 11,
               "hospitality / travel": 12, "construction / craftsmanship": 13,
               "clerical / administrative": 14, "political / government": 15,
               "rather not say": 16, "transportation": 17, "unemployed": 18,
               "retired": 19, "military": 20}
body_type_mapping = {"average": 0, "fit": 1, "athletic": 2, "thin": 3,
                     "curvy": 4, "a little extra": 5, "skinny": 6,
                     "full figured": 7, "overweight": 8, "jacked": 9,
                     "used up": 10,  "rather not to say": 11}

# Adding new columns to the data frame
df["job_code"] = df.job.map(job_mapping)
df["sex_code"] = df.sex.map(sex_mapping)
df["body_code"] = df.body_type.map(body_type_mapping)

# Create a new data frame
# Choose columns to work with
choose = ["income", "height"]
choose_Q2 = ["income", "height", "body_code", "job_code"]

# Add all features to one separate data frame
feature_data = df[choose]
feature_dataQ2 = df[choose_Q2]

# Question 1 - Drop all NaN in the data set and normalize data
feature_data = feature_data.dropna(subset=choose)
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_normalized = min_max_scaler.fit_transform(x)

# Question 2 - Drop all NaN in the data set and normalize data for
feature_dataQ2 = feature_dataQ2.dropna(subset=choose_Q2)
xQ2 = feature_dataQ2.values
min_max_scalerQ2 = MinMaxScaler()
x_normalizedQ2 = min_max_scalerQ2.fit_transform(xQ2)

# Question 1 - Create dataframe from normalized data
feature_data = pd.DataFrame(x_normalized, columns=feature_data.columns)
# Question 1 - Create labels from data set by dividing data to male and female
sex_labels = df.dropna(subset=choose)["sex_code"].reset_index(drop=True)
# Question 1 - Split data into training group and test group
train_data, test_data, train_labels, test_labels = train_test_split(feature_data, sex_labels, random_state=1)


# Question 2 - Create dataframe from normalized data
feature_dataQ2 = pd.DataFrame(x_normalizedQ2, columns=feature_dataQ2.columns)
# Question 2 - Create labels from data set by dividing data to male and female
sex_labelsQ2 = df.dropna(subset=choose_Q2)["sex_code"].reset_index(drop=True)
# Question 2 - Split data into training group and test group
train_dataQ2, test_dataQ2, train_labelsQ2, test_labelsQ2 = train_test_split(feature_dataQ2, sex_labelsQ2, random_state=1)

###############################################################################
###############################################################################
'''Classification - K-Neighbors supervized learning'''


def K_value(train_data, test_data, train_labels, test_labels):
    # Function determines proper K value
    # Defining range
    k_range = range(1, 151)

    # List for storing data
    acc_res = []
    pre_res = []
    rec_res = []
    score_list = []
    start = time.time()
    # Defining dictionary to find maximum score
    best = {'score': 0, 'K': 1}

    # Start timer to find out how long it will take to compute
    start = time.time()

    # For loop to find best K
    for k in k_range:
        print(k)
        KN_classifier = KNeighborsClassifier(n_neighbors=k)
        KN_classifier.fit(train_data, train_labels)
        Predicted = KN_classifier.predict(test_data)
        # Filling up empty lists
        acc_res.append(accuracy_score(test_labels, Predicted))
        pre_res.append(precision_score(test_labels, Predicted))
        rec_res.append(recall_score(test_labels, Predicted))
        # Defining score
        score = KN_classifier.score(test_data, test_labels)
        if (score > best['score']):
            best['score'] = score
            best['K'] = k
        score_list.append(score)
        print(best['K'], best['score'])

    # End timer
    end = time.time()

    # Print results
    print(best)
    print("Finding a proper K value took {} s".format(round(end - start, 2)))

    '''Printing figures of K-Neighbors'''

    # K range over Score
    plt.plot(k_range, score_list)
    plt.plot(k_range, pre_res)
    plt.plot(k_range, rec_res)
    plt.xlabel("K")
    plt.ylabel("Score, Precision, Recall")
    plt.title("Sex K-Neigbours K analysis")
    plt.legend(['Score', 'Precision', 'Recall'])
    plt.show()


def k_neighbor_analysis(train_data, test_data, train_labels, test_labels, n, q):
    start = time.time()
    KN_classifier = KNeighborsClassifier(n_neighbors=n)
    KN_classifier.fit(train_data, train_labels)
    Predicted = KN_classifier.predict(test_data)
    acc_res = (accuracy_score(test_labels, Predicted))
    pre_res = (precision_score(test_labels, Predicted))
    rec_res = (recall_score(test_labels, Predicted))
    end = time.time()
    print("############")
    print("K-Neighbors")
    print("Question {} results".format(q))
    print("Score of the K-Nearest Neighbors model with K = {}: {}".format(n, KN_classifier.score(test_data, test_labels)))
    print("Accuracy: {}, Recall: {}, Precision: {}".format(round(acc_res, 4), round(rec_res, 4), round(pre_res, 4)))
    print("Calculation of K-Nearest Neighbors itself took {} s".format(round(end - start, 2)))


###############################################################################
# Determine K value for Question 1
# K_value(train_data, test_data, train_labels, test_labels)
# {'score': 0.83724809822501, 'K': 37}
# Finding a proper K value took 760.29 s
###############################################################################

###############################################################################
# Final K-Neighbors model with K = 37
n_neighbors = 37
k_neighbor_analysis(train_data, test_data, train_labels, test_labels, n_neighbors, 1)
# Question 1 results:
# Score of the K-Nearest Neighbors model with K = 37: 0.83724809822501
# Accuracy: 0.8372, Recall: 0.8055, Precision: 0.7878
# Calculation of K-Nearest Neighbors itself took 2.41 s
###############################################################################

'''Create a figure that shows decision boundary for K-Neighbors'''
# Inspiration is from https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
# Create a step size for the mash
h = 0.01
# Create color maps
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#0000FF', '#FF0000'])

# Create a model of Neighbours Classifier and fit the data.
model = KNeighborsClassifier(n_neighbors, weights='uniform')
model.fit(train_data, train_labels)

# Determine minimum nad maximum for the mesh grid
x_min, x_max = train_data.iloc[:, 0].min() - 0.5, train_data.iloc[:, 0].max() + 0.5
y_min, y_max = train_data.iloc[:, 1].min() - 0.5, train_data.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict data
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot training points
plt.scatter(train_data.iloc[:, 0], train_data.iloc[:, 1], c=train_labels, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Sex Classification with heigh & income (K = {})".format(n_neighbors))
plt.xlabel("Income normalized")
plt.ylabel("Height normalized")
plt.show()

###############################################################################
###############################################################################
'''Quetion 1: Classification - SVC'''


def SVC_gamma_C(train_data, test_data, train_labels, test_labels):
    # Function to determine proper gamma and C values
    # Empty lists for data storing
    gamma_list = []
    C_list = []
    score_list = []
    acc_list = []
    rec_list = []
    pre_list = []

    # Start timer to find out how long it will take to compute
    best = {'score': 0, 'gamma': 1, 'C': 1}

    # Start timer to find out how long it will take to compute
    start = time.time()

    # Nested for loop to determine best possible combination of gamma and C
    for gamma in range(1, 2):
        print(gamma)  # To keep track where are we at the calculation
        for C in range(1, 11):
            start_loop = time.time()
            print(C)  # To keep track where are we at the calculation
            # Create classifier
            classifier = SVC(kernel='rbf', gamma=gamma, C=C)
            # Fit classifier
            classifier.fit(train_data, train_labels)
            score = classifier.score(test_data, test_labels)
            Predicted = classifier.predict(test_data)
            # Update best results
            if (score > best['score']):
                best['score'] = score
                best['gamma'] = gamma
                best['C'] = C
            # Append results
            gamma_list.append(gamma)
            C_list.append(C)
            score_list.append(score)
            acc_list.append(accuracy_score(test_labels, Predicted))
            rec_list.append(recall_score(test_labels, Predicted))
            pre_list.append(precision_score(test_labels, Predicted))
            print(best)  # Print the best possible result so far
            end_loop = time.time()
            print("This iteration took {} s to compute".format(round(end_loop - start_loop, 2)))
    # Stop timer
    end = time.time()
    print(best)
    print("Find best gamma and C values took {} s".format(round(end - start, 2)))
    return gamma_list, C_list, score_list, acc_list, rec_list, pre_list


def SVC_model(train_data, test_data, train_labels, test_labels, gamma, C, q):
    # SVC model with print
    # Start timer
    start = time.time()
    # Create Classifier
    classifier = SVC(kernel='rbf', gamma=2, C=4)
    classifier.fit(train_data, train_labels)
    score = classifier.score(test_data, test_labels)
    Predicted = classifier.predict(test_data)
    acc_SVC = accuracy_score(test_labels, Predicted)
    pre_SVC = precision_score(test_labels, Predicted)
    rec_SVC = recall_score(test_labels, Predicted)
    end = time.time()
    print("############")
    print("SVC")
    print("Question {}".format(q))
    print("Score of the SVC model with gamma = {} and C = {}: {}".format(gamma, C, score))
    print("Accuracy: {}, Recall: {}, Precision: {}".format(round(acc_SVC, 4), round(rec_SVC, 4), round(pre_SVC, 4)))
    print("Calculation of K-Nearest Neighbors itself took {} s".format(round(end - start, 2)))
    return classifier


###############################################################################
# Find best value of gamma and C
gamma_list, C_list, score_list, acc_list, rec_list, pre_list = SVC_gamma_C(train_data, test_data, train_labels, test_labels)
plt.plot(C_list[0:9], score_list[0:9])
plt.plot(C_list[0:9], rec_list[0:9])
plt.plot(C_list[0:9], pre_list[0:9])
plt.xlim(1, 9)
plt.xticks(np.arange(1, 10, step=1))
plt.xticks(np.arange(10), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.legend(["Score", "Recall", "Precision"])
plt.xlabel("C")
plt.ylabel("Score, Recall, Precision")
plt.title("Score Gamma = 1, C changing")
plt.show()
# {'score': 0.8369144534899239, 'gamma': 2, 'C': 4}
# Find best gamma and C values took 1336.31 s
###############################################################################

###############################################################################
'''Final SVC model with gamma = 1 and C = 8'''
# Create subplot for visualization
fig, ax = plt.subplots()
# Create scatter plot income vs height with distinguished males (blue) and females (red)
plt.scatter(x=feature_data["income"], y=feature_data["height"], c=sex_labels, cmap=plt.cm.coolwarm, alpha=0.25)
# Calculate SVC
classifier = SVC_model(train_data, test_data, train_labels, test_labels, 1, 8, 1)
# Draw SVC boundary - thank you Codecademy for sharing!
draw_boundary(ax, classifier)
# Add labels and title
plt.xlabel("Income")
plt.ylabel("Height")
plt.title("Decision boundary")
plt.show()
# Question 1 results:
# Score of the SVC model with gamma = 1 and C = 8: 0.8369144534899239
# Accuracy: 0.8369, Recall: 0.804, Precision: 0.788
# Calculation of K-Nearest Neighbors itself took 62.86 s

###############################################################################
###############################################################################
'''Classification Question number 2
# “How will the predictive power of models improve, if job and body_type
# variables are introduced?
"'''

#########################################
# K-Neighbors
#########################################
# Find best K for K Neighbor analysis
K_value(train_dataQ2, test_dataQ2, train_labelsQ2, test_labelsQ2)
# {'score': 0.8646174519190393, 'K': 39}
# Finding a proper K value took 488.17 s

# Question 2 Final K Neighbor
k_neighbor_analysis(train_dataQ2, test_dataQ2, train_labelsQ2, test_labelsQ2, 39, 2)
# Question 2 results:
# Score of the K-Nearest Neighbors model with K = 39: 0.8646174519190393
# Accuracy: 0.8646, Recall: 0.7823, Precision: 0.8492
# Calculation of K-Nearest Neighbors itself took 1.43 s

#########################################
# SVC
#########################################
# Find best gamma and C for SVC analysis
gamma_list, C_list, score_list, acc_list, rec_list, pre_list = SVC_gamma_C(train_dataQ2, test_dataQ2, train_labelsQ2, test_labelsQ2)
# Plot score, recall and precision
plt.plot(C_list[0:9], score_list[0:9])
plt.plot(C_list[0:9], rec_list[0:9])
plt.plot(C_list[0:9], pre_list[0:9])
plt.xlim(1, 9)
plt.xticks(np.arange(1, 10, step=1))
plt.xticks(np.arange(10), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.legend(["Score", "Recall", "Precision"])
plt.xlabel("C")
plt.ylabel("Score, Recall, Precision")
plt.title("Score Gamma = 1, C changing")
plt.show()
# {'score': 0.86572643276267, 'gamma': 5, 'C': 10}
# Find best gamma and C values took 1680.0 s


# Question 2 Final SVC
SVC_model(train_dataQ2, test_dataQ2, train_labelsQ2, test_labelsQ2, 5, 10, 2)
# Score of the SVC model with gamma = 5 and C = 10: 0.86572643276267
# Accuracy: 0.8657, Recall: 0.7795, Precision: 0.8075
# Calculation of SVC itself took 29.03 s


###############################################################################
###############################################################################
'''Question for Regression
Histogram shows that differences between man and woman (height) are significant

Question 1 Linear Regression:
    Can be sex predicted only by height with linear regression?

Question 2 Multiple Linear Regression:
    Will score of the model improve if additional variable () is introduced?

'''
###############################################################################
###############################################################################
'''Linear Regression:
    Question 3: Can be sex predicted only by height with linear regression?'''

# Choose variables to work with - set a list with the names
choose_Q3 = ["sex_code", "height"]
# Add all features to one separate data frame
feature_data_Q3 = df[choose_Q3]
# Drop all NaN in the data set and normalize data
feature_data_Q3 = feature_data_Q3.dropna(subset=choose_Q3)
# Split data to parts (train, test)
X_train, X_test, y_train, y_test = train_test_split(feature_data_Q3[["height"]], feature_data_Q3[["sex_code"]], test_size = 0.2, random_state = 1)


start = time.time()
# Create a model LinearRegression
model = LinearRegression()
# Fit model
model.fit(X_train, y_train)
# Predict values from the model
predict = model.predict(X_test)
end = time.time()

# Create a plot to vizualize linear regression
# Plot all males with respect to height
plt.plot((feature_data_Q3.loc[feature_data_Q3["sex_code"] == 0]).height, (feature_data_Q3.loc[feature_data_Q3["sex_code"] == 0]).sex_code, 'o', alpha=0.1)
# Plot all females with respect to height
plt.plot((feature_data_Q3.loc[feature_data_Q3["sex_code"] == 1]).height, (feature_data_Q3.loc[feature_data_Q3["sex_code"] == 1]).sex_code, 'o', alpha=0.1)
# Plot the regression line
plt.plot(X_test, predict)
# Set the parameters x label and y label
plt.ylim(-0.5, 1.5)
plt.yticks(np.arange(0, 1.5, step=1), ('Male', 'Female'))
plt.ylabel("Sex")
plt.xlim(40, 100)
plt.xlabel("Height (inches)")
# Set Title
plt.title("Linear Regression")
plt.show()

print("############")
print("Linear Regression")
# Print Slope of model
print("Slope {}".format(model.coef_))
# Print Intercept of model
print("Intercept {}".format(model.intercept_))
# How well it fit train data
print("Train Score: {}".format(round(model.score(X_train, y_train), 4)))
# How well it fit test data
print("Test Score: {}".format(round(model.score(X_test, y_test), 4)))
# Print the time of the calculation itself
print("Calculation of Linear Regression mode itself took {} s".format(round(end - start, 4)))

###############################################################################
###############################################################################
'''Multiple Linear Regression
    Question 4: Will score of the model improve if additional variable () is introduced?

Additional to the height data which were used in linear Regression model
for Multiple Linear Regression model wll be used:
    - body_code         - introduced in classification part of project
    - income            - introduced in classification part of project
    - woman_words_count - count in essays some of the most often used words by women (see reference below)
    - man_words_count   - count in essays some of the most often used words by women (see reference below)

Reference: http://languagelog.ldc.upenn.edu/nll/?p=13873
    '''

# Merge all essays together - name all the variables
essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5",
              "essay6", "essay7", "essay8", "essay9"]

# Merge all essays together - drop all NaN
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Merge all essays together
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

# Most often words used by women and men
# http://languagelog.ldc.upenn.edu/nll/?p=13873
woman_words_list = ["love", "excited", "happy", "chocolate", "cute"]
man_words_list = ["sport", "shit", "guy", "fifa", "ps3"]

# Create columns in the data frame - count all women words and all men words
df["woman_words_count"] = all_essays.apply(lambda x: sum(x.count(y) for y in woman_words_list))
df["man_words_count"] = all_essays.apply(lambda x: sum(x.count(y) for y in man_words_list))

# Choose variables to work with - set a list with the names
choose_Q4 = ["sex_code", "height", "body_code", "income", "woman_words_count", "man_words_count"]

# Add all features to one separate data frame
feature_data_Q4 = df[choose_Q4]

# Drop all NaN in the data set and normalize data
feature_data_Q4 = feature_data_Q4.dropna(subset=choose_Q4)

# Devide data for linear regression
regre_dataQ4 = feature_data_Q4[["height", "body_code", "income", "woman_words_count", "man_words_count"]]
sex_dataQ4 = feature_data_Q4[["sex_code"]]

# Split data to parts (train, test)
X_train, X_test, y_train, y_test = train_test_split(regre_dataQ4, sex_dataQ4, test_size = 0.2, random_state = 1)

# Start timer
start = time.time()
# Create a model LinearRegression
model = LinearRegression()
# Fit model
model.fit(X_train, y_train)
# Predict values from the model
predict = model.predict(X_test)
# How well it fit train data
print(model.score(X_train, y_train))
# How well it fit test data
print(model.score(X_test, y_test))
# Stop timer
end = time.time()

print("############")
print("Multiple Linear Regression")
# Print Slope of model
print("Slope {}".format(model.coef_))
# Print Intercept of model
print("Intercept {}".format(model.intercept_))
# How well it fit train data
print("Train Score: {}".format(round(model.score(X_train, y_train), 4)))
# How well it fit test data
print("Test Score: {}".format(round(model.score(X_test, y_test), 4)))
# Print the time of the calculation itself
print("Calculation of Multiple Linear Regression model itself took {} s".format(round(end - start, 4)))
# Print Scope for each variable used
print(sorted(list(zip(["height", "body_code", "income", "woman_words_count", "man_words_count"], model.coef_)), key = lambda x: abs(x[1]),reverse=True))
###############################################################################
###############################################################################
