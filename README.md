# DS_Portfolio
Data Science projects that I've worked on


# [**PROJECT 1: CREDIT SCORING CLASSIFICATION**](https://github.com/techtonyx/DS_Portfolio)
---

**Problem statement**

In the Credit Scoring ﬁeld, there are two major branches: Application Scoring and Behavioural Scoring.
Both are tools to screen the risk proﬁle of a client, but used in different scenarios. Application Scoring
(also known as Acquisition Scoring) is mainly used to determining whether or not a loan request should
be accepted or not by assessing its creditworthiness, while Behavioural Scoring aims to monitor the
likelihood of default for an existing credit, whose result is further used as an input in calculating
regulatory capital requirement for the bank. 

Depending on the purpose of a loan, there are various types of credits in the market. A common
example is the Mortgage loan which ﬁnances the purchase of a house for individuals. Besides granting
loans to individuals, banks also grant loans to professionals and companies (mainly SME for ABB) to
support their business. Such loans are called Professional loans. The objective of this data challenge is
then to develop an Applica;on Scoring model for Professional loans. Granting credits takes a part of risk,
as the client may default. Then, it is key to grant the credits to the good proﬁles. This is done at
acceptation thanks to acquisition model, on which acceptance criteria are set. Such model assess the
credit worthiness of the client. This is traditionally done by assessing their probability of default. In
another words, Application Scoring is basically a classification problem where we want to distinguish
between ‘good’ clients and ‘bad’ clients and grant loans only to ‘good’ clients. There may be diﬀerent
‘bad’ deﬁnitions, but generally, ‘bad’ clients are deﬁned as those who are likely to default on his credit
obligations over a particular time period (24 months in this case) after realization of a new credit.
Application Scoring model is hence crucial for a bank in the sense that it helps reduce the overall risk
exposure when lending money to customers. 

**Target variable**

CLASS 0: No default (has not defaulted credit obligations over a period of 24 months) 

CLASS 1: Default (defaulted credit obligations over a period of 24 months)

Default rate in dataset is 2,8%.  

**Features**
About 32.000 professional credits with 43 features are included in the dataset. The type of features are
the following: 

 
- Company informa;on (i.e. NACE, months in business,...) 
 
- Financial data (i.e. cash ﬂow, monthly income, average posi;ve saldo savings) 
 
- Socio-demographic (i.e. marital status, age group,...) 
 
- Behavioural features (i.e. behavioural score) 

---
## 1. Business Understanding
AXA Bank Belgium is a bank which (among others) caters to professionals and companies via provision of
loans. While the specific use of the loan may vary given the different needs of the applicants, any loan
that serves to support businesses may all fall under the category of Professional Loans. Upon accepting
such a loan request, AXA Bank consequently assumes the risk of loss incurred from customers defaulting
on credit payment; hence, it is highly important that the bank minimize its exposure to this risk as much
as possible through prudent selection of loan application.

To this extent, the practice of Application Scoring serves as grounds for the bank’s decision in accepting
or rejecting a particular loan request following an evaluation of its creditworthiness. More specifically,
Application Scoring calculates the probability that a new loan application would result in either a default
or complete payment of the loan amount to be granted. It is therefore in the bank’s interest to acquire
reliable probability estimates that could distinguish between potentially favorable and unfavorable loan
applications.

Application Scoring presents a clear data mining problem of class probability estimation which involves
predicting the probability that a given instance will fall within a class. In the case of the AXA Data Science
Challenge, the objective of the data mining process is then to build a classifier to model the probability
of a new loan application resulting in a default or successful repayment of its credit obligation.
Using domain-knowledge of the credit-scoring field, the students have inferred the following preliminary
set of features that are deemed to be informative in predicting an applicant's ability to repay the loan.
This preliminary feature set however, will not be revealed or discussed any further as it does not reflect
the students’ final decision in feature selection due to the implementation of other selection criteria as
will be described in the coming section.

## 2. Data Understanding
An analysis of the DSC_2021_Training dataset has resulted in the following issues of interest:
1. The dataset is comprised of 97.13% negative (non-default) and 2.87% positive cases (default),
resulting in extreme skewness of the target variable frequency distribution
2. Outliers are present in the dataset and were identified by generating a box plot for each feature
3. Multicollinearity between features is ignored as it does not damage the prediction’s
precision.
4. The table below distinguishes between continuous and discrete features of the dataset

![image](https://user-images.githubusercontent.com/117380503/225650155-5ea16d42-3c27-4843-98ad-a581136deffb.png)


## 3. Data Preparation
Data Preparation steps taken for the Data Science Challenge was conducted in the specific order of
explanation seen below
Feature Selection: Table 2.2 also serves to show 30 out of the total 43 features available in the dataset
that were selected for model training (highlighted green). The following points describe how the feature
selection process was conducted

● Domain knowledge: domain-knowledge of credit scoring was first employed to select variables
that are deemed relevant in predicting a loan application’s favorability

● Cut-off Value: Relevant variables were further filtered via an evaluation of their missing value
percentage. To avoid overfitting and misleading prediction results, variables missing ≥ 40% of
their values were removed.

● Redundancy: FINANCIAL_PRODUCT_TYPE_CD, INDUSTRY_CD_3, A1_TOT_DEB_INTEREST_PROF_1_AMT
and A1_OVERDRAWN_DAYS_PROF_24_CNT are removed as they can be appropriately replaced by
other variables which captures the same essence of information.

Split: DSC_2021_Training is split in accordance with the nested-holdout procedure into sub-training,
validation, and test set with a split ratio of 49:21:30.

Data of continuous variables are preprocessed in the following steps:
1. Treatment of Outliers: Outliers are defined as values outside 1.5 · Interquartile Range above the
upper quartile and below the lower quartile. They are subsequently truncated to the particular
feature’s maximum or minimum value. Outlier detection method via box-plots is preferred due
to the frequency distribution of the variables.
2. Normalization: Normalization using MinMaxScaler() is preferred after taking into account the
frequency distribution of the attributes.
3. Treatment of Missing Values: Attributes missing ≥ 40% of their values were removed vertically.
Missing values within the attributes that remain are then replaced with KNNImputer().
Data of discrete variables are preprocessed using the following methods
1. Treatment of Missing Values: Attributes missing ≥ 40% of their values were removed vertically.
Missing values within the attributes that remain are then replaced with its mode
2. Encoding: The variables: Type, Product_Desc, CREDIT_TYPE_CD, A2_MARITAL_STATUS_CD, and
A2_RESIDENT_STATUS_CD are preprocessed using OneHotEncoder
In addition, three variables (INDUSTRY_CD_4, ACCOUNT_PURPOSE_CD, A2_EMPLOYMENT_STATUS_CD)
contain 414, 57 and 34 unique values respectively. They were preprocessed using the Weight of Evidence
encoder as applying One Hot Encoding on each would result in detrimental increase in dimensionality.
Oversampling: To handle extreme skewness inherent in the dataset, the students employed
RandomOverSampler() to its minority class (i.e.Label_Default Y)

## 4. Modeling and Evaluation
The modeling phase involves the following steps:
1. Fitting each model to the sub-training set.
2. Conducting grid search on the validation set for hyperparameter optimization.
3. Calculating each model’s accuracy and Area Under the Curve (AUC) value on the validation set.
4. Selecting the model with the highest AUC value to then apply it to new unseen data points (i.e.
our test set). The following table shows the results of applying our chosen models to the
validation set:

![image](https://user-images.githubusercontent.com/117380503/225650483-b1902d55-d938-4f17-942e-fb2dd771ac3d.png)

5. Having produced the highest AUC value in the validation set, the students’ Random Forest
model was then applied to the test set. The resulting metrics are accuracy of 0.97465912014407
and an AUC value of 0.8871642499998433.
Training the Random Forest model using the preprocessed version of the entire DSC_2021_Training was
conducted as the final step of the nested-holdout procedure. The model was then deployed to the
DSC_2021_Test dataset to produce the required probability scores.

## 5. Conclusion
The outcome of our data mining process in the AXA Data Challenge is a result of maintaining careful
observance of the tools and concepts provided in the textbook, class, Python Tutorial and other reliable
resources. However, given the limited amount of time available and limitations to our own personal
capacity, the students admit that there is room for improvement to the results of our efforts (including
our best model and its properties).

---

# PROJECT 2: Crimes in Boston Dataset (from Kaggle)

Datasets 
Below there are some inspirations for each step, but you are free (and advised
to) include new ideas in your work:  
Crimes in Boston  
Link: [Crimes in Boston](https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston)
 
Overview:  
Crime incident reports are provided by Boston Police Department (BPD) to document
the initial details surrounding an incident to which BPD officers respond. This is a dataset
containing records from the new crime incident report system, which includes a reduced
set of fields focused on capturing the type of incident as well as when and where it
occurred.
Additionally, you are  demographic information on the district’s ethnicity, taken from
https://statisticalatlas.com/place/Massachusetts/Boston/Race-and-Ethnicity, where
other demographics can be found as well. The district code names are taken from
https://bpdnews.com/districts .
  
---
## **Pre-processing**

The Crimes in Boston is a database which records incidents reported by the Boston Police Department between June 2015 and September 2018. The record contains incidents of crimes as well as other events such as the recovery of a lost vehicle, medical assistance to a person and missing person reports. We loaded the data into Google Colab because this is an accessible tool and it allowed us to work together. The dataset consists of 319.073 rows and 17 columns. We decided to delete the columns Lat (latitude), Long (Longitude) and Location from the dataset, as location data has already been provided by the DISTRICT, REPORTING_AREA and STREET columns. Moreover, these columns had more missing values than other columns (about 20.000 instances). We also deleted Occured_on_date, because we have all the time data (year, month and day of the week, hour of the incident, etc) in separate columns already.

Further, we processed the data for missing values. There were no instances where the incident number or offense code were missing. However, there were some instances where location data was missing. We decided to delete rows that had missing values for District, Reporting Area and Street, because we would have no location data at all for those instances. In total, we dropped 1.764 rows. For the column Shooting, there were many missing values, or at least empty cells. However, the cells that were not empty all had the value ‘Y’. We therefore made the assumption that non-empty cells, containing ‘Y’, indicate the use of firearms in the particular incidents, while empty cells meant no firearms were used during the incident. We modified this column so ‘Y’ cells were denoted by a ‘1’ and previously empty cells were denoted by a ‘0’. In other columns with missing values, we replaced them with the most common one (in case of UCR_part), or the most common one in that district (for Street). 

Furthermore, we noticed that sometimes, the same incident number occurs more than once in the dataset. However, it seems that in those cases, the offense codes or offense descriptions were different, meaning that the person committed multiple crimes during the same incident. 


Target Variable = OFFENSE_CODE_GROUP

Drop Variables with the same information
- OCCURED_ON_DATE
- Lat
- Long
- Location
- OFFENSE_CODE
- OFFENSE_DESCRIPTION

Encode High-cardinality variables 
- REPORTING_AREA
- STREET

Encode discrete variables into numbers

## Exploratory Analysis
# Statistical Insights
Before we started with the Machine Learning part, we wanted to gather some insights into the data. We plotted the number of incidents per day of the week, which shows that the number of incidents is roughly divided equally across the week, although there seem to be slightly more incidents on Fridays. We also looked at the spread of incidents across a day. As can be seen on Figure 1, the number varies considerably throughout the day, with a peak at 17:00. For this reason, we decided it might be insightful to group the hours of the day into four parts: Night (00:00-05:59), Morning (06:00-11:59), Afternoon (12:00-17:59) and Evening (18:00-23:59). We chose this distribution to obtain bins of equal width. In order to do this, we created a new variable part_of_day. We can conclude that most incidents happen in the afternoon (see Figure 2), and the least incidents happen at night. This might be counterintuitive at first, however we have to remember that the dataset does not only consist of crimes, but also of incidents like Motor Vehicle Accidents, which are probably more likely to happen during commute hours. 

*Figure1*
![Figure1](https://user-images.githubusercontent.com/117380503/225674752-9d258e18-c519-4ac9-b573-5d38300981ab.png)
*Figure2*
![Figure2](https://user-images.githubusercontent.com/117380503/225675117-1f4d75b3-b0ff-49e5-bd9f-504cf1ea7a3d.png)

Furthermore, we looked into the amount of incidents per district. When we do a simple count of the amount of incidents per district, the districts Roxbury, Dorchester and South End make up the top three, as is visible in Figure 3. Of course, it would make sense that more incidents happen in districts with a higher population. Therefore, we compared the amount of incidents in a district with the population of that district. We extracted the population numbers from the Ethnicity_per_Neighborhood.csv file. Figure 4 shows the results; the top three now consists of Downtown & Charlestown, South End and Mattapan. Downtown & Charlestown is the remarkable outlier with almost 2.3 incidents per inhabitant. However, since this is the center of the city, it is not unlikely that this is in fact correct. This could be important information for the Police Department; it shows which district has the highest crime rate per capita and, therefore, might need more police forces. Additionally, we do not immediately see a link to ethnicity here. In the top three districts for total amount of incidents per capita, there are enormous differences in the composition of the communities, so a direct link seems far-fetched. We did not investigate this further as the reasons and mechanisms behind crime occurrence are very broad and there are a myriad of factors that have influence on it. Moreover, most of the events in the dataset are not crimes but other incidents and so linking ethnicity to this is probably not ideal.

*Figure3*
![Figure3](https://user-images.githubusercontent.com/117380503/225675676-79fff794-3c35-4d89-849a-2b10d4d916c5.png)
*Figure4*
![Figure4](https://user-images.githubusercontent.com/117380503/225675690-57b44df7-e268-4921-ad6e-3d75b76a4501.png)

Lastly, we looked at which incidents happened most often in Boston and also what UCR category occurred most often. The most common incident in Boston in this time period was Motor Vehicle Accident Response (so, not a crime), followed by Larceny and Medical Assistance. We also see this reflected in the UCR categories: the respective distributions for Part One, Part Two and Part Three were 61.411, 97.055 and 157.531. There were another 1.221 incidents under the category Other. This is actually a good sign: the most common incidents were in category three, which consists of minor crimes but mostly incidents such as motor vehicle accidents and verbal disputes. Category one, which has major crimes such as homicide, rape, aggravated assault and burglary, is the least common category aside from Other, which mostly consists of the recovery of stolen vehicles and license plates. The conclusion is that most incidents that the Police Department responds to aren’t crimes, which is a good thing, of course. 




