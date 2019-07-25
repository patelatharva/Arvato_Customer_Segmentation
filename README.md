## Customer Segmentation Report and Predictive Analysis of Customer Conversion of Mail-Order company for Arvato Financials
### Goal
Arvato Financial Solutions provided demographics dataset of the customers of mail-order sales company in Germany as well as the general population of Germany. The goal of the project is to help the client company of Arvato in identifying the segments of the general population which are highly representative of the existing customers of the company. Such kind of information can help the company in improving the effectiveness of marketing by targeting it to the right segment of population. It can help it better understand the existing customers and optimize the products, operations and services of the company to serve them in a better way. 

The company also provided with additional dataset about the response received to the marketing campaign of the mail-order company. The dataset contains similar attributes as those found in the earlier described datasets containing demographics, age group, financial status and purchasing behaviour etc. The goal is to predict the probability of a new person responding positively to become a customer of mail-order company based on the demographics data provided about that person.

### Approach
I used techniques of PCA (Principal Component Analysis) to reduce the dimensionality of datasets provided by Arvato Financials. I used K-means clustering to identify population groups to see if some cluster can be highly representative of the characteristics of the existing customer base of the company. 

Finally, I build statistical model for predicting the conversion of population sample to become a customer of a mail-order company based on the historical marketing campaign data. Considering the imbalanced nature of the target variable in the dataset, used LigthGBM gradient boosting algorithm as machine learning model. As the number of positive samples in the training dataset is very low as compared to negative responses, which is considered as an imbalanced dataset, the evaluation metric being used here is Receiver operating characteristic Area Under Curve.

### Files:
- `Part 1 Customer Segmentation.ipynb`: Notebook containing steps for preparing segmentation analysis
- `Part 2 Supervised Learning and Part 3 Kaggle.ipynb`: Notebook containing steps for predicting customer conversion of mail-order company
- `DIAS Attributes - Values 2017.xlsx`: Data Dictionary indicating meaning of each column name
- `DIAS Information Levels - Attributes 2017.xlsx`: Data Dictionary indicating meaning, levels and missing value representation of data

### Blog Post:
You can [read the blog post regarding the analysis here](https://medium.com/@patelatharva/identifying-customer-segments-and-predicting-conversions-of-mail-order-sales-company-for-arvato-a139c66686f6).