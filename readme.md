# Analysis of a Store's Customers

*Analysis of a Store's Customers using Data Mining tools.*

Project for the Data Mining course @ UniPi.

## Developers

- [Valerio Mariani](https://github.com/sd3ntato)
- [Antonio Strippoli](https://github.com/CoffeeStraw)

## Tasks
A project consists in data analysis based on the use of data mining tools. The project has to be performed by a team of 2/3 students. It has to be performed by using Python. The guidelines require to address specific tasks and results must be reported in a unique paper. The total length of this paper must be max 20 pages of text including figures. The students must deliver both: paper and well commented Python notebooks.

### 1) Data Understanding and Preparation
1. **Data Understanding**: explore the dataset with the analytical tools studied and write a concise “data understanding” report describing data semantics, assessing data quality, the distribution of the variables and the pairwise correlations
	* Data semantics
	* Distribution of the variables and statistics
	* Assessing data quality (missing values, outliers)
	* Variables transformations & generation
	* Pairwise correlations and eventual elimination of redundant variables
2. **Data Preparation**: improve the quality of your data and prepare it by extracting new features interesting for describing the customer profile and his purchasing behavior

### 2) Clustering analysis
Based on the customer’s profile explore the dataset using various clustering techniques.
Carefully describe your decisions for each algorithm and which are the advantages provided by the different approaches.
1. Clustering Analysis by K-means
	* Identification of the best value of k
	*  Characterization of the obtained clusters by using both analysis of the k centroids and comparison of the distribution of variables within the clusters and that in the whole dataset
	* Evaluation of the clustering results
2. Analysis by density-based clustering
	* Study of the clustering parameters
	* Characterization and interpretation of the obtained clusters
3. Analysis by hierarchical clustering
	* Compare different clustering results got by using different version of the algorithm
	* Show and discuss different dendrograms using different algorithms
4. Final evaluation of the best clustering approach and comparison of the clustering obtained
5. _(optional)_ Explore the opportunity to use alternative clustering techniques in the library [pyclustering](https://github.com/annoviko/pyclustering/)
	* Fuzzy C-Means
	* Optic
	* X-Means
	* Expectation Maximization
	* Genetic


### 3) Classification Analysis
Consider the problem of predicting for each customer a label that defines if (s)he is a **high-spending** customer, **medium-spending** customer or **low-spending** customer.
1. Define a customer profile that enables the above customer classification. Please, reason on the suitability of the customer profile, defined for the clustering analysis. In case this profile is not suitable for the above prediction problem you can also change the indicators
2. Compute the label for any customer. Note that, the class to be predicted must be nominal
3. Perform the predictive analysis comparing the performance of different models discussing the results and discussing the possible preprocessing that they applied to the data for managing possible problems identified that can make the prediction hard. Note that the evaluation should be performed on both training and test sets
	* Decision Tree
	* Random Forest
	* Naïve Bayes
	* k-Nearest Neighbors
	* Support Vector Machine
	* Artificial Neural Networks
	* Oversampling using SMOTE

### 4) Sequential Pattern Mining
1. Consider the problem of mining frequent sequential patterns. To address the task:
	* Model the customer as a sequence of baskets
	* Apply the Sequential Pattern Mining algorithm ([gsp](project/DM_25_TASK4/gsp.py) implementation)
	* Discuss the resulting patterns
2. _(optional)_ Handling _time constraint_ while building Sequential Patterns
