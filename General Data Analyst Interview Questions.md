## 1. Mention the differences between Data Mining and Data Profiling?


Data mining and data profiling are two distinct but complementary processes in data analysis. Data mining is used to discover patterns and trends from large datasets using techniques like clustering, classification, and regression. Its goal is to extract insights that can lead to better decision-making or predictive models.

On the other hand, data profiling is focused on evaluating and improving the quality of data. It involves checking for issues like missing values, duplicates, or inconsistencies, and it’s generally done before any in-depth analysis or modeling. While data mining helps uncover valuable insights, data profiling ensures the data is clean and reliable before analysis begins.

---
## 2. Define the term 'Data Wrangling in Data Analytics.

### **Definition of Data Wrangling in Data Analytics**

**Data Wrangling** (also known as **data munging**) is the process of cleaning, transforming, and organizing raw data into a structured and usable format for analysis. It involves preparing and shaping the data so that it is suitable for further processing, analysis, or visualization. This step is crucial for ensuring that the data is accurate, consistent, and formatted correctly for insights.

### **Key Steps in Data Wrangling**:
1. **Data Collection**: Gathering raw data from different sources (databases, APIs, spreadsheets, etc.).
2. **Cleaning**: Removing or fixing any inaccuracies, missing values, or outliers in the data.
3. **Transformation**: Converting the data into the appropriate format, which might include normalizing, scaling, or encoding.
4. **Integration**: Combining data from multiple sources into a single dataset.
5. **Enrichment**: Enhancing the data by adding additional attributes or data points.
6. **Normalization**: Ensuring the data follows a standard structure, such as consistent units or naming conventions.

### **Why Data Wrangling is Important**:
- **Improves Data Quality**: Ensures that data is clean, consistent, and ready for analysis.
- **Enables Better Analysis**: Raw data is often messy, and wrangling helps in preparing it for meaningful analysis or model building.
- **Time-Saving**: Proper data wrangling ensures that time isn’t wasted on irrelevant or flawed data during the actual analysis phase.

### **Tools Used in Data Wrangling**:
- **Python** (pandas, NumPy, regex)
- **R** (dplyr, tidyr)
- **SQL** (for querying and transforming data)
- **ETL tools** (Extract, Transform, Load)
- **Excel** (for basic cleaning tasks)

### **Example in Python (using pandas)**:
```python
import pandas as pd

# Load the dataset
data = pd.read_csv("raw_data.csv")

# Clean the data (e.g., remove missing values)
data = data.dropna()

# Transform the data (e.g., convert column to appropriate type)
data['Date'] = pd.to_datetime(data['Date'])

# Normalize data (e.g., scale numerical columns)
data['Price'] = (data['Price'] - data['Price'].mean()) / data['Price'].std()

# Preview the wrangled data
print(data.head())
```



### **How to Answer in an Interview**:

**Sample Response**:

**"Data wrangling is the process of cleaning, transforming, and organizing raw data into a usable format for analysis. It typically involves steps like cleaning the data by removing duplicates and handling missing values, transforming data into the required structure (such as converting data types or encoding categorical variables), and integrating data from different sources. The goal of data wrangling is to ensure that the data is accurate, consistent, and ready for analysis or modeling."**

---

# 3. 



