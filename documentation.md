# Detailed Documentation: Walmart Black Friday Sales Analysis

## 1. Introduction

### 1.1. Business Problem

Walmart's management team seeks to understand customer purchase behavior during Black Friday sales, specifically focusing on the purchase amount. They aim to analyze how spending differs based on customer demographics, particularly gender, marital status, and age. A key question is whether women spend more per transaction than men during this high-volume sales event. The insights derived will help Walmart make data-driven decisions regarding marketing, inventory management, and customer segmentation.

### 1.2. Objectives

*   Import and perform initial analysis of the Black Friday transaction dataset.
*   Clean the data, handle missing values (where relevant to the core questions), and convert data types.
*   Conduct Exploratory Data Analysis (EDA) using univariate and bivariate visualizations to understand data distributions and relationships between purchase amount and demographic factors.
*   Analyze average spending per transaction for different demographic groups (Gender, Marital Status, Age).
*   Apply the Central Limit Theorem (CLT) to calculate confidence intervals for the population mean spending within these demographic groups.
*   Determine if the confidence intervals for different groups (e.g., male vs. female) overlap, indicating whether observed differences in sample averages are statistically significant.
*   Derive actionable insights and provide recommendations to Walmart based on the findings.

## 2. Dataset

The analysis utilizes the `walmart_data.csv` dataset.

### 2.1. Features

*   **User_ID:** Unique identifier for the customer. (Type: `int64` initially, converted to `str`)
*   **Product_ID:** Unique identifier for the product. (Type: `object`, treated as `str`)
*   **Gender:** Sex of the customer (M/F). (Type: `object`, converted to `category`)
*   **Age:** Age range of the customer (e.g., '0-17', '26-35'). (Type: `object`, converted to `category`)
*   **Occupation:** Occupation code (masked). (Type: `int64`)
*   **City_Category:** Category of the city (A, B, C). (Type: `object`, converted to `category`)
*   **Stay_In_Current_City_Years:** Number of years the customer has stayed in the current city (e.g., '1', '4+'). (Type: `object`, converted to `category`)
*   **Marital_Status:** Marital status of the customer (0 = Single, 1 = Married). (Type: `int64`, converted to `category`)
*   **Product_Category:** Main category of the product (masked). (Type: `int64`)
*   **Purchase:** Purchase amount in dollars. (Type: `int64`)

*(Note: The dataset might contain `Product_Category_2` and `Product_Category_3` with many missing values. These were noted but not imputed or used in the primary analysis focused on demographic spending averages.)*

## 3. Methodology

The analysis was conducted using Python and the `walmart_analysis.py` script.

### 3.1. Data Loading and Initial Inspection

*   The dataset was loaded using `pandas.read_csv`.
*   Initial checks included `.shape`, `.info()`, `.describe(include='all')`, and `.isnull().sum()` to understand dimensions, data types, missing values, and basic statistics.

### 3.2. Data Cleaning and Preparation

*   **Missing Values:** Checked for missing values in the target variable `Purchase`. Confirmed no missing values. Noted significant missing values in `Product_Category_2` and `Product_Category_3` but did not impute them as they weren't central to the core questions about average spending by demographics.
*   **Data Type Conversion:** Converted `Gender`, `Age`, `City_Category`, `Stay_In_Current_City_Years`, and `Marital_Status` to the `category` dtype for efficiency and semantic correctness. Converted `User_ID` and `Product_ID` to `str` as they are identifiers.
*   **Age Binning:** Created a new categorical column `Age_Group` based on specified life stages ('0-17', '18-25', '26-35', '36-50', '51+') by mapping the original `Age` categories. Set an order for these categories for logical plotting.

### 3.3. Exploratory Data Analysis (EDA)

*   **Univariate Analysis:**
    *   Used `seaborn.histplot` to visualize the distribution of the `Purchase` amount.
    *   Used `seaborn.countplot` for categorical features (`Gender`, `Age_Group`, `City_Category`, `Marital_Status`, `Stay_In_Current_City_Years`) to understand frequency distributions.
*   **Bivariate Analysis:**
    *   Used `seaborn.boxplot` to compare the distribution of `Purchase` amounts across different categories of `Gender`, `Marital_Status`, `Age_Group`, and `City_Category`. This helps visualize differences in median spending and spread.
*   **Plot Saving:** All generated plots were saved as PNG files for review.

### 3.4. Statistical Analysis and Hypothesis Testing (Implicit)

*   **Average Spending Calculation:** Used `pandas.groupby()` and `.mean()` to calculate the average `Purchase` amount for each category within `Gender`, `Marital_Status`, and `Age_Group`.
*   **Central Limit Theorem (CLT) Application:** Leveraged the CLT, which states that the distribution of sample means will approximate a normal distribution for large sample sizes, regardless of the population distribution. This allows using the normal distribution to calculate confidence intervals for the population mean.
*   **Confidence Interval Calculation:**
    *   Defined a function `calculate_confidence_interval` using `scipy.stats.sem` (Standard Error of the Mean) and `scipy.stats.norm.interval`.
    *   Calculated 90%, 95%, and 99% confidence intervals for the mean `Purchase` amount for male and female customers separately.
    *   Calculated 95% confidence intervals for single vs. married customers and for each `Age_Group`.
*   **Overlap Analysis:** Compared the calculated confidence intervals (primarily at the 95% level) for different groups (e.g., male vs. female, single vs. married).
    *   If intervals **do not overlap**, it suggests a statistically significant difference between the population means of the groups at that confidence level.
    *   If intervals **do overlap**, we cannot conclude a statistically significant difference between the population means.
*   **Sample Size Effect:** Demonstrated how increasing the sample size (by resampling the male purchase data) leads to narrower (more precise) confidence intervals.

## 4. Key Findings

Based on the analysis, particularly the 95% confidence intervals:

1.  **Gender:** There is a **statistically significant difference** in average spending per transaction between males and females. Males spend significantly more on average during Black Friday sales (`$9437.53` vs. `$8734.57` in the sample, with non-overlapping 95% CIs: Males `(9422.02, 9453.03)`, Females `(8709.21, 8759.92)`).
2.  **Marital Status:** There is **no statistically significant difference** in average spending per transaction between single (0) and married (1) customers. The sample averages are very close (`$9265.91` vs. `$9261.17`), and their 95% confidence intervals overlap considerably (Single `(9248.62, 9283.20)`, Married `(9240.46, 9281.89)`).
3.  **Age:** Average spending varies across age groups.
    *   The `51+` age group shows the highest average spending (`$9463.66`), which is statistically significantly higher than some younger adult groups like `26-35` (`$9252.69`) based on non-overlapping 95% CIs.
    *   The `0-17` age group shows the lowest average spending (`$8933.46`).
    *   Many adjacent adult age groups (e.g., `18-25`, `26-35`, `36-50`) have overlapping confidence intervals, suggesting their average spending might not be significantly different from each other, although they generally spend more than the `0-17` group.
4.  **City Category:** Boxplots suggest potential differences, with City Category 'C' possibly having higher median spending and a wider range compared to 'A' and 'B'. This wasn't formally tested with CIs in this analysis but could be an area for further investigation.
5.  **Purchase Distribution:** The overall distribution of purchase amounts is right-skewed, indicating most transactions are for lower amounts, but there's a tail of high-value purchases.

## 5. Recommendations for Walmart

Based on the findings:

1.  **Primary Strategy - Broad Appeal:** Since marital status showed no significant difference in average spending, the primary Black Friday marketing strategy should focus on broad appeal, highlighting popular products and general deals attractive to all demographics.
2.  **Secondary Strategy - Gender Considerations:** Given that males have a statistically significantly higher average spend, consider secondary, targeted promotions or product highlights for male-associated categories, especially if inventory and margins allow. However, this shouldn't overshadow the primary broad appeal.
3.  **Target High-Value Segments:** Investigate beyond averages. Identify specific demographic combinations (age, occupation, city, gender) or customer segments (based on past behavior) that contribute disproportionately to total sales volume or purchase high-margin/high-price items. Analyze product category preferences within these segments.
4.  **Age Group Focus:** Continue strong marketing towards the core `18-50` age groups. Recognize the `51+` group as having the highest average spending and tailor specific offers or communications if appropriate. Address the lower spending `0-17` group based on strategic goals (e.g., specific product categories or long-term customer building).
5.  **Investigate City Differences:** Further analyze the spending patterns across City Categories ('A', 'B', 'C') suggested by the boxplots. If significant differences are confirmed, explore reasons (e.g., income, product mix, competition) and consider tailored local promotions or logistical adjustments.
6.  **Leverage Personalization:** Utilize `User_ID` and purchase history (`Product_ID`) for personalized recommendations and offers. Past behavior is often a stronger predictor of future purchases than broad demographics alone.
7.  **Monitor Trends:** Continuously track these spending patterns across different sales events and years to identify shifts in customer behavior.

## 6. Conclusion

The analysis successfully identified statistically significant differences in average Black Friday spending based on gender (males higher) and age (older groups generally higher), but not based on marital status. These insights provide a foundation for Walmart to refine its marketing and customer strategies for future Black Friday events, balancing broad appeal with targeted approaches towards demonstrably higher-spending segments.
