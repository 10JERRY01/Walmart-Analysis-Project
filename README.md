# Walmart Black Friday Sales Analysis

## Project Overview

This project analyzes customer purchase behavior during a Walmart Black Friday sales event using transactional data. The primary goal is to understand spending patterns based on demographics like gender, age, and marital status, specifically investigating if spending habits differ significantly between these groups. The analysis utilizes Python with libraries like Pandas, NumPy, Matplotlib, Seaborn, and SciPy to perform data cleaning, exploratory data analysis (EDA), and statistical inference using the Central Limit Theorem (CLT) and confidence intervals.

## Dataset

The analysis uses the `walmart_data.csv` dataset, which contains transactional data with the following features:

*   **User_ID:** Unique identifier for the customer.
*   **Product_ID:** Unique identifier for the product.
*   **Gender:** Sex of the customer (M/F).
*   **Age:** Age range of the customer.
*   **Occupation:** Occupation code (masked).
*   **City_Category:** Category of the city (A, B, C).
*   **Stay_In_Current_City_Years:** Number of years the customer has stayed in the current city.
*   **Marital_Status:** Marital status of the customer (0 = Single, 1 = Married).
*   **Product_Category:** Category of the product (masked).
*   **Purchase:** Purchase amount in dollars.

*(Note: The original dataset might contain additional product category columns (`Product_Category_2`, `Product_Category_3`) which have missing values and were not the primary focus of this specific analysis task regarding demographic spending averages.)*

## Analysis Script

The core analysis is performed by the `walmart_analysis.py` script. It executes the following steps:

1.  Loads the dataset (`walmart_data.csv`).
2.  Performs initial data inspection (shape, types, null values, summary statistics).
3.  Cleans and prepares the data (handles missing values in 'Purchase' if any, converts data types, creates age bins).
4.  Conducts Exploratory Data Analysis (EDA):
    *   Univariate analysis (distribution plots, count plots).
    *   Bivariate analysis (boxplots comparing Purchase amount across Gender, Marital Status, Age Group, City Category).
5.  Answers specific business questions using statistical analysis:
    *   Calculates average spending per transaction by gender, marital status, and age group.
    *   Computes confidence intervals (90%, 95%, 99%) for the average spending of different demographic groups using the Central Limit Theorem.
    *   Checks for overlaps in confidence intervals to determine statistically significant differences in spending.
6.  Prints key insights and actionable recommendations for Walmart based on the analysis.
7.  Saves generated plots as PNG files in the project directory.

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scipy
    ```
3.  Place the `walmart_data.csv` file in the same directory as the `walmart_analysis.py` script.
4.  Run the script from your terminal:
    ```bash
    python walmart_analysis.py
    ```

## Outputs

*   **Console Output:** Detailed analysis results, including statistical summaries, average spending figures, confidence intervals, overlap analysis, insights, and recommendations.
*   **Plot Files (PNG):** Visualizations saved in the project directory, such as:
    *   `purchase_distribution.png`
    *   `Gender_countplot.png`
    *   `Age_Group_countplot.png`
    *   `City_Category_countplot.png`
    *   `Marital_Status_countplot.png`
    *   `Stay_In_Current_City_Years_countplot.png`
    *   `purchase_vs_gender_boxplot.png`
    *   `purchase_vs_marital_status_boxplot.png`
    *   `purchase_vs_age_group_boxplot.png`
    *   `purchase_vs_city_category_boxplot.png`

## Further Documentation

For a more detailed explanation of the project, analysis steps, findings, and recommendations, please refer to `documentation.md`.
