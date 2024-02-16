# Databricks notebook source
# Unmount the storage
dbutils.fs.unmount("/mnt/IkigaiGuide")

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://ikigai-guide-data@ikigaiguideunidata.blob.core.windows.net",
  mount_point = "/mnt/IkigaiGuide",
  extra_configs = {"fs.azure.account.key.ikigaiguideunidata.blob.core.windows.net": "+tu0bU1lIVTvT5xpfonfXJ6gJqH4a8pU8/MOTtGLD0l5Tkh/Qf8oGcyOvVBWCsoRzjiQ9cN6vOzb+AStfhQGmw=="})


# COMMAND ----------

# List files in the mounted directory to verify the mount
display(dbutils.fs.ls("/mnt/IkigaiGuide"))


# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/TIMES.csv"

# Load the dataset
TIMES = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
TIMES.show()


# COMMAND ----------

# Display the data types of each column in the TIMES DataFrame
TIMES.dtypes


# COMMAND ----------

from pyspark.sql.functions import col, when, count

# Count missing values for each column
missing_value_counts = TIMES.select([
    count(
        when(
            col(c).isNull() | (col(c) == ""), c
        )
    ).alias(c) for c in TIMES.columns
])

missing_value_counts.show()


# COMMAND ----------

from pyspark.sql.functions import col, lit
from pyspark.sql.window import Window
from pyspark.sql.functions import rank

# Step 1: Calculate the mode
mode_df = TIMES.groupBy("stats_female_male_ratio").count()\
               .withColumn("rank", rank().over(Window.orderBy(col("count").desc())))\
               .filter(col("rank") == 1).drop("rank")

mode_value = mode_df.collect()[0]["stats_female_male_ratio"]

# Step 2: Fill missing values with the mode
TIMES_filled = TIMES.na.fill({"stats_female_male_ratio": mode_value})

# Verify the operation
TIMES_filled.select("stats_female_male_ratio").show()


# COMMAND ----------

# Filter the DataFrame to only include rows where 'subjects_offered' is null
missing_subjects_offered = TIMES.filter(TIMES["subjects_offered"].isNull())

# Show the rows with missing 'subjects_offered' values
missing_subjects_offered.show()


# COMMAND ----------

# Filter out rows where 'subjects_offered' is null
TIMES_filtered = TIMES.filter(TIMES["subjects_offered"].isNotNull())

# Show the count of rows before and after to verify rows are excluded
print("Original count:", TIMES.count())
print("Filtered count:", TIMES_filtered.count())

# You can continue working with TIMES_filtered as your DataFrame without the missing 'subjects_offered' records.


# COMMAND ----------

# Specify the path within the mounted storage where you want to save the file
output_path = "/mnt/IkigaiGuide/ikigai-transformed-data/TIMES_filtered"

# Save the filtered DataFrame to CSV in the mounted Azure Blob Storage
TIMES_filtered.write.mode("overwrite").option("header", "true").csv(output_path)


# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/cwurdata.csv"

# Load the dataset
cwur = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
cwur.show()


# COMMAND ----------

cwur.dtypes

# COMMAND ----------

from pyspark.sql.functions import col, sum as _sum

# Count missing values for each column
missing_value_counts = cwur.select([_sum(col(c).isNull().cast("int")).alias(c) for c in cwur.columns])

missing_value_counts.show()


# COMMAND ----------

total_rows = cwur.count()
print(f"Total rows: {total_rows}")


# COMMAND ----------

# Calculate descriptive statistics for 'broad_impact'
cwur.describe("broad_impact").show()


# COMMAND ----------

# Calculate the median - Spark doesn't have a built-in median function, but you can approximate it using approxQuantile
median_broad_impact = cwur.approxQuantile("broad_impact", [0.5], 0.01)
print(f"Median of broad_impact: {median_broad_impact[0]}")

# COMMAND ----------

# Generating histogram data for 'broad_impact'
histogram_data = cwur.select("broad_impact").rdd.flatMap(lambda x: x).histogram(10)

# The histogram data would contain two lists: bin edges and counts
bin_edges = histogram_data[0]
counts = histogram_data[1]

# You can print these out to plot them externally
print("Bin edges: ", bin_edges)
print("Counts: ", counts)


# COMMAND ----------

# Calculating the lower and upper quartiles
Q1, Q3 = cwur.approxQuantile("broad_impact", [0.25, 0.75], 0.01)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound for outliers: {lower_bound}")
print(f"Upper bound for outliers: {upper_bound}")


# COMMAND ----------

unique_broad_impact_count = cwur.select("broad_impact").distinct().count()
print(f"Number of unique values in 'broad_impact': {unique_broad_impact_count}")


# COMMAND ----------

# Displaying unique values (or a sample if there are too many)
cwur.select("broad_impact").distinct().show(n=20, truncate=False)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# Assuming the 'cwur' DataFrame is already loaded and 'broad_impact' is the target column
# Select relevant feature columns for the prediction model
feature_columns = ['quality_of_education', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations', 'patents', 'score', 'year']  # Example feature columns

# Prepare the data: filter out rows with and without 'broad_impact' values
data_with_broad_impact = cwur.filter(cwur.broad_impact.isNotNull())
data_without_broad_impact = cwur.filter(cwur.broad_impact.isNull())

# Create a features vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Add the assembler to the stages of your pipeline
pipeline_stages = [assembler]


# COMMAND ----------

# Add the linear regression model to the pipeline
lr = LinearRegression(featuresCol="features", labelCol="broad_impact")
pipeline_stages.append(lr)

# Create the pipeline
pipeline = Pipeline(stages=pipeline_stages)

# Train the model using data with 'broad_impact' values
model = pipeline.fit(data_with_broad_impact)


# COMMAND ----------

# Use the model to predict 'broad_impact' for data where it is missing
predictions = model.transform(data_without_broad_impact)


# COMMAND ----------

from pyspark.sql.functions import col

# Select the original columns, replace 'broad_impact' with the predicted values
predictions = predictions.select(col("prediction").alias("broad_impact"), *[c for c in cwur.columns if c != "broad_impact"])

# Union the datasets back together
updated_cwur = data_with_broad_impact.unionByName(predictions)

# Show some of the updated dataset to verify
updated_cwur.show()


# COMMAND ----------

# Define the path in DBFS where you want to save the updated dataset
dbfs_path = "/mnt/IkigaiGuide/ikigai-transformed-data/updated_cwur"

# Save the updated DataFrame as a CSV file
updated_cwur.write.mode("overwrite").option("header", "true").csv(dbfs_path)


# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/education_expenditure.csv"

# Load the dataset
expenditure = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
expenditure.show()

# COMMAND ----------

expenditure.dtypes

# COMMAND ----------

from pyspark.sql.functions import col, sum as sum_col

# Count missing values for each column
missing_values = expenditure.select([sum_col(col(c).isNull().cast("int")).alias(c) for c in expenditure.columns])

missing_values.show()


# COMMAND ----------

total_rows = expenditure.count()
print(f"Total number of rows: {total_rows}")


# COMMAND ----------

from pyspark.sql.functions import col, count, when, isnan

# Rows with Exactly One Missing Value
rows_with_one_missing = expenditure.where(sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0) for c in expenditure.columns) == 1)


# COMMAND ----------

rows_with_one_missing.show()

# COMMAND ----------


# Rows with All Values Filled (No Missing Values)
rows_with_all_filled = expenditure.dropna(how='any')  # This removes rows with any null values
rows_with_all_filled.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Assuming 'rows_with_one_missing' is your DataFrame
# For year 1995
missing_1995 = rows_with_one_missing.filter(col("1995").isNull())
missing_1995.show()
# For year 2000
missing_2000 = rows_with_one_missing.filter(col("2000").isNull())
missing_2000.show()
# For year 2005
missing_2005 = rows_with_one_missing.filter(col("2005").isNull())
missing_2005.show()
# For year 2009
missing_2009 = rows_with_one_missing.filter(col("2009").isNull())
missing_2009.show()
# For year 2010
missing_2010 = rows_with_one_missing.filter(col("2010").isNull())
missing_2010.show()
# For year 2011
missing_2011 = rows_with_one_missing.filter(col("2011").isNull())
missing_2011.show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Assuming `rows_with_all_filled` is your Spark DataFrame
feature_columns = ['2000', '2005', '2009', '2010', '2011']  # List of feature columns
target_column = '1995'  # Target column

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")


# COMMAND ----------

# Split the data
train_data, test_data = rows_with_all_filled.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

# Initialize the RandomForest model
rf = RandomForestRegressor(featuresCol="features", labelCol=target_column)


# COMMAND ----------

# Define the pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Train the model
model = pipeline.fit(train_data)


# COMMAND ----------

# Make predictions
predictions = model.transform(test_data)

# Select example rows to display
predictions.select("prediction", target_column, "features").show(5)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# COMMAND ----------

# Note: Ensure `missing_1995` has the same feature structure as your training data
# This might involve selecting the same feature columns and assembling them

# Predict missing values for 1995
missing_predictions_1995 = model.transform(missing_1995)

# Show predicted values
missing_predictions_1995.select("prediction", "features").show()


# COMMAND ----------

missing_predictions_1995.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Select the 'country' and 'prediction' columns, renaming 'prediction' to 'predicted_1995'
predicted_1995_df = missing_predictions_1995.select(
    col("country"),
    col("prediction").alias("predicted_1995")
)

predicted_1995_df.show()


# COMMAND ----------

# Adjust feature columns for predicting 2000, exclude 2000 from features
feature_columns_2000 = ['1995', '2005', '2009', '2010', '2011']
target_column_2000 = '2000'

# Assemble features for 2000
assembler_2000 = VectorAssembler(inputCols=feature_columns_2000, outputCol="features")


# COMMAND ----------

# Define the pipeline for 2000
pipeline_2000 = Pipeline(stages=[assembler_2000, RandomForestRegressor(featuresCol="features", labelCol=target_column_2000)])

# Train the model for 2000
model_2000 = pipeline_2000.fit(rows_with_all_filled)


# COMMAND ----------

# Predict missing values for 2000
missing_predictions_2000 = model_2000.transform(missing_2000)

# Show predicted values for 2000
missing_predictions_2000.select("prediction", "features").show()


# COMMAND ----------

missing_predictions_2000.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Select the 'country' and 'prediction' columns, renaming 'prediction' to 'predicted_2000'
predicted_2000_df = missing_predictions_2000.select(
    col("country"),
    col("prediction").alias("predicted_2000")
)

predicted_2000_df.show()


# COMMAND ----------

# Adjust feature columns for predicting 2010, exclude 2010 from features
feature_columns_2010 = ['1995', '2000', '2005', '2009', '2011']
target_column_2010 = '2010'

# Assemble features for 2010
assembler_2010 = VectorAssembler(inputCols=feature_columns_2010, outputCol="features")

# Define the pipeline for 2010
pipeline_2010 = Pipeline(stages=[assembler_2010, RandomForestRegressor(featuresCol="features", labelCol=target_column_2010)])

# Train the model for 2010
model_2010 = pipeline_2010.fit(rows_with_all_filled)

# Predict missing values for 2010
missing_predictions_2010 = model_2010.transform(missing_2010)

# Show predicted values for 2010
missing_predictions_2010.select("prediction", "features").show()


# COMMAND ----------

missing_predictions_2010.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Select the 'country' and 'prediction' columns, renaming 'prediction' to 'predicted_2010'
predicted_2010_df = missing_predictions_2010.select(
    col("country"),
    col("prediction").alias("predicted_2010")
)

predicted_2010_df.show()


# COMMAND ----------

# Adjust feature columns for predicting 2011, exclude 2011 from features
feature_columns_2011 = ['1995', '2000', '2005', '2009', '2010']
target_column_2011 = '2011'

# Assemble features for 2011
assembler_2011 = VectorAssembler(inputCols=feature_columns_2011, outputCol="features")

# Define the pipeline for 2011
pipeline_2011 = Pipeline(stages=[assembler_2011, RandomForestRegressor(featuresCol="features", labelCol=target_column_2011)])

# Train the model for 2011
model_2011 = pipeline_2011.fit(rows_with_all_filled)

# Predict missing values for 2011
missing_predictions_2011 = model_2011.transform(missing_2011)

# Show predicted values for 2011
missing_predictions_2011.select("prediction", "features").show()


# COMMAND ----------

missing_predictions_2011.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Select the 'country' and 'prediction' columns, renaming 'prediction' to 'predicted_2011'
predicted_2011_df = missing_predictions_2011.select(
    col("country"),
    col("prediction").alias("predicted_2011")
)

predicted_2011_df.show()


# COMMAND ----------

from pyspark.sql.functions import coalesce

# Assuming expenditure is your original DataFrame
# And that predicted_1995_df, predicted_2000_df, predicted_2010_df, predicted_2011_df
# have been correctly defined as per previous steps

# Join the original dataset with each of the predicted DataFrames on 'country'
updated_expenditure = expenditure \
    .join(predicted_1995_df, on="country", how="left") \
    .join(predicted_2000_df, on="country", how="left") \
    .join(predicted_2010_df, on="country", how="left") \
    .join(predicted_2011_df, on="country", how="left")

# Update the original year columns with the predictions where original values are null
updated_expenditure = updated_expenditure \
    .withColumn("1995", coalesce(col("1995"), col("predicted_1995"))) \
    .withColumn("2000", coalesce(col("2000"), col("predicted_2000"))) \
    .withColumn("2010", coalesce(col("2010"), col("predicted_2010"))) \
    .withColumn("2011", coalesce(col("2011"), col("predicted_2011")))

# Drop the temporary prediction columns as they're no longer needed
updated_expenditure = updated_expenditure.drop("predicted_1995", "predicted_2000", "predicted_2010", "predicted_2011")

# Display the updated DataFrame to verify the changes
updated_expenditure.show()


# COMMAND ----------

# Define the path in DBFS where you want to save the updated dataset
dbfs_path = "/mnt/IkigaiGuide/ikigai-transformed-data/updated_expenditure"

# Save the updated DataFrame as a CSV file
updated_expenditure.write.mode("overwrite").option("header", "true").csv(dbfs_path)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/educational_attainment.csv"

# Load the dataset
attainment = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
attainment.show()

# COMMAND ----------

attainment.dtypes

# COMMAND ----------


attainment.describe().show()


# COMMAND ----------

from pyspark.sql.functions import col, sum

# Assuming 'attainment' is your DataFrame
# Count the missing values in each column
missing_values = attainment.select([sum(col(column).isNull().cast("int")).alias(column) for column in attainment.columns])

# Display the result
missing_values.show()


# COMMAND ----------

from pyspark.sql.functions import col

# Select the columns for which you want to analyze the data distribution
selected_columns = ['1985', '1986', '1987', '1990', '1991', '1992', '1993', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2015']

# Calculate the statistical measures for each selected column
summary = attainment.select(*[col(column).cast("double") for column in selected_columns]).summary()

# Display the result
summary.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Select the columns for visualization
selected_columns = ['1985', '1986', '1987', '1990', '1991', '1992', '1993', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2015']

# Iterate through each selected column and create a histogram
for column in selected_columns:
    # Convert the column to a Pandas series for plotting
    data = attainment.select(column).dropna().toPandas()[column]
    
    # Create a histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Select the columns for visualization
selected_columns = ['1985', '1990', '1995']

# Iterate through each selected column and create a box plot
for column in selected_columns:
    # Convert the column to a Pandas series for plotting
    data = attainment.select(column).dropna().toPandas()[column]
    
    # Create a box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, vert=False)
    plt.xlabel(column)
    plt.title(f'Boxplot of {column}')
    plt.show()

# COMMAND ----------

# Define the path to save the DataFrame within the mounted storage
output_path = "/mnt/IkigaiGuide/ikigai-transformed-data/attainment"

# Write the updated DataFrame as a CSV file to the mounted Blob Storage
attainment.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/school_and_country.csv"

# Load the dataset
school_and_country = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
school_and_country.show()

# COMMAND ----------

school_and_country.describe().show()

# COMMAND ----------

school_and_country.dtypes

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Count the missing values for each column
missing_values = school_and_country.select([sum(col(column).isNull().cast("int")).alias(column) for column in school_and_country.columns])

# Show the missing values count
missing_values.show()

# COMMAND ----------

# Define the path to save the DataFrame within the mounted storage
output_path = "/mnt/IkigaiGuide/ikigai-transformed-data/school_and_country"

# Write the updated DataFrame as a CSV file to the mounted Blob Storage
school_and_country.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/shanghaiData.csv"

# Load the dataset
shanghaiData = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
shanghaiData.show()

# COMMAND ----------

shanghaiData.describe().show()

# COMMAND ----------

shanghaiData.dtypes

# COMMAND ----------

from pyspark.sql.functions import col, sum

missing_values = shanghaiData.select([sum(col(c).isNull().cast("int")).alias(c) for c in shanghaiData.columns])
missing_values.show()

# COMMAND ----------

# Count the total number of rows
total_rows = shanghaiData.count()

# Get the number of columns
total_columns = len(shanghaiData.columns)

# Print the results
print("Total number of rows:", total_rows)
print("Total number of columns:", total_columns)

# COMMAND ----------

missing_university_rows = shanghaiData.filter(col("university_name").isNull())
missing_university_rows.show(truncate=False)

# COMMAND ----------

clean_data = shanghaiData.filter(~col("university_name").isNull())

# COMMAND ----------

missing_values = clean_data.select([sum(col(c).isNull().cast("int")).alias(c) for c in clean_data.columns])
missing_values.show()

# COMMAND ----------

missing_award_rows = shanghaiData.filter(col("award").isNull())
missing_award_rows.show(truncate=False)

# COMMAND ----------

shanghaiData = shanghaiData.filter((col("award").isNotNull()) & (col("university_name").isNotNull()))

# COMMAND ----------

shanghaiData.show(truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col, sum

missing_values = shanghaiData.select([sum(col(c).isNull().cast("int")).alias(c) for c in shanghaiData.columns])
missing_values.show()

# COMMAND ----------

from pyspark.sql.functions import mean

# Calculate the mean of the "ns" column
mean_value = shanghaiData.select(mean(col("ns"))).collect()[0][0]

# Fill the missing values in the "ns" column with the mean value
shanghaiData = shanghaiData.fillna(mean_value, subset=["ns"])

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Remove rows with missing values in "award" and "university_name" columns
shanghaiData = shanghaiData.filter((col("award").isNotNull()) & (col("university_name").isNotNull()))

# Count missing values in each column
missing_values = shanghaiData.select([sum(col(c).isNull().cast("int")).alias(c) for c in shanghaiData.columns])
missing_values.show()

# COMMAND ----------

# Count the total number of rows
total_rows = shanghaiData.count()

# Get the number of columns
total_columns = len(shanghaiData.columns)

# Print the results
print("Total number of rows:", total_rows)
print("Total number of columns:", total_columns)

# COMMAND ----------

# Define the path to save the DataFrame within the mounted storage
output_path = "/mnt/IkigaiGuide/ikigai-transformed-data/shanghaiData"

# Write the updated DataFrame as a CSV file to the mounted Blob Storage
shanghaiData.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# COMMAND ----------

# Adjusted file path to point to the specific file inside the ikigai-raw-data folder
file_path = "/mnt/IkigaiGuide/ikigai-raw-data/times2011.csv"

# Load the dataset
times2011 = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
times2011.show()

# COMMAND ----------

times2011.describe().show()

# COMMAND ----------

times2011.dtypes


# COMMAND ----------

# Count the number of rows in the updated_times2011 DataFrame
row_count = times2011.count()
row_count

# COMMAND ----------

from pyspark.sql.functions import col, sum

missing_values = times2011.select([sum(col(c).isNull().cast("int")).alias(c) for c in times2011.columns])
missing_values.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Check the distinct values in the "female_male_ratio" column
distinct_ratios = times2011.select(col("female_male_ratio")).distinct()

# Count the occurrences of each distinct value
value_counts = distinct_ratios.groupBy("female_male_ratio").count().orderBy("count", ascending=False)

# Display the distinct values and their counts
value_counts.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Calculate the mode of the "female_male_ratio" column
mode_values = times2011.groupBy("female_male_ratio").count().orderBy(col("count").desc())
mode_value = mode_values.first()["female_male_ratio"]

# If mode_value is None, it means there are no unique mode values
# In this case, we'll replace missing values with "Not Available" string
if mode_value is None:
    mode_value = "Not Available"

# Fill the missing values in the "female_male_ratio" column with the mode value
updated_times2011 = times2011.fillna(mode_value, subset=["female_male_ratio"])

# COMMAND ----------

# Count the number of rows in the updated_times2011 DataFrame
row_count = updated_times2011.count()
row_count

# COMMAND ----------

# Define the path to save the DataFrame within the mounted storage
output_path = "/mnt/IkigaiGuide/ikigai-transformed-data/updated_times2011"

# Write the updated DataFrame as a CSV file to the mounted Blob Storage
updated_times2011.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# COMMAND ----------


