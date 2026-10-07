import pandas as pd
from sqlalchemy import text

# Import the SQLAlchemy engine that we created earlier
# in src/database.py.
from database import engine


# ---------------------------------------------------------
# STEP 1: Test the MySQL connection
# ---------------------------------------------------------
# "with engine.connect()" opens a connection to MySQL.
# SELECT 1 is a very simple query used to verify that
# Python can communicate with the MySQL database.
with engine.connect() as connection:
    result = connection.execute(text("SELECT 1"))

    print("MySQL connection successful!")
    print("Connection test result:", result.scalar())


# ---------------------------------------------------------
# STEP 2: Check how many records are in loan_data
# ---------------------------------------------------------
# This query counts the total number of rows imported
# from train.csv into the MySQL loan_data table.
count_query = """
SELECT COUNT(*) AS total_rows
FROM loan_data
"""

# pd.read_sql() executes the SQL query and returns
# the result as a Pandas DataFrame.
count_df = pd.read_sql(count_query, engine)

print("\nTotal rows in loan_data:")
print(count_df)


# ---------------------------------------------------------
# STEP 3: Load the complete MySQL table into Pandas
# ---------------------------------------------------------
# This query retrieves all columns and rows from loan_data.
query = """
SELECT *
FROM loan_data
"""

# SQLAlchemy connects to MySQL and Pandas converts
# the SQL result into a DataFrame.
df = pd.read_sql(query, engine)


# ---------------------------------------------------------
# STEP 4: Display basic information
# ---------------------------------------------------------

# Display number of rows and columns.
print("\nDataset shape:")
print(df.shape)


# Display the first 5 records.
print("\nFirst 5 records:")
print(df.head())


# Display column names.
print("\nColumn names:")
print(df.columns.tolist())


# Display data types and non-null counts.
print("\nDataset information:")
df.info()