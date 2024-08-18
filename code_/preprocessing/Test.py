import pandas as pd

# Create a simple DataFrame
data = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}

df = pd.DataFrame(data)

# Export the DataFrame to a CSV file
df.to_csv('simple_dataset.csv', index=False)
print("CSV file has been created successfully.")
