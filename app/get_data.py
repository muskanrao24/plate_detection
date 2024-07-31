import sqlite3
import csv

# Connect to your SQLite database
conn = sqlite3.connect('plates.db')
cursor = conn.cursor()

# Define your query to select all data from the table
cursor.execute('SELECT * FROM plates')

# Fetch all rows
rows = cursor.fetchall()

# Get column names
column_names = [description[0] for description in cursor.description]

# Write the data to a CSV file
with open('out.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the column names as the first row
    csvwriter.writerow(column_names)

    # Write the data
    csvwriter.writerows(rows)

# Close the cursor and connection
cursor.close()
conn.close()

print('Data exported to your_table.csv successfully.')
