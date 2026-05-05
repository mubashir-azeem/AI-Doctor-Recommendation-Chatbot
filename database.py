# creating database:

import sqlite3
import pandas as pd

conn = sqlite3.connect('doctors.db')
cursor = conn.cursor()

# creating table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS doctors (
    name TEXT,
    specialization TEXT,
    hospital TEXT,
    city TEXT,
    availability TEXT
)
''')

# deleting old data so duplicates dont come if we run again
cursor.execute("DELETE FROM doctors")

df = pd.read_csv('doctors.csv')

# inserting all rows from csv into database
for _, row in df.iterrows():
    cursor.execute("INSERT INTO doctors VALUES (?, ?, ?, ?, ?)", tuple(row))

conn.commit()
conn.close()

print("Database created successfully")