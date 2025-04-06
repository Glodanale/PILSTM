import mysql.connector
import pandas as pd

with open('largeLoss.txt', 'r') as file:
    content = file.read()
    group_ids = content.split(',')

group_ids = [id.strip() for id in group_ids]

host1 = "localhost"
port1 = 3306
user1 = 'root'
database1 = "ngsim" 
password1 = "dB79*dG2024!" #fill in your password

connection = mysql.connector.connect(
            host = host1,
            port = port1,
            user = user1,
            database = database1,
            password = password1
)

cursor = connection.cursor()

group_ids_str = ','.join(f"'{id}'" for id in group_ids)

query = f"SELECT * FROM i_80_processed WHERE Group_ID NOT IN ({group_ids_str})"

cursor.execute(query)

results = cursor.fetchall()

print("Received from MySQL")

columns = [col[0] for col in cursor.description]

df = pd.DataFrame(results, columns=columns)

distinct_groups = df["Group_ID"].nunique()

print(distinct_groups)

df.to_csv("i_80_full_95.csv")

cursor.close()
connection.close()

print("Complete")