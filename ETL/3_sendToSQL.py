import pandas as pd
from sqlalchemy import create_engine
import mysql.connector

host1 = "localhost"
port1 = 3306
user1 = 'root'
database1 = "ngsim"
password1 = "dB79*dG2024!" #use your password

connection = mysql.connector.connect(
            host = host1,
            port = port1,
            user = user1,
            database = database1,
            password = password1
)

cursor = connection.cursor()

csvFromSQL = "2_processedI80.csv" #enter the name of the csv from running the SQL query

df = pd.read_csv(csvFromSQL)

engine = create_engine(f'mysql+mysqlconnector://{user1}:{password1}@{host1}/{database1}')

df.to_sql(name='i_80_processed', con=engine, if_exists='replace', index=False, chunksize=1000)