import psycopg2
from psycopg2 import sql, OperationalError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def test_connection():
    try:
        print("Attempting to connect to PostgreSQL...")
        connection = psycopg2.connect(
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port"),
            dbname=os.getenv("dbname")
        )
        print("Connection established.")

        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("Query executed successfully. PostgreSQL version:", record)

        cursor.close()
        connection.close()
        print("Connection closed successfully.")

    except OperationalError as e:
        print("Error while connecting to PostgreSQL:", e)
        print("Connection failed.")

if __name__ == "__main__":
    test_connection()
