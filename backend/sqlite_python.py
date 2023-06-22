import sqlite3
import os
import csv

class SQL:
    
    # create a connection to the database
    def getConnection(self):
        return sqlite3.connect('onex.db')
    
    # A cursor is an object used to execute SQL commands
    def getCursors(self):
        conn = self.getConnection()
        return conn.cursor()
    
    def commit(self, conn):
        conn.commit()
        conn.close()

    def initialize_sentiments_table(self):
        sql = SQL()
        sql.getConnection()
        cursor = sql.getCursors()
        cursor.execute('''CREATE TABLE IF NOT EXISTS sentiments
                        (id INT PRIMARY KEY NOT NULL,
                        file_name TEXT NOT NULL,
                        sentiment TEXT NOT NULL)'''),

        # test data
        data = [(1, "onex_q1_2021_interim_report.pdf", "NEGATIVE"),
                (2, "onex_q1_2022_intrerim_report.pdf", "NEGATIVE"),
                (3, "onex_q2_2022_intrerim_report.pdf", "POSITIVE")]

        cursor.executemany("INSERT INTO sentiments (id, file_name, sentiment) VALUES (?,?,?)", data)

        cursor.execute("SELECT * FROM sentiments")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

    def initialize_stock_table(self):
        sql = SQL()
        cursor = sql.getCursors()
        cursor.execute('''CREATE TABLE IF NOT EXISTS stock
                        (id INT PRIMARY KEY,
                        date DATE NOT NULL,
                        price REAL NOT NULL)'''),

        # Read the CSV file containing the stock data from yahoo finance for ONEX
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        data_file = os.path.join(parent_directory, 'data/uploads', 'ONEX_stock_Price_close.csv')
        with open(data_file, 'r') as file:
            data = csv.reader(file)
            next(data)  # Skip header row if it exists
            id = 1
            for row in data:
                row = [id] + row
                cursor.execute("INSERT INTO stock (id, date, price) VALUES (?, ?, ?)", row)
                id += 1

        # # test : visualize some of the data
        # cursor.execute("SELECT * FROM stock LIMIT 10")
        # rows = cursor.fetchall()
        # for row in rows:
        #     print(row)

sql = SQL()
# sql.initialize_sentiments_table()
# sql.initialize_stock_table()