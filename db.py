#THIS FILE CREATES A NEW COPY OF THE DATABASE AND POPULATES IT WITH SAMPLE DATA
#DO NOT RUN THIS IF YOU DON'T WANT TO LOSE ANY CHANGES


from abc import get_cache_token
import sqlite3
import itertools as it
from typing import get_args
con = sqlite3.connect('chat.db')
cur = con.cursor()

def dropAllTables():
    cur.execute( '''DROP TABLE IF EXISTS facility''')
    cur.execute( '''DROP TABLE IF EXISTS customer''')
    cur.execute( '''DROP TABLE IF EXISTS storage''')

def standUpDB():
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    dropAllTables()

    cur.execute('''CREATE TABLE facility
                    (facility_pk int PRIMARY KEY, facility_name varchar, facility_address varchar, facility_city varchar, facility_state varchar, facility_country varchar, facility_phone varchar, facility_email varchar)''')
    cur.execute('''CREATE TABLE customer
                    (customer_pk int PRIMARY KEY, customer_name varchar, customer_address varchar, customer_city varchar, customer_state varchar, customer_country varchar, customer_phone varchar, customer_email varchar)''')
    cur.execute('''CREATE TABLE storage
                    (storage_pk int PRIMARY KEY, unit_number int, storage_type char, storage_size varchar, storage_price int, facility_id int, customer_id int, FOREIGN KEY (facility_id) REFERENCES facility(facility_name), FOREIGN KEY (customer_id) REFERENCES customer(customer_id))''')
    insertFacData()
    insertCustData()
    insertStorageData()
    con.commit()
    con.close()

def insertFacData():
    cur.execute('''INSERT INTO facility 
        (facility_name, facility_address, facility_city, facility_state, facility_country, facility_phone, facility_email)
        VALUES ('catonsville', 'cville_address', 'Baltimore', 'Maryland', 'USA', 'cville_phone', 'cville_email')''')
def insertCustData():
    cur.execute('''INSERT INTO customer 
        (customer_name, customer_address, customer_city, customer_state, customer_country, customer_phone, customer_email)
        VALUES ('testuser', 'address', 'city', 'state', 'country', 'phone', 'email')''')
def insertStorageData():
    for i in it.chain(range(100,115), range(200,211), range(300,311), range(400,411), range(500,511), range(600,615)):
        data_tuple = (i, 'A', 'SMALL', 100, 'catonsville', 'null')
        sqlite_insert_params = '''INSERT INTO storage 
            (unit_number, storage_type, storage_size, storage_price, facility_id, customer_id)
            VALUES (?,?,?,?,?,?)'''
        cur.execute(sqlite_insert_params, data_tuple)

def getUnitAvailability(location, unitSize):
    #returns number of open units
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT COUNT(*) FROM storage WHERE storage_size = ? AND facility_id = ?'''
    params = location, unitSize
    selectData = cur.execute(select_statement, params)
    try:
        return selectData.fetchone()[0]
    finally:
        con.close()

def getLocationPhone(location):
    #return phone number of location
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT facility_phone FROM facility WHERE ? = facility_name'''
    selectData = cur.execute(select_statement, [location])
    try:
        return selectData.fetchone()[0]
    finally:
        con.close()

def getLocationAddress(location):
    #return address of location
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT facility_address, facility_city, facility_state, facility_country FROM facility WHERE ? = facility_name'''
    selectData = cur.execute(select_statement, [location])
    try:
        return selectData.fetchone()
    finally:
        con.close()


#print table contents - for debugging
#cur.execute('''SELECT * FROM storage''')
#rows = cur.fetchall()
#
#for row in rows:
#        print(row)

#print(getLocationAddress('catonsville'))
#print(getLocationPhone('catonsville'))
#print(getUnitAvailability('SMALL', 'catonsville'))