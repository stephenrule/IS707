#standUpDB CREATES A NEW COPY OF THE DATABASE AND POPULATES IT WITH SAMPLE DATA
#DO NOT RUN THIS IF YOU DON'T WANT TO LOSE ANY CHANGES


from abc import get_cache_token
from os import getlogin
import sqlite3
import itertools as it
from typing import get_args



def standUpDB():
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    dropAllTables(cur)

    cur.execute('''CREATE TABLE facility
                    (facility_pk int PRIMARY KEY, facility_name varchar, facility_address varchar, facility_city varchar, facility_state varchar, facility_country varchar, facility_phone varchar, facility_email varchar)''')
    cur.execute('''CREATE TABLE customer
                    (customer_pk int PRIMARY KEY, customer_name varchar, customer_address varchar, customer_city varchar, customer_state varchar, customer_country varchar, customer_phone varchar, customer_id varchar)''')
    cur.execute('''CREATE TABLE storage
                    (storage_pk int PRIMARY KEY, unit_number int, storage_type char, storage_size varchar, storage_price int, facility_id varchar, customer_id varchar, FOREIGN KEY (facility_id) REFERENCES facility(facility_name), FOREIGN KEY (customer_id) REFERENCES customer(customer_id))''')
    cur.execute('''CREATE TABLE response
                    (respone_pk int PRIMARY KEY, user_question varchar, bot_response varchar)''')
    insertFacData(cur)
    insertCustData(cur)
    insertStorageData(cur)
    insertResponseData(cur)
    con.commit()
    con.close()

def dropAllTables(cur):
    cur.execute( '''DROP TABLE IF EXISTS facility''')
    cur.execute( '''DROP TABLE IF EXISTS customer''')
    cur.execute( '''DROP TABLE IF EXISTS storage''')
    cur.execute( '''DROP TABLE IF EXISTS response''')
def insertFacData(cur):
    cur.execute('''INSERT INTO facility 
        (facility_name, facility_address, facility_city, facility_state, facility_country, facility_phone, facility_email)
        VALUES ('catonsville', 'cville_address', 'Baltimore', 'Maryland', 'USA', 'cville_phone', 'cville_email')''')
def insertCustData(cur):
    cur.execute('''INSERT INTO customer 
        (customer_name, customer_address, customer_city, customer_state, customer_country, customer_phone, customer_id)
        VALUES ('testuser', 'address', 'city', 'state', 'country', 'phone', 'UID')''')
def insertStorageData(cur):
    for i in it.chain(range(200,211), range(300,311), range(400,411), range(500,511)):
        data_tuple = (i, 'A', 'MEDIUM', 100, 'catonsville')
        sqlite_insert_params = '''INSERT INTO storage 
            (unit_number, storage_type, storage_size, storage_price, facility_id)
            VALUES (?,?,?,?,?)'''
        cur.execute(sqlite_insert_params, data_tuple)
    for i in it.chain(range(100,115), range(600,615)):
        data_tuple = (i, 'A', 'LARGE', 150, 'catonsville')
        sqlite_insert_params = '''INSERT INTO storage 
            (unit_number, storage_type, storage_size, storage_price, facility_id)
            VALUES (?,?,?,?,?)'''
        cur.execute(sqlite_insert_params, data_tuple)
    for i in range(1,43):
        data_tuple = (i, 'A', 'SMALL', 50, 'catonsville')
        sqlite_insert_params = '''INSERT INTO storage 
            (unit_number, storage_type, storage_size, storage_price, facility_id)
            VALUES (?,?,?,?,?)'''
        cur.execute(sqlite_insert_params, data_tuple)

def insertResponseData(cur):
    cur.execute('''INSERT INTO response
        (user_question, bot_response)
        VALUES ('Test User Question', 'Test Bot Response')''')

def getUnitAvailability(location, unitSize):
    #returns number of open units
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT COUNT(*) FROM storage WHERE storage_size = ? AND facility_id = ? AND customer_id IS NULL'''
    params = unitSize, location
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

def getOpenUnits(location):
    #return list of IDs of available units for a specific location
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT unit_number FROM storage WHERE ? = facility_id AND customer_id IS NULL'''
    selectData = cur.execute(select_statement, [location])
    try:
        return selectData.fetchall()
    finally:
        con.close()

def getUserUnits(userID, unitSize):
    #return list of unit IDs associated to a user ID
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT unit_number FROM storage WHERE customer_id = ? AND storage_size = ? '''
    params = userID, unitSize
    selectData = cur.execute(select_statement, params)
    try:
        return selectData.fetchall()
    finally:
        con.close()

def addUserUnit(userID, unitID, location):
    #takes in userID and unitID, and sets storage unit as reserved
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    update_statement = '''UPDATE storage SET customer_id = ? WHERE unit_number = ? AND facility_id = ?'''
    params = userID, unitID, location
    update = cur.execute(update_statement, params)
    con.commit()
    con.close()
    #modify customer_id field in storage table and set as userID if location and unitID match

def removeUserUnit(unitID, location):
    #given a unit ID and location, sets the userID field to null.
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    update_statement = '''UPDATE storage SET customer_id = null WHERE unit_number = ? AND facility_id = ?'''
    params = unitID, location
    update = cur.execute(update_statement, params)
    con.commit()
    con.close()

def addUserAccount(cName, cAddr, cCity, cState, cCountry, cPhone, cID):
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    insert_statement = '''INSERT INTO customer
        (customer_name, customer_address, customer_city, customer_state, customer_country, customer_phone, customer_id)
        VALUES (?,?,?,?,?,?,?)'''
    params = cName, cAddr, cCity, cState, cCountry, cPhone, [cID]
    insert = cur.execute(insert_statement, params)
    con.commit()
    con.close()

def addSimpleUserAccount(cID):
    #cName, cAddr, cCity, cState, cCountry, cPhone, cID
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    insert_statement = '''INSERT INTO customer
        (customer_id)
        VALUES (?)'''
    params = [cID]
    insert = cur.execute(insert_statement, params)
    con.commit()
    con.close()

def addResponse(user_msg, bot_rsp):
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    insert_statement = '''INSERT INTO response
        (user_question, bot_response)
        VALUES (?,?)'''
    params = user_msg, bot_rsp
    insert = cur.execute(insert_statement, params)
    con.commit()
    con.close()


# View Database
def getUsers():
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT * FROM customer'''
    selectData = cur.execute(select_statement)
    try:
        for row in selectData:
            print(row)
    finally:
        con.close()

def getFacilities():
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT * FROM facility'''
    selectData = cur.execute(select_statement)
    try:
        for row in selectData:
            print(row)
    finally:
        con.close()

def getUnits():
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT * FROM storage'''
    selectData = cur.execute(select_statement)
    try:
        for row in selectData:
            print(row)
    finally:
        con.close()

def getResponse():
    con = sqlite3.connect('chat.db')
    cur = con.cursor()
    select_statement = '''SELECT * FROM response'''
    selectData = cur.execute(select_statement)
    try:
        for row in selectData:
            print(row)
    finally:
        con.close()