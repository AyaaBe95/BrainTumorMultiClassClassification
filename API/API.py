import base64
from concurrent.futures import thread
from datetime import datetime
import random
import threading
import json
import sqlite3
import time
import firebase_admin
from firebase_admin import credentials , db
import urllib.request
import pandas as pd
import cv2
from flask import Flask ,request, jsonify


cred = credentials.Certificate("databases/finalproject-d0098-firebase-adminsdk-6xssm-13f86075b6.json")
firebase_admin.initialize_app(cred,{'databaseURL':'https://finalproject-d0098-default-rtdb.firebaseio.com/','httpTimeout':30})

conn = sqlite3.connect('databases/Project.db',check_same_thread=False)
curr=conn.cursor()

query='''CREATE TABLE IF NOT EXISTS Data  (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    img BLOB,
    img_size TEXT,
	date TEXT,
    time TEXT,
	result TEXT,
	flag INTEGER
);'''

#curr.execute(query)



# convert image to binary to insert it in local database
def convertToBinaryData(filename):
    with open(filename,"rb") as file:
        blobData = file.read()
    return blobData

# convert image to binary to insert it in firebase
def convertToBinaryData2(filename):
    img_dict = {}
    with open(filename,"rb") as file:
        data = file.read()
        img_dict['image']= str(data)
        json_str = json.dumps(img_dict)
        dict_from_json = json.loads(json_str)
        image_base64_string = dict_from_json['image']
    return image_base64_string

app = Flask(__name__)


# get image from the user,then insert the image,image's size,current date,current time and the result of prediction 

@app.route('/add')
def insert_data():
        conn
        curr
        image=str(request.args.get('image'))
        sql = """ INSERT INTO Data
                (img,img_size,date,time,result,flag) VALUES (?, ?, ?, ?, ?, ?)"""

        img = convertToBinaryData(image) # local database
        img2=convertToBinaryData2(image) # firebase
        now = datetime.now()
        curr_date = now.strftime("%d/%m/%Y")
        curr_time=now.strftime("%H:%M:%S")
        result='none'
        flag=0 # if 1 means data has been inserted into local database and firebase ,if not means data has been inserted only in firebase
        data_tuple = (img, 1 , str(curr_date), str(curr_time), result,flag)
        try:
            curr.execute(sql, data_tuple)
            imgID = curr.lastrowid
            coded_image=readImage(int(imgID))
            img_size=coded_image.shape # get size of the image
            size=updateImageSize(imgID,img_size) # update its size

            if connect(): # check if there is an internet

                updateFlag(imgID) # set flag = 1
                insert_into_firebase(imgID,img2,size,curr_date,curr_time,result) # insert data into firebase
            conn.commit()
            return jsonify(size)
        except sqlite3.Error as error:
            return "Failed to insert data ", error

        finally:
            if conn:
                conn.commit()

# get a reprot that contains info between two dates
@app.route('/get')
def get_data_from_interval():
    curr
    conn
    sql = 'SELECT img_size,date,time,result FROM  Data WHERE date between ? and ? '
    start=str(request.args.get('start'))
    end=str(request.args.get('end'))

    try:
        curr.execute(sql,(start,end))
        data = curr.fetchall()
        conn.commit()
        df=pd.DataFrame(data=data,columns=['Image Size','Date','Time','Result'])
        n = random.randint(1,99)
        filename='report' + str(n) # generate the file
        path ='reports/' + filename +'.csv' 
        df.to_csv(path,index=False)
        return 'Report has been generated in reports folder as ' + str(filename)
    except Exception as e :
        return e
    finally:
        if conn:
            conn.commit()
            conn.close()


def run_app():
    app.run(debug=False,threaded=True)

# insert data into firebase
def insert_into_firebase(id,img,size,date,time,result):
    ref='/data/' + str(id) + '/'
    root=db.reference(ref)
    try:
        data={
            'id':id,
            'img':img,
            'img_size':size,
            'date':date,
            'time':time,
            'result':result
            }
        root.set(data)
        return True
    except Exception as e:
         return False

# check if there is an internet
def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


# retrive the info based on id 
def getImageById(id):
    conn
    curr
    sql="SELECT * FROM Data where id == " + str(id)
    try:
        curr.execute(sql)
        conn.commit()
        record = curr.fetchall()
        for row in record:
            img = row[1]
        path= './resulted_images/img' + str(id) + '.jpg'

        with open(path,"wb") as file:
            file.write(img)
    except:
        return "Data does not exsist"
    finally:
        if conn:
            conn.commit()
    return path

# read the image 
def readImage(id):
    img=getImageById(id)
    img=cv2.imread(img)
    return img

# update image's size after insertion
def updateImageSize(id,size):
    conn
    curr
    tupleSize=tuple(size)
    sql = 'UPDATE Data SET img_size = ? where id == ? '
    try:
        curr.execute(sql,(str(tupleSize),id))
        conn.commit()
        return str(tupleSize)
    except:
        return "Data does not exsist"
    finally:
        if conn:
            conn.commit()

# update the result after prediction
def update_result(id,result):
    conn
    curr
    sql = 'UPDATE Data SET result = ? where id == ? '
    try:
        curr.execute(sql,(result,id))
        conn.commit()
        return result
    except:
        return "Data does not exsist"
    finally:
        if conn:
            conn.commit()

# update the flag when there is an internet connection
def updateFlag(id):
    conn
    curr
    sql = 'UPDATE Data SET flag = ? where id == ? '
    try:
        curr.execute(sql,(1,id))
        conn.commit()
    except:
        return "Data does not exsist"
    finally:
        if conn:
            conn.commit()

# retrieve the data that not inserted into firebase
def getData():
    conn
    curr 
    sql = 'SELECT id, img , img_size,date,time,result FROM  Data WHERE flag == 0'
    try:
        curr.execute(sql)
        data = curr.fetchall()
        conn.commit()
        return data
    except:
        return "Data does not exsist"
    finally:
        if conn:
            conn.commit()

# upload the data that has not inserted into firebase in order to inserted it when there is an internet

def upload_unsaved_data():
    data = getData()
    dict={}
    list=[]
    if(len(data)> 0 and connect()):
        for i in range(len(data)):
            dict={
                'id':data[i][0],
                'img':str(data[i][1]),
                'img_size':data[i][2],
                'date':str(data[i][3]),
                'time':str(data[i][4]),
                'result': str(data[i][5])
            }
            list.append(dict)

        for i in list:
            ref='/data/' + str(i['id']) + '/'
            root=db.reference(ref)
            root.set(i)
            updateFlag(i['id'])


# run the app every 5 minutes in order to upload unsaved data into firebase
if __name__ == "__main__":
    while 1:
        first_thread = threading.Thread(target=run_app)
        second_thread = threading.Thread(target=upload_unsaved_data)
        first_thread.start()
        second_thread.start()
        time.sleep(20)

