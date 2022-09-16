import requests

class Request:

     def sendData(self,patientID,image):
        res = requests.get('http://127.0.0.1:5000/add',params={'patientID':int(patientID),'image':str(image)}) 
        print(res.text) 
    
     def get(self,start,end):
        res = requests.get('http://127.0.0.1:5000/get',params={'start':str(start),'end':str(end)}) 
        print(res.text) 
    
    
data=Request()

#data.sendData(35,'C:\\Users\\HP\Desktop\\g1.jpg')

data.get('10/09/2022','12/09/2022')


