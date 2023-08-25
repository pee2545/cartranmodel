import cv2 # เพื่อประมวลผลภาพ
import requests # เพื่อใช้ทำการ requests ไปยัง api ที่เราสร้างไว้เเละรันใน Docker Desktop => Run the Containers
import numpy as np
import pickle # สำหรับการจัดเก็บและโหลดข้อมูลในรูปแบบของไบนารี (binary data) หรือเรียกว่า Serialization (การเปลี่ยนข้อมูลในรูปของออบเจ็กต์ให้เป็นรูปของไบนารี) และ Deserialization (การแปลงข้อมูลในรูปของไบนารีกลับมาเป็นออบเจ็กต์)
import os # สำหรับจัดการระบบไฟล์ เเละ การทำงานกับ os ซึ่งรวมถึงการเข้าถึงไฟล์ เเละ Folder ในระบบไฟล์
import base64 # ใช้ในการเเปลงรูปภาพให้เป็นการเข้ารหัสในรูปแบบ base64

url = 'http://localhost:8080/api/genhog'

# function img2vec() ใช้ในการรับภาพเข้ามาเพื่อทำการส่งไปยัง API ที่รันอยู่บน docker เพื่อหาเอกลักษณ์ของภาพด้วย HOG โดยส่งไปเเบบ base64
def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img) 
    img_str = base64.b64encode(buffer) 
    data = "image data,"+str.split(str(img_str),"'")[1] 
    response = requests.post(url, json={"image_base64":data})
    
    return response.json()

#img = cv2.imread('train\\Audi\\110.jpg')
#print(img2vec(img))

# ส่วนนี้เป็นการโหลดรูปภาพ Folder Train เพื่อเเปลงจากภาพให้กลายมาเป็น feature vector ด้วยการเรียก API ที่รันอยู่บน docker desktop 
PathTrain = 'train'
FeatureVectorTrain = []

for y in os.listdir(PathTrain): # วน loop เพื่อเเยก folder ย่อย, y จะเก็บ folder ยี่ห้อรถ 7 ยี่ห้อ
    for fn in os.listdir(os.path.join(PathTrain,y)): # ทำการวน loop เพื่อเเยกไฟล์ที่อยู่ใน y โดยผลลัพธ์จะได้ชื่อไฟล์ออกมาเช่น 90.jpg
        img_file_name = os.path.join(PathTrain,y)+"/"+fn # สร้างชื่อไฟล์ภาพ output = train\Audi/90.jpg
        X = cv2.imread(img_file_name) # ทำการอ่านรูปภาพออกมา
        res = img2vec(X) # ทำการส่ง X เข้าไปเพื่อ requests ไปยัง api เพื่อหาเอกลักษณ์ของภาพด้วย HOG
        vec = list(res["vector"])
        vec.append(y)
        FeatureVectorTrain.append(vec)



# ส่วนนี้เป็นการโหลดรูปภาพ Folder Test เพื่อเเปลงจากภาพให้กลายมาเป็น feature vector ด้วยการเรียก API ที่รันอยู่บน docker desktop 
PathTest = 'test'
FeatureVectorTest = []

for y in os.listdir(PathTest): # วน loop เพื่อเเยก folder ย่อย, y จะเก็บ folder ยี่ห้อรถ 7 ยี่ห้อ
    for fn in os.listdir(os.path.join(PathTest,y)): # ทำการวน loop เพื่อเเยกไฟล์ที่อยู่ใน y โดยผลลัพธ์จะได้ชื่อไฟล์ออกมาเช่น 90.jpg
        img_file_name = os.path.join(PathTest,y)+"/"+fn # สร้างชื่อไฟล์ภาพ output = test\Audi/90.jpg
        X = cv2.imread(img_file_name) # ทำการอ่านรูปภาพออกมา
        res = img2vec(X) # ทำการส่ง X เข้าไปเพื่อ requests ไปยัง api เพื่อหาเอกลักษณ์ของภาพด้วย HOG
        vec = list(res["vector"])
        vec.append(y)
        FeatureVectorTest.append(vec)


for index, data in enumerate(FeatureVectorTest):
    print(data)
    if index == 1:
        break

# ส่วนนี้ทำการเขียนข้อมูลจาก FeatureVectorTrain and FeatureVectorTest ลงในไฟล์โดยใช้รูปแบบการเก็บข้อมูล serialization เป็นระบบไฟล์ที่สามารถเก็บและกู้คืนได้
write_path = "featurevectortrain.pkl"
pickle.dump(FeatureVectorTrain, open(write_path,"wb"))
print("data preparation is done")

write_path = "featurevectortest.pkl"
pickle.dump(FeatureVectorTest, open(write_path,"wb"))
print("data preparation is done")