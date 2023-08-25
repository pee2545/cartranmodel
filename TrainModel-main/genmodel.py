import pickle
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# ทำการอ่านไฟล์ featurevectortrain.pkl and featurevectortest.pkl ที่เก็บข้อมูล feature ค่า HOG ของรถยนต์ทั้งหมด 
featurevectortrain = pickle.load(open('featurevectortrain.pkl','rb'))
featurevectortest = pickle.load(open('featurevectortest.pkl','rb'))

# สร้าง Method เพื่อ set ข้อมูลให้กับ x_train, y_train, x_test, y_test
def setData(dataset):
    x = []
    y = []
    # กำหนดค่าที่อยู่ใน dataset(2 มิติ) ให้กับ x_train และ y_train
    # ใช้ enumberate เพื่อเข้าถึง list โดยจะส่ง ตำแหน่ง และค่าที่อยู่ใน ตำแหน่งนั้นๆ
    for index, value in enumerate(dataset):

        # เอาข้อมูลที่อยู่ใน dataset [ในตำแหน่งนั้นๆ] [ ในตำแหน่ง x ที่ 0 : จนถึงตัวสุดท้าย ] และลบ(-1)ไปอีก 1 ตำแหน่งเพราะนับ 0 ด้วย
        x.append( dataset[index][ : len(dataset[index])-1] )

        # เอาข้อมูลที่อยู่ใน dataset [ในตำแหน่งนั้นๆ] [ตำแหน่งสุดท้าย - 1 เพราะนับ 0 ด้วย]
        y.append( dataset[index][len(dataset[index])-1] )

    # ส่งกลับ feature vector(x) และ brand(y) ของ feature นั้นๆ
    return x, y

# เก็บข้อมูล feature vector(x) และ brand(y) ของ feature นั้นๆ จากการเรียกใช้ Method setData
x_train, y_train = setData(featurevectortrain)
x_test, y_test = setData(featurevectortest)

print('x_train:', len(x_train))
print('x_test:', len(x_test))
print('y_train:', len(y_train))
print('y_test:', len(y_test))

# สร้าง object ของ DecisionTreeClasssifier()
model = DecisionTreeClassifier()
# ส่งช้อมูล x_train, y_train เข้าไปทดสอบ
model = model.fit(x_train, y_train)

# ทดสอบประสิทธิภาพจากชุดข้อมูล โดยส่ง x_test เข้าไป
Ypred = model.predict(x_test)
# ค่าประสิทธิภาพที่ได้ โดยส่ง Ypred, y_test เข้าไป
accuracy = metrics.accuracy_score(y_test, Ypred) * 100

# matrix จากการ test model คำตอบที่ถูกต้องจะอยู่ในรูปแบบ แนวทแยง
matrix = confusion_matrix(y_test, Ypred)

print("\nAccuracy:", accuracy)

print("Confusion matrix:\n", matrix)


# ส่วนนี้ทำการเขียนข้อมูลจาก model ที่ได้เรียนรู้โมเดลไว้ลงในไฟล์โดยใช้รูปแบบการเก็บข้อมูล serialization เป็นระบบไฟล์ที่สามารถเก็บและกู้คืนได้
write_path = 'ClassifierCarModel.pkl'
pickle.dump(model, open(write_path, 'wb'))
print("\nClassifierCarModel.pkl file saved.")