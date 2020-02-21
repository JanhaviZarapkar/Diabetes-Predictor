import numpy as np
from tkinter import *
from tkinter import messagebox
root=Tk()
root.geometry("1600x800+0+0")
root.title("Diabetes Predictor")
preg=IntVar()
gl=IntVar()
BP=IntVar()
ST=IntVar()
insulin=IntVar()
bmi=IntVar()
dpf=DoubleVar()
age=IntVar()
img=PhotoImage(file="img2.png")

def clear():
    preg.set(" ")
    gl.set(" ")
    BP.set(" ")
    ST.set(" ")
    insulin.set(" ")
    bmi.set(" ")
    dpf.set(" ")
    age.set(" ") 

    
def Submit():
    p=preg.get()
    g=gl.get()
    bp=BP.get()
    st=ST.get()
    i=insulin.get()
    b=bmi.get()
    d=dpf.get()
    a=age.get()  
    if (p==" " or g==" " or bp==" " or st==" " or i==" " or b==" " or d==" " or a==" ") :
        messageBox.showinfo("ERROR", "Please fill all the entries")
    else:
        
        l1=[[p,g,bp,st,i,b,d,a]]
        print(l1)
        # Logistic Regression
        # Importing the libraries
        import numpy as np
        #import matplotlib.pyplot as plt
        import pandas as pd
        
        # Importing the dataset
        dataset = pd.read_csv('diabetes.csv')  
        X = np.array(dataset.drop('Outcome', 1))
        y = np.array(dataset['Outcome'])
        l1 = np.array(l1)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Data Preprocessing
        # Taking care of missing data
        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
        imputer = imputer.fit(X[:, 1:9])  # Upper bound is excluded only Index 1 and 2 is included
        X[:, 1:9] = imputer.transform(X[:, 1:9])  
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        l1_test = sc_X.transform(l1)
        
        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Outcome of cm shows array two rows - correct and incorrect predictions
        # example of making a single class prediction
        
        from sklearn.datasets.samples_generator import make_blobs
        # generate 2d classification dataset
        X, y = make_blobs(n_samples=100, centers=2, n_features=8, random_state=1)
        # fit final model
        model = LogisticRegression()
        model.fit(X, y)
        # define one new instance
        Xnew = l1_test
        # make a prediction
        ynew = model.predict(Xnew)
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
        out=ynew[0]
        if out==0:
            messagebox.showinfo("Result","You don't have diabetes")
        elif out==1:
            messagebox.showinfo("Result","You have diabetes")
   
frame0=Frame(root,height=800,width=1600,background="#FFB6C1").place(x=0,y=0)
l=Label(root,height=800,width=1600,image=img,background="#0000FF").place(x=0,y=0)
l1=Label(frame0,relief="groove",background="#0000FF",text="Welcome to Diabetes Predictor",height=5,width=100,font="Bold").place(x=100,y=20)
l2=Label(frame0,relief="raised",background="#0000FF",text="No. of Pregnencies",height=2,width=40,font="Bold").place(x=100,y=170)
l3=Label(frame0,relief="raised",background="#0000FF",text="Glucose",height=2,width=40,font="Bold").place(x=100,y=240)
l4=Label(frame0,relief="raised",background="#0000FF",text="Blood Pressure",height=2,width=40,font="Bold").place(x=100,y=310)
l5=Label(frame0,relief="raised",background="#0000FF",text="Skin Thickness",height=2,width=40,font="Bold").place(x=100,y=380)
l6=Label(frame0,relief="raised",background="#0000FF",text="Insulin",height=2,width=40,font="Bold").place(x=100,y=450)
l7=Label(frame0,relief="raised",background="#0000FF",text="BMI",height=2,width=40,font="Bold").place(x=100,y=520)
l8=Label(frame0,relief="raised",background="#0000FF",text="Diabetes Pedigree Function",height=2,width=40,font="Bold").place(x=100,y=590)
l9=Label(frame0,relief="raised",background="#0000FF",text="Age",height=2,width=40,font="Bold").place(x=100,y=660)
t1=Entry(frame0,text=preg,width=40,font="Bold").place(x=560,y=170)
t2=Entry(frame0,text=gl,width=40,font="Bold").place(x=560,y=240)
t3=Entry(frame0,text=BP,width=40,font="Bold").place(x=560,y=310)
t4=Entry(frame0,text=ST,width=40,font="Bold").place(x=560,y=380)
t5=Entry(frame0,text=insulin,width=40,font="Bold").place(x=560,y=450)
t6=Entry(frame0,text=bmi,width=40,font="Bold").place(x=560,y=520)
t7=Entry(frame0,text=dpf,width=40,font="Bold").place(x=560,y=590)
t8=Entry(frame0,text=age,width=40,font="Bold").place(x=560,y=660)
preg.set(" ")
gl.set(" ")
BP.set(" ")
ST.set(" ")
insulin.set(" ")
bmi.set(" ")
dpf.set(" ")
age.set(" ")
b1 = Button(frame0,text = "Submit" , height = 2, width = 20 , font = "Bold", command = Submit,bg="#0000FF").place(x = 250,y=720)
b2 = Button(frame0,text = "Clear" , height = 2, width = 20 , font = "Bold", command = clear,bg="#0000FF").place(x = 500,y=720)
#b3 = Button(frame0,text = "Exit" , height = 2, width = 20 , font = "Bold", command = root.quit,bg="#0000FF").place(x = 650,y=720)
root.mainloop()
