from django.shortcuts import render
from ml_app.myform import user_info
"""import numpy as np
import matplotlib.pyplot as mp
import pandas as pa
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split"""

# Create your views here.
def ml_code_fun(data):
    # -*- coding: utf-8 -*-

    """
    Created on Mon Jan  6 19:14:53 2020

    @author: umgkr
    """

    import numpy as np
    import matplotlib.pyplot as mp
    import pandas as pa
    k=pa.read_csv('C:\mechine_learing\Machine-Learning-A-Z-New\Machine Learning A-Z New\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data.csv')
    l=k.iloc[:,:-1].values
#data = np.array(f['abc']['efg']['xyz'])
#print(l[0]["Country"])
    print(type(l))
    """data =np.array([["c_n",11,11000],])
    data1 = data1
#l[len(l)-1]=data
    data1=l
#l.append(data,ignore_index=True)
#l=np.append(l,data,axis=0)
#l.append(data)

    data=pa.DataFrame(data)"""
    l1 = pa.DataFrame(l)
    data=pa.DataFrame(data)
    l1=l1.append(data,ignore_index=True)
    l=np.array(l1)
    ll = l
    print(type(l1))
    y=k.iloc[:,3].values
    """y1=np.array(["Yes"])
    y1=pa.DataFrame(y1)
    y = pa.DataFrame(y)
    y=y.append(y1,ignore_index=True)
    y=np.array(y)
    y1=y"""
    #print(sklearn.impute.objets.all())
    from sklearn.impute import SimpleImputer
    #,LabelEncoder,OneHotEncoder
    m=SimpleImputer(missing_values=np.nan , strategy='mean')
    m=m.fit(l[:,1:3])
    l[:,1:3]=m.transform(l[:,1:3])

    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.compose import ColumnTransformer

    #l[:,0]=la.fit_transform(l[:,0])
    #ohe=OneHotEncoder(categorical_features = [0])

    #ohe =  OneHotEncoder(categories='auto')

    ohe = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')

    #ohe1 = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')


    la=LabelEncoder()


    l=np.array(ohe.fit_transform(l))
    #data1 = np.array(ohe.fit_transform(data1))
    y=la.fit_transform(y)
    """from sklearn.model_selection import train_test_split
    x_tr,x_te,y_tr,y_te=train_test_split(l,y,test_size=0.2,random_state=0)"""

    x_tr = l[:-1,:]
    y_tr = y
    x_te = l[len(l)-1:len(l),:]
    x = np.array(l[-1,:])
    print(type(x_te))
    print(type(x))
    """from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    x_tr=ss.fit_transform(x_tr)
    x_te=ss.transform(x_te)"""
    from sklearn.linear_model import LinearRegression
    re=LinearRegression()
    re.fit(x_tr,y_tr)
    y_pr = re.predict(x_te)
    return y_pr


def home(request):
    return render(request,"ml_files/home.html")

def result(request,key="FIRST FILL USER DATA"):
          
    return render(request,"ml_files/result.html",{"key":key})


def info_page(request):
    key="none"
    ui = user_info()
    if request.method == "POST":

        ui = user_info(request.POST)
        if ui.is_valid():
            c_n = ui.cleaned_data["country_name"]
            age = ui.cleaned_data["Age"]
            sal = ui.cleaned_data["Salary"]

            #import pandas as pa
            import numpy as np

            data =np.array([[c_n,str(age),str(sal)],])

            p_data=ml_code_fun(data)
            print(p_data)
            if p_data >=1:
                key="Yes"
            else:
                key="No"
            return result(request,key)
    data_dic = {"form":ui}
    return render(request,"ml_files/user_form.html",context=data_dic)
