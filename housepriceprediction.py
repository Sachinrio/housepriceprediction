import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import plotly_express as px
import streamlit as st
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,KFold,cross_val_score,RandomizedSearchCV,LeaveOneOut
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,roc_auc_score

data=pd.read_csv(r"C:\Users\sachi\Downloads\train.csv")
st.title("HOUSE PRICE PREDICTION")
st.set_page_config("house_price_prediction",layout="centered")
st.subheader("Data Preview")
st.write(data.head())
l=LabelEncoder()

a=data.drop("SalePrice",axis=1)
b=data.drop("SalePrice",axis=1)
str_col=data.select_dtypes(include="object").columns.to_list()
mapping={}
for i in str_col:
    data[i]=l.fit_transform(data[i])
    for k,v in enumerate(l.classes_):
        mapping[v]=k
print(mapping)
corr=data.corr()
data.dropna(inplace=True)
corr=data.corr()
top5=corr.unstack().sort_values(ascending=False)
top5=top5[top5<1]
top5=top5.head(20)
cor=[]
for k,v in top5.items():
    for i in range(len(k)):
       if k[i] in cor :
          continue
       else:
           cor.append(k[i])
x=data[cor]
x=x.drop("SalePrice",axis=1)
print(x.columns)
y=data["SalePrice"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train_st=StandardScaler().fit_transform(x_train)
x_test_st=StandardScaler().fit_transform(x_test)
models=st.selectbox("Select the model",options=[LinearRegression(),RandomForestRegressor(random_state=42),DecisionTreeRegressor(random_state=42),KNeighborsRegressor(n_neighbors=4)])
models.fit(x_train_st,y_train)
y_pred=models.predict(x_test_st)
st.write("r2_score",r2_score(y_test,y_pred))
st.write("MSE",mean_squared_error(y_test,y_pred))
st.write("MAE",mean_absolute_error(y_test,y_pred))
st.sidebar.title("DATA VIZUALIZATION")
chart=st.sidebar.selectbox("Enter the chart you want",options=["LineChart","BarChart","Histogram","Heatmap","PieChart","ScatterChart"])
xaxis=st.sidebar.selectbox("X_AXIS",options=data.select_dtypes(include="int").columns.to_list())
yaxis=st.sidebar.selectbox("Y_AXIS",options=data.select_dtypes(include="int").columns.to_list())
if chart=="LineChart":
    st.subheader(chart)
    fig=px.line(data,xaxis,yaxis,title=f"{xaxis} X {yaxis}")
    st.plotly_chart(fig)
elif chart=="BarChart":
    st.subheader(chart)
    fig=px.bar(data,xaxis,yaxis,title=f"{xaxis} X {yaxis}")
    st.plotly_chart(fig)
elif chart=="Histogram":
    st.subheader(chart)
    fig=px.histogram(data,xaxis,nbins=20,title=f"{xaxis}")
    st.plotly_chart(fig)
elif chart=="PieChart":
    st.subheader(chart)
    fig=px.pie(data,names=xaxis,values=yaxis,title=f"{xaxis} X {yaxis}")
    st.plotly_chart(fig)
elif chart=="Heatmap":
    cor=data[[xaxis,yaxis]].corr()
    st.subheader(chart)
    fig,ax=plt.subplots()
    sns.heatmap(cor, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.subheader(chart)
    fig=px.scatter(data,xaxis,yaxis,title=f"{xaxis}  X  {yaxis}")
    st.plotly_chart(fig)
st.subheader("Get the new customer Data")
new_data=[]
with st.form("MyForm"):
    for i in x.columns:
        if i not in str_col:
           a=st.number_input(f"Enter the {i} value")
           new_data.append(a)
        else:
           b[i]=l.fit_transform(b[i])
           a=st.selectbox(f"Enter the {i} value",options=l.classes_)
           new_data.append(mapping[a])
    sub=st.form_submit_button("prediction")
print(x.columns)
if sub:
    models.fit(x_train_st,y_train)
    input=np.array(new_data).reshape(1,-1)
    y_pred=models.predict(input)
    st.success(y_pred)
    st.success(r2_score(y_test,y_pred))
