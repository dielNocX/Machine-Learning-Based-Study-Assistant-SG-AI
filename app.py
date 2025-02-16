import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score    
import matplotlib.pyplot as plt
import seaborn as sns
from sentimentAnalysis import chatbot
import seaborn as sns

def data():
    st.write('Rows, Column: ',df.shape)
    with st.spinner("Loading dataset..."):
        st.write("Dataset:", df.head())
    st.divider()


def heatmap():
    with st.spinner("Loading heatmap..."):
        correlation_matrix = fiturNumerik.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap of Numerical Features")
        st.pyplot(plt)


def histogram():
    column = st.selectbox('Select Column',kolomNumerik)
    with st.spinner("Loading histogram..."):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], kde =True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)


def boxplot():

    kolomNumerik.remove('Exam_Score')
    column = st.selectbox('Pilih kolom', kolomNumerik)
    with st.spinner("Loading boxplot..."):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=column, y='Exam_Score', palette='Set3', hue=column, legend=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8, 'linestyle': 'none'})
        plt.title(f'Exam Score Distribution by {column}')
        st.pyplot(plt)

def statistic():
    st.write(df.describe())

def home():
    data()
    Visualization = st.radio('Visualization', ["None","Heatmap","Histogram","Statistic"])

    if Visualization == "Heatmap" :
        st.write("Heatmap")
        heatmap()
    #elif Visualization == "Boxplot":
    #    st.write("Boxplot")
    #    boxplot()
    elif Visualization == "Histogram":
        st.write("Histogram")
        histogram()
    elif Visualization == "Statistic":
        st.write("Statistic")
        statistic()

    st.divider()

    st.subheader("Model Evaluation:")
    st.write(f"R² Score: {regresi_r2:.2f}")
    
    st.divider()

def convert_to_numeric(var):
    if var == "Low":
        return 1
    elif var == "Medium":
        return 2
    else:
        return 3

def tool():
    #English
    st.header("Exam Score Predictor Tool")


    hours_studied = st.number_input("Hours Studied", min_value=0 ,value="min", placeholder="Type a number...")
    attendance = st.number_input("Attendance", min_value=0 ,value="min", placeholder="Type a number...")
    parental_involvement = st.selectbox("Parental Involvement",("Low", "Medium", "High"))
    access_to_resources = st.selectbox("Access to Resources",("Low", "Medium", "High"))
    previous_scores = st.number_input("Previous Score", min_value=0, max_value=100 ,value="min", placeholder="Type a number...")
    st.caption("_If you don't have previous test/exam scores, enter a score of '75' or whatever score you like._")
    motivation_level = st.selectbox("Motivation Level",("Low", "Medium", "High"))
    tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0 ,value="min", placeholder="Type a number...")
    family_income = st.selectbox("Family Income",("Low", "Medium", "High"))
    peer_influence = st.selectbox("Peer Influence",("Negative", "Neutral", "Positive"))
    parental_education_level = st.selectbox("Parental Education Level",("High School", "College", "Postgraduate"))

    st.divider()
    st.write("Hours Studied: ", hours_studied)
    st.write("Attendance: ", attendance)
    st.write("Parental Involvement: ", parental_involvement)
    st.write("Access to Resources: ", access_to_resources)
    st.write("Previous Scores: ", previous_scores)
    st.write("Motivation Level: ", motivation_level)
    st.write("Tutoring Sessions: ", tutoring_sessions)
    st.write("Family Income: ", family_income)
    st.write("Peer Influence: ", peer_influence)
    st.write("Parental Education Level: ", parental_education_level)

    parental_involvement = convert_to_numeric(parental_involvement)
    access_to_resources = convert_to_numeric(access_to_resources)
    motivation_level = convert_to_numeric(motivation_level)
    family_income = convert_to_numeric(family_income)
    
    if peer_influence == "Negative":
        peer_influence = 1
    elif peer_influence == "Neutral":
        peer_influence = 2
    else:
        peer_influence = 3

    if parental_education_level == "High School":
        parental_education_level = 1
    elif parental_education_level == "College":
        parental_education_level = 2
    else:
        parental_education_level = 3

    predict = st.button("Predict")
    if predict:
        predict_result = prediksi_nilai_ujian(hours_studied, attendance, parental_involvement, access_to_resources, previous_scores, motivation_level, tutoring_sessions, family_income, peer_influence, parental_education_level, regresi)
        st.write("Exam Score prediction result:", max(min(predict_result,100),0))
    

#   Hours_Studied
#   Attendance
#   Parental_Involvement	
#   Access_to_Resources	
#   Previous_Scores
# 	Motivation_Level
# 	Tutoring_Sessions
# 	Family_Income
# 	Peer_Influence
# 	Parental_Education_Level


    

def menu():
    choice = st.sidebar.radio(label="Opsi", options = ["Home","Tool","Chatbot"])
    if choice == "Home":
        home()
    elif choice == "Tool":
        tool()
    elif choice == "Chatbot":
        chatbot()

st.title("Student Performance Regression Model")
st.divider()
path = './StudentPerformanceFactors.csv'
df = pd.read_csv(path)
# st.write("tes",df.head())

#st.subheader("Handling Missing Values")
categorical_columns_with_missing = ['Parental_Education_Level', 'Distance_from_Home', 'Teacher_Quality']
for column in categorical_columns_with_missing:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)

kolomNumerik = ['Hours_Studied', 'Attendance',  'Previous_Scores', 'Tutoring_Sessions', 'Exam_Score']
fiturNumerik = df[kolomNumerik]


# #st.subheader("Encoding Categorical Variables")
# kolomKategori = df.select_dtypes(include=['object']).columns.tolist()
# df = pd.get_dummies(df, columns=kolomKategori, drop_first=True)
# #st.write("Data after encoding:", df.head())

df.dropna(inplace=True)
mapping = {'Low': 1, 'Medium': 2, 'High': 3}
for col in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality']:
  if col in df.columns:
    df[col] = df[col].map(mapping)


binary_mapping = {'Yes': 1, 'No': 0}
for col in ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']:
    if col in df.columns:
        df[col] = df[col].map(binary_mapping)

school_type_mapping = {'Public': 1, 'Private': 0}
if 'School_Type' in df.columns:
    df['School_Type'] = df['School_Type'].map(school_type_mapping)


parental_education_mapping = {'High School': 1, 'College': 2, 'Postgraduate': 3}
if 'Parental_Education_Level' in df.columns:
    df['Parental_Education_Level'] = df['Parental_Education_Level'].map(parental_education_mapping)

gender_mapping = {'Male': 0, 'Female': 1}
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map(gender_mapping)

distance_mapping = {'Near': 1, 'Moderate': 2, 'Far': 3}
if 'Distance_from_Home' in df.columns:
    df['Distance_from_Home'] = df['Distance_from_Home'].map(distance_mapping)

Peer_influence_mapping = {'Negative':1, 'Neutral': 2, 'Positive': 3}
if 'Peer_Influence' in df.columns:
    df['Peer_Influence'] = df['Peer_Influence'].map(Peer_influence_mapping)



#st.subheader("Removing Outliers")
z = np.abs((df - df.mean()) / df.std())
threshold = 3
df = df[(z < threshold).all(axis=1)]
#st.write("Data after removing outliers:", df.head())


#before
# Hours_Studied
# Attendance
# Parental_Involvement
# Access_to_Resources
# Extracurricular_Activities
# Sleep_Hours
# Previous_Scores	
# Motivation_Level	
# Internet_Access	
# Tutoring_Sessions	
# Family_Income	
# Teacher_Quality	
# School_Type	
# Peer_Influence	
# Physical_Activity	
# Learning_Disabilities	
# Parental_Education_Level	
# Distance_from_Home	
# Gender	
# Exam_Score


df = df.drop(["Internet_Access",
 "School_Type",
 "Gender",
 "Sleep_Hours",
 "Physical_Activity",
 "Extracurricular_Activities",
 "Learning_Disabilities",
 "Teacher_Quality", "Distance_from_Home"], axis=1, errors = "ignore")

# After
#   Hours_Studied
#   Attendance
#   Parental_Involvement	
#   Access_to_Resources	
#   Previous_Scores
# 	Motivation_Level
# 	Tutoring_Sessions
# 	Family_Income
# 	Peer_Influence
# 	Parental_Education_Level
# 	Exam_Score



#st.subheader("Splitting the Data")
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
with st.spinner("Splitting the data..."):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#st.write("Training and testing data prepared.")


#st.subheader("Training Linear Regression Model")
regresi = LinearRegression()
with st.spinner("Training the data..."):
    regresi.fit(X_train, y_train)


with st.spinner("Counting R^2 score..."):
    regresi_pred = regresi.predict(X_test)
    regresi_r2 = r2_score(y_test, regresi_pred)

def prediksi_nilai_ujian(hours_studied, attendance, parental_involvement, access_to_resources, previous_scores, motivation_level, tutoring_sessions, family_income, peer_influence, parental_education_level, model=regresi):
    # Membuat input untuk prediksi
    input_data = [[hours_studied, attendance, parental_involvement, access_to_resources, previous_scores, motivation_level, tutoring_sessions, family_income, peer_influence, parental_education_level]]

    # Melakukan prediksi
    predicted_score = model.predict(input_data)

    return int(predicted_score)

st.sidebar.title('Menu')
menu()
st.sidebar.write('')

