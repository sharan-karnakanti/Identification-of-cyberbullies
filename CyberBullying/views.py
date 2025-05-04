from django.shortcuts import render, redirect
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import datetime
import re
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from numpy.linalg import norm
from numpy import dot
import os

classifier = None
tfidf_vectorizer = None
corpus = []
label_count = 0

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def SendPost(request):
    if request.method == 'GET':
        return render(request, 'SendPost.html', {})

def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})

def Admin(request):
    if request.method == 'GET':
        return render(request, 'Admin.html', {})

def Login(request):
    if request.method == 'GET':
        return render(request, 'Login.html', {'data': ''})

def AddCyberMessages(request):
    if request.method == 'GET':
        return render(request, 'AddCyberMessages.html', {})

def RunAlgorithms(request):
    if request.method == 'GET':
        return render(request, 'RunAlgorithms.html', {})

def delete_post(request):
    posttime = request.GET.get('posttime')
    if posttime:
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("DELETE FROM posts WHERE posttime = %s", (posttime,))
            con.commit()
    return redirect('MonitorPost')

def MonitorPost(request):
    if request.method == 'GET':
        strdata = '<table border=1 align=center width=100%><tr><th>Sender Name</th><th>File Name</th><th>Message</th><th>Post Time</th><th>Status</th><th>Delete</th></tr>'
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM posts")
            rows = cur.fetchall()
            for row in rows:
                filename = os.path.basename(str(row[1]))
                strdata += '<tr>'
                strdata += '<td>' + str(row[0]) + '</td>'
                strdata += '<td><img src="/static/photo/' + filename + '" width=200 height=200 alt="Image not found"></td>'
                strdata += '<td>' + str(row[2]) + '</td>'
                strdata += '<td>' + str(row[3]) + '</td>'
                strdata += '<td>' + str(row[4]) + '</td>'
                strdata += '<td><a href="/delete_post/?posttime=' + str(row[3]) + '" onclick="return confirm(\'Are you sure you want to delete this post?\')">Delete</a></td>'
                strdata += '</tr>'
                print(f"Image URL: /static/photo/{filename}")
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                print(f"File path: {file_path}, Exists: {os.path.exists(file_path)}")
        context = {'data': strdata}
        return render(request, 'MonitorPost.html', context)

def AddBullyingWords(request):
    if request.method == 'POST':
        message = request.POST.get('t1', False)
        label = request.POST.get('t2', False)
        message = message.strip('\n').strip().lower()
        message = re.sub(r'[^a-zA-Z\s]+', '', message)
        file = open('dataset.txt', 'a+')
        file.write(message + "," + label + "\n")
        file.close()
        context = {'data': 'Cyber Words added to dataset as ' + label}
        return render(request, 'AddCyberMessages.html', context)

def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        db_connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO users(username,password,contact_no,email,address,status) VALUES(%s,%s,%s,%s,%s,'Accepted')"
        db_cursor.execute(student_sql_query, (username, password, contact, email, address))
        db_connection.commit()
        if db_cursor.rowcount == 1:
            context = {'data': 'Signup Process Completed'}
        else:
            context = {'data': 'Error in signup process'}
        return render(request, 'Register.html', context)

def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM users")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password and row[5] == 'Accepted':
                    with open('session.txt', 'w') as file:
                        file.write(username)
                    context = {'data': 'welcome ' + username}
                    return render(request, 'UserScreen.html', context)
        context = {'data': 'login failed'}
        return render(request, 'Login.html', context)
    return render(request, 'Login.html', {'data': ''})

def AdminLogin(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context = {'data': 'welcome ' + username}
            return render(request, 'AdminScreen.html', context)
        context = {'data': 'login failed'}
        return render(request, 'Admin.html', context)
    return render(request, 'Admin.html', {'data': ''})

def ViewUsers(request):
    if request.method == 'GET':
        color = '<font size="" color=black>'
        strdata = '<table border=1 align=center width=100%><tr><th>Username</th><th>Password</th><th>Contact No</th><th>Email ID</th><th>Address</th><th>Status</th></tr>'
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM users")
            rows = cur.fetchall()
            for row in rows:
                strdata += '<tr><td>' + color + row[0] + '</td><td>' + color + row[1] + '</td><td>' + color + row[2] + '</td><td>' + color + str(row[3]) + '</td><td>' + color + str(row[4]) + '</td><td>' + color + row[5] + '</td></tr>'
        context = {'data': strdata + '</table><br/><br/><br/>'}
        return render(request, 'ViewUsers.html', context)

def ViewUserPost(request):
    if request.method == 'GET':
        strdata = '<table border=1 align=center width=100%><tr><th>Sender Name</th><th>File Name</th><th>Message</th><th>Post Time</th><th>Status</th></tr>'
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM posts")
            rows = cur.fetchall()
            for row in rows:
                filename = os.path.basename(str(row[1]))
                strdata += '<tr><td>' + str(row[0]) + '</td><td><img src="/static/photo/' + filename + '" width=200 height=200 alt="Image not found"></td><td>' + str(row[2]) + '</td><td>' + str(row[3]) + '</td><td>' + str(row[4]) + '</td></tr>'
                print(f"Image URL: /static/photo/{filename}")
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                print(f"File path: {file_path}, Exists: {os.path.exists(file_path)}")
        context = {'data': strdata}
        return render(request, 'ViewUserPost.html', context)

def word_count(str):
    counts = dict()
    words = str.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred


def cal_accuracy(y_test, y_pred, details):
    msg = ''
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    msg += f"<h3>{details}</h3>"
    msg += f"<p>Accuracy: {accuracy:.2f}%</p>"
    msg += "<h4>Classification Report:</h4>"
    msg += "<pre>" + str(classification_report(y_test, y_pred)) + "</pre>"
    msg += "<h4>Confusion Matrix:</h4>"
    msg += "<table border='1' style='border-collapse: collapse; text-align: center;'>"
    msg += "<tr><th></th><th>Predicted Non-Bullying</th><th>Predicted Bullying</th></tr>"
    msg += f"<tr><td>Actual Non-Bullying</td><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>"
    msg += f"<tr><td>Actual Bullying</td><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>"
    msg += "</table>"
    return msg

def PostSent(request):
    global classifier, tfidf_vectorizer
    if request.method == 'POST' and request.FILES['t2']:
        myfile = request.FILES['t2']
        msg = request.POST.get('t1', False)
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = os.path.basename(fs.save(myfile.name, myfile))
        saved_file_path = os.path.join(settings.MEDIA_ROOT, filename)
        print(f"Saved file path: {saved_file_path}")
        print(f"File exists: {os.path.exists(saved_file_path)}")
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if not msg or len(msg.strip()) < 5:
            return render(request, 'SendPost.html', {'data': 'Error: Message too short or empty.'})

        msg = msg.strip().lower()
        msg = re.sub(r'[^a-zA-Z\s]+', '', msg)
        print("Input message:", msg)
        print("Processed message:", msg)

        if 'tfidf_vectorizer' not in globals() or tfidf_vectorizer is None:
            return render(request, 'SendPost.html', {'data': 'Vectorizer not initialized. Run the algorithm first.'})

        X1 = tfidf_vectorizer.transform([msg])
        print("Input feature vector shape:", X1.shape)
        print("Non-zero feature indices:", X1.nonzero())
        print("Non-zero feature values:", X1.data)

        if classifier is None:
            return render(request, 'SendPost.html', {'data': 'Classifier not initialized. Run the algorithm first.'})

        probs = classifier.predict_proba(X1)
        bully_prob = probs[0][1]
        print("Bullying probability:", bully_prob)

        threshold = 0.7
        if bully_prob >= threshold:
            status = f'Cyber Harassers ({round(bully_prob * 100, 2)}% confidence)'
        else:
            status = f'Non-Cyber Harassers ({round((1 - bully_prob) * 100, 2)}% confidence)'

        user = ''
        with open("session.txt", "r") as file:
            user = file.read().strip()

        db_connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='!@#$', database='cyber', charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO posts(sender,filename,msg,posttime,status) VALUES(%s,%s,%s,%s,%s)"
        db_cursor.execute(student_sql_query, (user, filename, msg, current_time, status))
        db_connection.commit()

        if db_cursor.rowcount == 1:
            context = {'data': 'Post details added'}
        else:
            context = {'data': 'Error in adding post details'}
        return render(request, 'SendPost.html', context)

def RunAlgorithm(request):
    global classifier, tfidf_vectorizer, label_count, corpus
    if request.method == 'POST':
        name = request.POST.get('t1', False)
        stop_words = set(stopwords.words('english'))
        corpus.clear()
        posts = []
        df = pd.read_csv('dataset.txt')
        print("Label distribution:", df['Text Label'].value_counts())
        X = df.iloc[:, :-1].values
        Y = df.iloc[:, -1].values

        for i in range(len(Y)):
            if Y[i] == 'Non-Bullying':
                Y[i] = 0
            else:
                Y[i] = 1

        for i in range(len(X)):
            line = str(X[i]).strip('\n').strip().lower()
            # line = re.sub(r'[^a-zA-Z\s]+', '', line)
            posts.append(line)

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(posts)
        corpus = vectorizer.get_feature_names()
        label_count = X.shape[1]
        Y = Y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        output = ""
        if name == "SVM Algorithm":
            param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 'scale']}
            cls = svm.SVC(kernel='rbf', random_state=2, probability=True, class_weight='balanced')
            grid = GridSearchCV(cls, param_grid, cv=5)
            grid.fit(X_train, y_train)
            cls = grid.best_estimator_
            prediction_data = cls.predict(X_test)
            print("Best parameters:", grid.best_params_)
            print("Classification Report:\n", classification_report(y_test, prediction_data))
            print("Confusion Matrix:\n", confusion_matrix(y_test, prediction_data))
            msg = cal_accuracy(y_test, prediction_data, 'SVM Accuracy')
            output = f'SVM Algorithm Output details<br/><br/>{msg}'
            classifier = cls
            tfidf_vectorizer = vectorizer
        elif name == "Naive Bayes":
            cls = MultinomialNB()
            cls.fit(X_train, y_train)
            prediction_probs = cls.predict_proba(X_test)
            predictions = (prediction_probs[:, 1] >= 0.5).astype(int)
            msg = cal_accuracy(y_test, predictions, 'Naive Bayes Accuracy')
            output = f'Naive Bayes Algorithm Output details<br/><br/>{msg}'
            classifier = cls
            tfidf_vectorizer = vectorizer
        elif name == "Random Forest":
            cls = RandomForestClassifier(n_estimators=100, random_state=2)
            cls.fit(X_train, y_train)
            prediction_data = cls.predict(X_test)
            msg = cal_accuracy(y_test, prediction_data, 'Random Forest Accuracy')
            output = f'Random Forest Algorithm Output details<br/><br/>{msg}'
            classifier = cls
            tfidf_vectorizer = vectorizer
        elif name == "Decision Tree":
            cls = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=2)
            cls.fit(X_train, y_train)
            prediction_data = cls.predict(X_test)
            msg = cal_accuracy(y_test, prediction_data, 'Decision Tree Accuracy')
            output = f'Decision Tree Algorithm Output details<br/><br/>{msg}'
            classifier = cls
            tfidf_vectorizer = vectorizer
        elif name == "KNearest Neighbors":
            cls = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
            cls.fit(X_train, y_train)
            prediction_data = cls.predict(X_test)
            msg = cal_accuracy(y_test, prediction_data, 'KNearest Neighbor Accuracy')
            output = f'KNearest Neighbor Algorithm Output details<br/><br/>{msg}'
            classifier = cls
            tfidf_vectorizer = vectorizer

        context = {'data': output}
        return render(request, 'RunAlgorithms.html', context)