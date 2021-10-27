import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Daily Time Spent on Site':30, 'Age':45, 'Area Income':60000,'Daily Internet Usage':300,'Male':1})

print(r.json())