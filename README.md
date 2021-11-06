<div align = "center">
  
## User-Response-Prediction-System
<h3> <i> iNeuron Internship Project</i></h3>
<h4>Project Teammates : Vivek Limbad, Manali Kadam, Rueben Patil, Siddhanth Ghag </h4>
  
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-ff2d55.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Lines of code](https://img.shields.io/tokei/lines/github/7Vivek/User-Response-Prediction-System?color=5856d6)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/7Vivek/User-Response-Prediction-System?color=ff9500)
![GitHub repo size](https://img.shields.io/github/repo-size/7Vivek/User-Response-Prediction-System?color=5ac8fa)
![GitHub last commit](https://img.shields.io/github/last-commit/7Vivek/User-Response-Prediction-System?color=4cd964)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/7Vivek/User-Response-Prediction-System?color=dd04fa)
![Relative date](https://img.shields.io/date/1635338107?color=1c80f6)
![GitHub top language](https://img.shields.io/github/languages/top/7Vivek/User-Response-Prediction-System?color=ffff66)
![GitHub language count](https://img.shields.io/github/languages/count/7Vivek/User-Response-Prediction-System?color=04e2b5)
</div>

<h2 align = "center"> 🎬 Project Demo </h2>

<h4>📑 <a href="https://github.com/7Vivek/User-Response-Prediction-System/blob/main/Documentation/User%20Response%20Prediction%20System%20using%20Machine%20Learning%20Techniques.pdf">Documentation</a></h4>
<h4>🎞 <a href="https://github.com/7Vivek/User-Response-Prediction-System/blob/main/Documentation/Wireframe%20Documentation.pdf">Wireframe</a></h4>
<h4>📂 <a href="https://github.com/7Vivek/User-Response-Prediction-System/tree/main/Dataset">Dataset</a></h4>
<h4> ✔ Application is hosted on Heroku. You can see the <a href="https://user-response-prediction.herokuapp.com/">Demo</a></h4>

https://user-images.githubusercontent.com/77670138/139003060-000ff0fd-9b7b-4cb4-aba9-201b8c20b436.mp4

<div align = "center">
  
[![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://github.com/7Vivek/User-Response-Prediction-System/tree/main/Python%20Code)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://github.com/7Vivek/User-Response-Prediction-System/tree/main/Model)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://github.com/7Vivek/User-Response-Prediction-System/tree/main/EDA)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://github.com/7Vivek/User-Response-Prediction-System/tree/main/EDA)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://github.com/7Vivek/User-Response-Prediction-System/tree/main/EDA)
[![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)](https://github.com/7Vivek/User-Response-Prediction-System/blob/main/templates/index.html)
[![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)](https://github.com/7Vivek/User-Response-Prediction-System/blob/main/static/css/style.css)
[![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)](https://github.com/7Vivek/User-Response-Prediction-System/blob/main/templates/index.html)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://github.com/7Vivek/User-Response-Prediction-System/blob/main/app.py)
[![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)](https://user-response-prediction.herokuapp.com/)
 </div>

<h2>🖱 User-Response-Prediction: What, why, and how ?</h2>

<img align = "right" src="https://miro.medium.com/max/960/1*hIPMAi6s0xF23Y8GWcPWWA.gif" style="width:300px;height:160px;"></img>    

- The online advertisement industry has become a multi-billion industry, and predicting ad **CTR** (click-through rate) is now central. Nowadays, different types of advertisers and search engines rely on modeling to predict ad CTR accurately.

- We will be predicting the ad click-through rate using the machine learning approach.

- **CTR**: It is the metric used to measure the percentage of impressions that resulted in a click.

- Paid search advertising is a popular form of **Pay per click (PPC)** advertising in which brands or advertisers pay (bid amount) to have their ads displayed when users search for specific keywords.

- How to calculate ad click prediction by performing machine learning modelling on a dataset. We have build a Logistic Regression model that would help us predict whether a user will click on an ad or not based on the features of that user. And hence calculate the probability of a user clicking on an ad.

- Using these probabilities, search engines could decide which ads to display by multiplying the possibilities with the bid amount and sorting it out.

<h2> ✒ Problem Statement</h2>

- Develop a machine learning algorithm that predicts if a particular user will click on an advertisement.

<h3> 📂 Dataset </h3>

- The data consists of **10** variables:

**'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', Timestamp' and 'Clicked on Ad'.**

- The primary variable we are interested in is ```'Clicked on Ad'```.

<i>This variable can have two possible outcomes: 0 and 1, where 0 refers to a user who didn't click the advertisement, while one refers to the scenario where a user clicks the ad.</i>

- We will see if we can use the other **9** variables to accurately predict the value **'Clicked on Ad'** variable. 

- We have also performed exploratory data analysis to see how **'Daily Time Spent on Site'** in combination with **'Ad Topic Line'** affects the user's decision to click on the ad.

<h2> 🗃 Machine Learning Models </h2>

#### Below are the machine learning models used for this project

### 📝 Experimental Results

| Algorithm                               | Accuracy |
| :-------------------------------------  | :------- | 
| `LogisticRegression`                    | 95.33% |
| `RandomForestClassifier`                | 96% |
| `XGBClassifier`                         | 95% |
| `Linear Support Vector Classification`  | 95.33% |
| `k Nearest Neighbors Classifier`        | 95% |

<h2> ✔ How to Install and Use</h2>

- Install Python. Download my repo and change to directory of repo.
- To install the required packages and libraries, run this command in the project directory after cloning the repository:

**On command prompt, run**
```
pip install -r requirements.txt
```
**Running the app**

Go to this app's directory and run
```
python app.py
```

<h2> 🧩 Contributors </h2>

[![LinkedIn](https://img.shields.io/badge/ManaliKadam-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manali-kadam-aa855b215)
<br>
[![LinkedIn](https://img.shields.io/badge/RuebenPatil-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rueben-patil-3b14b91b3)
<br>
[![LinkedIn](https://img.shields.io/badge/SiddhanthGhag-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/siddhanth-ghag-68b7511a9)
<br>
[![LinkedIn](https://img.shields.io/badge/VivekLimbad-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vivek-limbad-412667215/)
[![GitHub](https://img.shields.io/badge/VivekLimbad-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/7Vivek)




