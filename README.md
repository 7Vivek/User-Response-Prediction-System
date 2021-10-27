<div align = "center">
  
## User-Response-Prediction-System
<h3> <i> iNeuron Internship Project</i></h3>
<h4>Project Teammates : Vivek Limbad, Manali Kadam, Rueben Patil, Siddhanth Ghag </h4>
</div>
<br>
<h2 align = "center"> 🎬 Project Demo </h2>

<h4>📑 <a href="https://github.com/7Vivek/User-Response-Prediction-System/blob/main/User%20Response%20Prediction%20System%20using%20Machine%20Learning%20Techniques.pdf">Documentation</a></h4>
<h4>🎞 <a href="https://github.com/7Vivek/User-Response-Prediction-System/blob/main/Wireframe%20Documentation.pdf">Wireframe</a></h4>
<h4>📂 <a href="https://github.com/7Vivek/User-Response-Prediction-System/tree/main/Dataset">Dataset</a></h4>
<h4> ✔ Application is hosted on Heroku. You can see the <a href="https://user-response-prediction.herokuapp.com/">Demo</a></h4>

https://user-images.githubusercontent.com/77670138/139003060-000ff0fd-9b7b-4cb4-aba9-201b8c20b436.mp4

<br>
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

<h2> 🧩 Contributors </h2>

- <h4><a href="https://github.com/7Vivek">Vivek Limbad</a></h4>
- <h4><a href="https://www.linkedin.com/in/manali-kadam-aa855b215">Manali Kadam</a></h4>
- <h4><a href="https://www.linkedin.com/in/rueben-patil-3b14b91b3">Rueben Patil</a></h4>
- <h4><a href="https://www.linkedin.com/in/siddhanth-ghag-68b7511a9">Siddhanth Ghag</a></h4>



