<div align = "center">
  
# User-Response-Prediction-System
  
<h3> <i> iNeuron Internship Project</i></h3>
  
<h3>Project Teammates : Vivek Limbad, Manali Kadam, Rueben Patil, Siddhanth Ghag </h3>
  
</div>

## 🎬 Project Demo

<h4 align = "center">Application is hosted on Heroku. You can see the <a href="https://user-response-prediction.herokuapp.com/">Demo</a></h4>

https://user-images.githubusercontent.com/77670138/139003060-000ff0fd-9b7b-4cb4-aba9-201b8c20b436.mp4


## User-Response-Prediction: What, why, and how ?

<img align = "right" src="https://miro.medium.com/max/960/1*hIPMAi6s0xF23Y8GWcPWWA.gif" style="width:300px;height:160px;"></img>    

- The online advertisement industry has become a multi-billion industry, and predicting ad **CTR** (click-through rate) is now central. Nowadays, different types of advertisers and search engines rely on modelling to predict ad CTR accurately.

- We will be predicting the ad click-through rate using the machine learning approach. Before that, let us first understand few essential concepts and a general practice followed by search engines to decide which ads to display.

- **CTR**: It is the metric used to measure the percentage of impressions that resulted in a click.

<img src = "https://user-images.githubusercontent.com/77670138/138988506-56d6da1a-0fa3-4a4b-9b51-1738c27d9613.jpg" style="width:425px;height:80px;"></img>   

- ```Search ads```: Advertisements that get displayed when a user searches for a particular keyword. 

- Paid search advertising is a popular form of **Pay per click (PPC)** advertising in which brands or advertisers pay (bid amount) to have their ads displayed when users search for specific keywords.

- Relevance of Predicting CTR through a real-life example:

<i>Typically, the primary source of income for search engines like Google is through advertisement. Many companies pay these search engines to display their ads when a user searches for a particular keyword. Our focus is on search ads and CTR, i.e. the amount is paid only when a user clicks on the link and redirects to the brand’s website.</i>

- Different advertisers approach these search engines with their ads and the bidding amount to display their ads. The main objective of these search engines is to maximize their revenue. So the question is, how does a search engine decide which ads to display when a user searches for a particular keyword?

- Till now, we have seen what ad click prediction is and why is it important. Let us now explore how to calculate ad click prediction by performing machine learning modelling on a dataset. We will build a Logistic Regression model that would help us predict whether a user will click on an ad or not based on the features of that user. And hence calculate the probability of a user clicking on an ad.

- Using these probabilities, search engines could decide which ads to display by multiplying the possibilities with the bid amount and sorting it out.

## ✒ Problem Statement

- In this project, we will work with the advertising data of a marketing agency to develop a machine learning algorithm that predicts if a particular user will click on an advertisement.

- The data consists of **10** variables:

**'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', Timestamp' and 'Clicked on Ad'.**

- The primary variable we are interested in is ```' Clicked on Ad'```.

This variable can have two possible outcomes: 0 and 1, where 0 refers to a user who didn't click the advertisement, while one refers to the scenario where a user clicks the ad.

- We will see if we can use the other **9** variables to accurately predict the value **'Clicked on Ad'** variable. 

- We will also perform some exploratory data analysis to see how **'Daily Time Spent on Site'** in combination with **'Ad Topic Line'** affects the user's decision to click on the ad.

## 📂 Dataset

This project's goals are to deeply explore data to do with advertising, perform quantitive analysis, and achieve predictions from the data using machine learning techniques.

- This data set contains the following features:

- ```Daily Time Spent on Site```: consumer time on-site in minutes

- ```Age```: customer age in years

- ```Area Income```: Avg. Income of geographical area of consumer

- ```Daily Internet Usage```: Avg. minutes a day consumer is on the internet

- ```Ad Topic Line```: Headline of the advertisement

- ```City```: City of consumer

- ```Male```: Whether or not the consumer was male

- ```Country```: Country of consumer

- ```Timestamp```: Time at which consumer clicked on Ad or closed window

- ```Clicked on Ad```: 0 or 1 indicated clicking on Ad
