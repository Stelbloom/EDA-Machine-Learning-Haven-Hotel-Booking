Predictive-Model-for-HotelHaven-Booking-Status-and-Cancellation-Risk 
Turning hotel booking data into fewer empty rooms and smarter revenue decisions.
A full end‑to‑end machine learning project to predict hotel booking cancellations and uncover the drivers behind guest behaviour.

●	Table of Contents
● Project Overview
● Business Problem
● Project Objectives
● Data Source
● Key Features
● Tools and Libraries
● Data Preprocessing
● Exploratory Data Anaysis
● Machine Learning Model
● Evaluation Metrics
● Key Insights
● Recommendations
● Conclusion

This repository contains the code and analysis behind the “HotelHaven KPI Booking Intelligence” project, where I:
●	Explore 36,285 bookings and 16 key booking features

●	Build a cancellation prediction model (show vs. cancel/no‑show)

●	Analyse guest profiles, room/meal/segment mix, price sensitivity, and loyalty

●	Translate the model’s insights into concrete revenue and operations strategies

Project Objectives: this project aimed at producing insights from messy business question → clean data → robust model → clear, actionable story.
_________________________________________________________________
🔍 Project Overview
Business problem

Hotel Haven, a luxury hotel chain, has been losing significant revenue to booking cancellations. This project analyzes 36,285 booking records to uncover why customers cancel, identify the key factors driving cancellation behavior, and build a machine learning model to predict booking status, enabling Hotel Haven to move from reactive cancellation management to proactive prevention.Last‑minute cancellations and no‑shows create uncertainty in occupancy, staffing, and revenue. HotelHaven wants to move from reactive to proactive by estimating the likelihood that a booking will be cancelled before the guest arrives.

Business Problem
Hotel Haven struggled with frequent booking cancellations. Their existing system could not provide insights as to why clients cancel and the behaviour behind cancellation, which has impacted their ability to allocate resources, retain customers and improve profitability. Beyond lost room revenue, late cancellations inflate operational costs through unproductive staff hours and wasted resources, a direct consequence of the perishable nature of hotel inventory. They needed a system that did not only predictive cancellations but also explain the behaviour behind cancellations.

Project Objectives
The primary tasks of this project is to clean and prepare the booking dataset for analysis and modeling, explore booking patterns and cancellation behavior, identify key factors influencing booking cancellations, engineer meaningful features to enhance analysis and model performance, build a machine learning model to predict booking status, evaluate and fine-tune model performance, generate actionable recommendations to reduce cancellations.

Use historical bookings to predict booking_status (Canceled vs. Not Canceled) and identify the drivers of cancellation risk across:

The dataset used was provided by 10Alytics teamns and contains 36,285 rows of hotel booking records collected over a 3 years, 11 months period.

●	Guest profile (adults, children, repeat guests)

●	Stay pattern (weekend/weeknight mix, length of stay)

●	Commercial decisions (room type, meal plan, market segment)

●	Booking behaviour (lead time, price, special requests, loyalty flags)

Target audience  
●	Revenue Management & Pricing

●	Operations & Front Desk

●	Data & Analytics / DS Hiring Managers


Key Features
The key variables captured in this were 16 features, including the Lead time, average price, special requests, room type, market segment type, number of nights, date of reservation and the target variable (booking status).

● Tools and Libraries
● Python (Jupyter Notebook)
● Pandas
● Matplotlib
● Seaborn
● Scikit-Learn
● XGBoost
● Imbalanced-Learn

Data Preprocessing
Following business and data understanding, the data was cleaned during which invalidate date was corrected, data types were fixed and dropped irrelevant dropped. Column engineering was implored to generating the total nights, total revenue, price range, and lead time range. Categorical variables were encoded. RobustScaler was applied to handle skewed numerical features. Class Imbalance (67:33 class distribution) was identified and handled using SMOTE. A final cross validation was done before training.

Exploratory Data Analysis
A comprehensive analysis, both univariate and bivariate was done to uncover cancellation patterns.

Machine Learning Model
The hotel cancellation predictive model was built using a supervised machine learning approach. The class sets were split into 80:20 for training and testing. Multiple classification algorithms were experimented with, including but not limited to

● Logistic Regression
● Random Forest Classifier
● Gradient Boosting
● XGBoost
● AdaBoost
● Support Vector Machine
● K-Nearest Neighbors
● Decision Tree The best performing model was selected and underwent hyperparameter tuning to improve its performance
● Evaluation Metrics

The following metrics were used to assess the performance of the model:

Accuracy: The overall proportion of correctly predicted booking status (both cancelled and not-cancelled)
Precision: The proportion of correctly identified cancelled bookings among all bookings classified as cancelled (false alarm rate)
Recall: The proportion of correctly identified cancellations among all actual cancelled bookings (correctly flagged cancellation)
F1-score: The harmonic mean of precision and recall, providing a balanced metric for model evaluation
ROC-AUC score: The ability of the model to distinguish between class
Key Insights
1 in 3 bookings made at Hotel Haven was cancelled
The hotel recorded a 32.8% cancellation rate over a 3 year, 11 month period, resulting in approximately 4.2 million dollars in lost revenue.
Lead time was the strongest predictor of cancellation; the longer the booking window, the higher the cancellation risk. Guests who booked <= 30 days before arrival were more likely to follow through with their reservation.
Special requests was a reliable signal of booking commitment. Guests with zero requests cancel at 43%. Each additional speacial request made at booking significantly reduced the risk of cancellation.
Online bookings had the highest cancellation rate (~36%), while corporate bookings cancel at just ~10%. No cancellations recorded for the complementary market segments.
Pricing was a weak predictor of cancellations. The 101-$200 price range accounts for the highest volume of bookings and cancellations at Hotel Haven. Hotel Haven do not attract premium customers.
Though slight variations occured across group, Room Type 6 had the highest cancellation rate (42%) while Room Type 7 is the most reliable (23%).
Random forest was the best performing model demonstrating a strong predictive performance, with 90% accuracy, 90% F1 score, and a 96% ROC-AUC score, indicating that it can reliably identify booking cancellations and support better operational decision-making and improve profitability.
Recommendations
Use deposits, prepaid rates, or credit card to increase booking commitment
Offer flexible options such as rescheduling or partial refunds instead of full cancellation for lower priced rooms, giving budget guests a financial incentive to commit at the point of booking
Room upgrades or loyalty reward to guests who book directly (offline market segment)
Identify long lead-time bookings at an early stage and proactively engage these guests with personalized campaigns, promotional upgrades and experience packages
Offer small incentives for guests who keep long-lead bookings
Improve listing accuracy by ensuring room descriptions, photos, amenities, and prices match the actual experience
Actively grow the corporate segment by proactive contracts negotiation and corporate rate agreements with companies, and travel agencies, as their accommodation partner for corporate travellers
Incorporate more options in special-request fields
Conclusion
While they are not the direct causes of cancellations, lead time, special requests and pricing are the strongest predictors of cancellations. This project clearly demonstrates that by leveraging machine learning models, Hotel Haven can make proactive, data-driven decisions that move the business from reacting to cancellations to preventing them, ultimately reducing cancellation rates, improving profitability, and strengthening overall business performance.

_________________________________________________________________

