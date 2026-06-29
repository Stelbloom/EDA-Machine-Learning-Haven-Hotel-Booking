# Predictive Model for HotelHaven Booking Status and Cancellation Risk

> **Turning hotel booking data into fewer empty rooms and smarter revenue decisions.**

A full end‑to‑end machine learning project to predict hotel booking cancellations and uncover the drivers behind guest behaviour.

This repository contains the code and analysis behind the **“HotelHaven KPI Booking Intelligence”** project, where I:

- Explore **36,285 bookings** and **16 key booking features**
- Build a **cancellation prediction model** (show vs. cancel/no‑show)
- Analyse **guest profiles, room/meal/segment mix, price sensitivity, and loyalty**
- Translate the model’s insights into **concrete revenue and operations strategies**

This project showcases the full journey from **messy business question → clean data → robust model → clear, actionable story.**

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Project Objectives](#project-objectives)
4. [Data Source](#data-source)
5. [Key Features](#key-features)
6. [Tools and Libraries](#tools-and-libraries)
7. [Data Preprocessing](#data-preprocessing)
8. [Exploratory Data Analysis](#exploratory-data-analysis)
9. [Machine Learning Model](#machine-learning-model)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Key Insights](#key-insights)
12. [Recommendations](#recommendations)
13. [Conclusion](#conclusion)

---

## 🔍 Project Overview

**Project title:** Predictive Model for HotelHaven Booking Status and Cancellation Risk  
**Context:** Hotel Haven, a luxury hotel chain, has been losing significant revenue to booking cancellations and no‑shows.  
**Goal:** Use historical booking data to predict **booking status** and understand **why guests cancel**.

This project analyzes **36,285 booking records** to:

- Diagnose **why customers cancel**
- Identify **who cancels and under what conditions**
- Build a machine learning model that predicts **cancellation risk** before arrival
- Provide **practical recommendations** for revenue management and operations

Hotel Haven wants to move from **reactive** cancellation management to **proactive prevention**.

---

## 🧩 Business Problem

Hotel Haven struggled with frequent booking cancellations and no‑shows:

- The existing system could not explain *why* clients cancel or which guests are most at risk.
- Cancellations reduced room revenue and created **operational inefficiencies**:
  - Unproductive staff hours
  - Wasted housekeeping and F&B preparation
  - Difficulty in forecasting occupancy and staffing
- Hotel inventory is **perishable** – an empty room tonight can never be sold again.

**They needed a system that not only predicts cancellations but also explains the behaviour behind them.**

---

## 🎯 Project Objectives

The primary objectives of this project were to:

1. **Clean and prepare** the booking dataset for analysis and modelling.
2. **Explore booking patterns** and cancellation behaviour over time and segments.
3. **Identify key drivers** of booking cancellations (guest, stay, commercial, behavioural).
4. **Engineer meaningful features** (e.g. total nights, total revenue, lead time ranges) to improve interpretability and model performance.
5. **Build a supervised ML model** to predict `booking_status` (Canceled vs. Not Canceled).
6. **Evaluate and fine‑tune** model performance using appropriate classification metrics.
7. **Generate actionable recommendations** for revenue management and operations.

The model focuses on drivers across:

- **Guest profile:** adults, children, repeat guests
- **Stay pattern:** weekend/weeknight mix, length of stay
- **Commercial decisions:** room type, meal plan, market segment
- **Booking behaviour:** lead time, price, special requests, loyalty/profile flags

**Target audience:**

- Revenue Management & Pricing teams  
- Operations & Front Desk management  
- Data & Analytics / DS Hiring Managers

---

## 📂 Data Source

The dataset was provided by the **10Alytics** team and contains:

- **36,285 rows** of hotel booking records
- Collected over a **3 years, 11 months** period

Each row represents a single booking, with information about **guest composition, stay characteristics, commercial terms, and booking outcome**.

---

## 🔑 Key Features

The model uses **16 key booking‑level fields**, including:

- `Booking_ID` – unique booking identifier (removed from features for modelling)
- `number_of_adults` – number of adults per booking
- `number_of_children` – number of children per booking
- `number_of_weekend_nights` – weekend nights
- `number_of_week_nights` – weekday nights
- `type_of_meal` – meal plan selected (Plan 1–3 / None / Complimentary)
- `room_type` – room category booked
- `car_parking_space` – parking included (0 = No, 1 = Yes)
- `market_segment_type` – booking channel (Online, Offline, Corporate, Aviation, etc.)
- `lead_time` – days from booking to check‑in
- `repeated` – repeat customer flag
- `P-C`, `P-not-C` – customer profile flags
- `special_requests` – number of special requests at booking
- `average_price` – average nightly rate ($)
- `booking_status` – **target variable** (Canceled vs. Not Canceled)

Additional engineered features (see below) extend this base set.

---

## 🛠 Tools and Libraries

- **Python** (Jupyter Notebook)
- **pandas**, **NumPy** – data manipulation and feature engineering
- **Matplotlib**, **Seaborn** – exploratory data analysis and visualisation
- **scikit‑learn** – preprocessing, model training, evaluation
- **XGBoost**, **Gradient Boosting**, **AdaBoost** – ensemble models for experimentation
- **imbalanced‑learn (SMOTE)** – handle class imbalance

---

## 🧹 Data Preprocessing

_Subtitle: From raw bookings to clean, structured training data_

Key preprocessing steps:

1. **Data cleaning**
   - Corrected invalid dates and data types
   - Dropped irrelevant or redundant columns (e.g. IDs not useful for prediction)
   - Treated missing values appropriately

2. **Feature engineering**
   - `total_nights` = `number_of_weekend_nights` + `number_of_week_nights`
   - `total_revenue` = `average_price` × `total_nights`
   - `price_range` buckets (e.g. `0–100`, `101–200`, `>200`)
   - `lead_time_range` buckets (e.g. `<= 30 days`, `31–60`, `61–120`, `>120`)

3. **Categorical encoding**
   - One‑hot encoding for `room_type`, `type_of_meal`, `market_segment_type`, and other categorical features.

4. **Scaling**
   - Applied **RobustScaler** to skewed numerical features (`lead_time`, `average_price`, `total_nights`, `total_revenue`).

5. **Class imbalance handling**
   - Observed a **67:33** class distribution (Not Canceled : Canceled).
   - Applied **SMOTE** to oversample the minority class and balance the dataset.

6. **Train‑test split & cross‑validation**
   - Split data into **80% training / 20% testing**.
   - Used **cross‑validation** on the training set before final model selection.

---

## 📊 Exploratory Data Analysis

_Subtitle: Understanding who cancels, when, and why_

A comprehensive **univariate and bivariate** EDA was conducted to uncover cancellation patterns:

- Distribution of `booking_status` (Canceled vs. Not Canceled)
- Cancellation rates by:
  - **Lead time** buckets
  - **Special requests** count
  - **Room type** and **market segment**
  - **Price range**
  - Seasonality (month, quarter, year)
- Guest composition patterns (adults, children) and their relation to cancellations

**Example visuals for the README:**

```markdown
![HotelHaven KPI Overview](reports/figures/kpi_overview.png)
*Top‑line booking and cancellation KPIs across 36k+ reservations.*

![Cancellation by Market Segment & Room Type](reports/figures/cancellation_by_segment.png)
*Online + Room Type 1 emerges as the highest combined cancellation risk segment.*

![Top 15 Features](reports/figures/feature_importance_top15.png)
*Lead time, average_price, market_segment_type, and special_requests dominate feature importance.*
```

---

## 🤖 Machine Learning Model

_Subtitle: Predicting who will cancel before they arrive_

This is a **supervised classification** problem with:

- **Target:** `booking_status` (Canceled vs. Not Canceled)
- **Primary evaluation focus:** ability to correctly flag cancellations while controlling false alarms

### Algorithms explored

I experimented with multiple classifiers:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting
- XGBoost
- AdaBoost
- Support Vector Machine (SVM)
- K‑Nearest Neighbours (KNN)
- Decision Tree

After comparison and tuning, **Random Forest** emerged as the **best‑performing model**, balancing accuracy, robustness, and interpretability.


A `scikit‑learn` pipeline with `ColumnTransformer` was used to combine preprocessing and model training in a clean, reproducible way.

---

## 📏 Evaluation Metrics

To fully understand model performance, I used:

- **Accuracy** – overall proportion of correctly predicted bookings (canceled + not canceled).
- **Precision (Canceled class)** – among all bookings predicted as canceled, how many truly canceled? (Controls false alarms.)
- **Recall (Canceled class)** – among all truly canceled bookings, how many were correctly flagged? (Critical for risk reduction.)
- **F1‑score** – harmonic mean of precision and recall; balances the trade‑off between missing cancellations and over‑flagging.
- **ROC‑AUC** – ability of the model to distinguish between canceled and not canceled bookings across thresholds.

**Final model (Random Forest) performance:**

- ~**90% Accuracy**
- ~**90% F1‑score** (cancellation class)
- ~**0.96 ROC‑AUC**

This indicates the model can **reliably identify high‑risk bookings** and support better operational decision‑making.

---

## 🔑 Key Insights

From the combined EDA and model interpretation:

1. **High cancellation rate and revenue impact**  
   - Roughly **1 in 3 bookings** at Hotel Haven was canceled.  
   - The hotel recorded a **32.8% cancellation rate** over **3 years, 11 months**, representing an estimated **$4.2M in lost revenue**.

2. **Lead time is the strongest predictor of cancellation**  
   - The **longer the lead time**, the higher the probability of cancellation.  
   - Guests booking **≤ 30 days before arrival** are significantly more likely to follow through.

3. **Special requests signal commitment**  
   - Guests with **zero special requests** cancel at around **43%**.  
   - Each additional special request at booking **reduces cancellation risk**, indicating higher engagement and intent.

4. **Market segment behaviour**  
   - **Online bookings** exhibit the **highest cancellation rate (~36%)**.  
   - **Corporate bookings** cancel at only ~**10%**, making them highly reliable.  
   - **Complementary segment** (complimentary stays) recorded **no cancellations**.

5. **Price is a weak predictor of cancellations**  
   - The **$101–$200** price range accounts for the **highest volume of bookings and cancellations**, reflecting Hotel Haven’s main positioning.  
   - Hotel Haven does **not attract many premium customers**, and price alone does not explain cancellation behaviour.

6. **Room type patterns**  
   - Room Type 6 has the **highest cancellation rate (~42%)**.  
   - Room Type 7 is the **most reliable (~23% cancellation rate)**.

7. **Model explainability**  
   - Feature importance ranked **lead_time, special_requests, market_segment_type, room_type, and average_price** as the most influential drivers of cancellation risk.

---

## 💡 Recommendations

Based on the insights and model outputs, the project proposes several actions:

1. **Strengthen booking commitment for long lead times**  
   - Use **deposits, prepaid rates, or credit card guarantees** for bookings made far in advance (e.g. > 60 days).  
   - Prioritise these controls on segments/room types with historically high cancellation rates.

2. **Offer flexible alternatives instead of outright cancellations**  
   - Provide options like **rescheduling, partial refunds, or credits** for lower‑priced rooms, especially for price‑sensitive segments.  
   - This keeps guests in the funnel and reduces lost revenue.

3. **Reward reliable, direct channels and segments**  
   - Offer **room upgrades or loyalty rewards** for guests booking directly (offline / direct channels).  
   - Actively grow the **corporate segment** via contracts and corporate rates: these guests show very low cancellation rates.

4. **Target high‑risk, long‑lead bookings with proactive engagement**  
   - Use the model’s risk scores to identify **high‑risk bookings early**.  
   - Engage these guests with **personalised campaigns, promotional upgrades, and experience packages** to keep them committed.

5. **Leverage special requests as a loyalty lever**  
   - Encourage guests to share preferences during booking by **offering more special‑request options**.  
   - Use this data to personalise stays while also benefiting from the associated lower cancellation risk.

6. **Improve listing accuracy and expectation management**  
   - Ensure **room descriptions, photos, amenities, and prices** accurately reflect the real experience.  
   - Misaligned expectations can drive cancellations, especially for online bookings.

7. **Strategic pricing and inventory for risky room types**  
   - Re‑evaluate **Room Type 6** (high cancellations) and **Room Type 7** (more reliable) in terms of pricing, placement, and restrictions.  
   - Consider minimum‑stay requirements or stricter terms for risky types.

---

## ✅ Conclusion

While they are **not direct causes**, variables such as **lead time**, **special requests**, **market segment**, and **room type** are powerful **predictors** of cancellation behaviour.

This project demonstrates that by leveraging **machine learning on historical bookings**, Hotel Haven can:

- Move from **reacting** to cancellations to **preventing** them
- Reduce cancellation rates and protect millions in potential revenue
- Improve **operational planning** (staffing, inventory, F&B)
- Strengthen overall **guest satisfaction and business performance**

For recruiters and stakeholders, this project is a concrete example of how I:

- Understand a **business problem** in a commercial context
- Turn raw data into a **clean, model‑ready dataset**
- Evaluate and compare multiple **machine learning algorithms**
- Communicate **insights and recommendations** in a way that supports real decisions.
