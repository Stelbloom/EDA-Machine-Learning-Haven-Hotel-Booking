HotelHaven Booking Intelligence
**Turning hotel booking data into fewer empty rooms and smarter revenue decisions.**
A full end‑to‑end machine learning project to predict hotel booking cancellations and uncover the drivers behind guest behaviour.
This repository contains the code and analysis behind the “HotelHaven KPI Booking Intelligence” project, where I:
●	Explore 36,285 bookings and 16 key booking features
●	Build a cancellation prediction model (show vs. cancel/no‑show)
●	Analyse guest profiles, room/meal/segment mix, price sensitivity, and loyalty
●	Translate the model’s insights into concrete revenue and operations strategies
Recruiter view: this project demonstrates my ability to go from messy business question → clean data → robust model → clear, actionable story.
_________________________________________________________________
🔍 Project Overview
Business problem
Last‑minute cancellations and no‑shows create uncertainty in occupancy, staffing, and revenue. HotelHaven wants to move from reactive to proactive by estimating the likelihood that a booking will be cancelled before the guest arrives.
Objective
Use historical bookings to predict booking_status (Canceled vs. Not Canceled) and identify the drivers of cancellation risk across:
●	Guest profile (adults, children, repeat guests)
●	Stay pattern (weekend/weeknight mix, length of stay)
●	Commercial decisions (room type, meal plan, market segment)
●	Booking behaviour (lead time, price, special requests, loyalty flags)
Target audience  
●	Revenue Management & Pricing
●	Operations & Front Desk
●	Data & Analytics / DS Hiring Managers
_________________________________________________________________
🧾 Data & Features
The model uses 16 booking‑level fields, grouped as follows:
Core identifiers & counts
●	Booking_ID — unique booking identifier
●	number_of_adults — adults per booking
●	number_of_children — children per booking
●	number_of_weekend_nights — weekend nights in stay
●	number_of_week_nights — weekday nights in stay
Product & commercial attributes
●	type_of_meal — meal plan selected (Plan 1–3 / None / Complimentary)
●	room_type — room category booked (Types 1–5, Other)
●	car_parking_space — parking included (0 = No, 1 = Yes)
●	market_segment_type — booking channel (Online, Offline, Corporate, Aviation, etc.)
Behavioural & profile flags
●	lead_time — days from reservation to check‑in
●	repeated — repeat customer (0 = No, 1 = Yes)
●	P-C — customer profile flag
●	P-not-C — non‑customer profile flag
●	special_requests — number of special requests
Target & monetary
●	average_price — average nightly rate (in USD)
●	booking_status — target (Canceled / Not Canceled)
From the KPI analysis:
●	Total bookings: 36,285
●	Cancellation rate: ~29.2%
●	Avg lead time: 71 days
●	Avg price: $104/night
●	Repeat guest rate: 97%
_________________________________________________________________
📊 Key Insights (from EDA & KPIs)
Some highlights from the exploratory analysis and the KPI deck:
●	Guest profile
○	Typical booking: ~2.4 adults, 0.6 children, and 1.6 weekend + 2.5 weekday nights.
○	Only ~14% of bookings include children; family + kids behave differently in cancellation patterns.
●	Room, meal & segment mix
○	Room demand is concentrated in Room Types 1–3 (Type 1 being the workhorse).
○	Market segments skew towards Online and Offline channels, with Corporate and Aviation as smaller but important segments.
●	Cancellations
○	Online + Room Type 1 is the highest combined cancellation risk segment.
○	Summer months (Jun–Aug) have peak bookings but also the highest cancellation share.
○	Bookings with zero special requests cancel significantly more often than those with 3+ requests.
●	Price, requests & loyalty
○	Higher‑priced bookings show different cancellation behaviour by room type.
○	Repeat guests and positive P-C flags are associated with better retention.
These insights directly informed our feature engineering and business recommendations.
_________________________________________________________________
🧠 Modelling Approach
Problem framing
●	Type: Supervised binary classification  
●	Target: booking_status → Canceled (1) vs. Not Canceled (0)  
●	Primary metric: F1‑score (macro) + ROC‑AUC and confusion matrix for business interpretation
Pipeline
flowchart LR
  A[Raw Hotel Booking Data] --> B[Data Cleaning & Feature Engineering]
  B --> C[Train/Test Split]
  C --> D[Handle Class Imbalance (SMOTE vs. No SMOTE)]
  D --> E[Model Training\n(Logistic Regression, Random Forest, ...)]
  E --> F[Evaluation\n(ROC, Confusion Matrix, F1-macro)]
  F --> G[Feature Importance & KPI Deep-Dive]
  G --> H[Business Strategies & Recommendations]
Data preparation
●	Removed non‑informative IDs (Booking_ID) from features
●	Cleaned missing/invalid records
●	Encoded categorical features (room_type, type_of_meal, market_segment_type, etc.)
●	Standardised numerical features where appropriate
●	Addressed class imbalance with and without SMOTE to compare impact
●	Split into train/test sets with stratified sampling to preserve the cancellation rate
Models tested
Several models were trained and compared:
●	Logistic Regression
●	Gradient Boosting
●	Random Forest (final choice)
From the modelling comparison:
●	Random Forest provided the best balance of accuracy, stability, and interpretability.
●	ROC curve and confusion matrix showed strong separation and useful business trade‑offs.
●	Feature importance highlighted lead time, market segment, average_price, room_type, and special_requests as key drivers.
**Highlight:** Random Forest chosen as the final model for its robustness and ability to capture non‑linear interactions across booking attributes.
_________________________________________________________________
🧪 Example: Training Pipeline (Code Snippet)
The core training logic is wrapped in a scikit‑learn pipeline to keep preprocessing and modelling tightly coupled and reproducible:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Load data
bookings = pd.read_csv("data/hotel_bookings_clean.csv")

X = bookings.drop(columns=["booking_status", "Booking_ID"])
y = bookings["booking_status"].map({"Canceled": 1, "Not_Canceled": 0})

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", rf),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))
In the repo, this is extended with SMOTE experiments, cross‑validation, ROC/PR curves, and feature importance plots.
_________________________________________________________________
📁 Repository Structure
A suggested layout for this project (or similar structure in the repo):
.
├── data/
│   ├── hotel_bookings_raw.csv
│   └── hotel_bookings_clean.csv
├── notebooks/
│   ├── 01_eda_kpis.ipynb
│   ├── 02_modelling_baselines.ipynb
│   └── 03_random_forest_final.ipynb
├── src/
│   ├── data_prep.py
│   ├── train_model.py
│   └── evaluate.py
├── reports/
│   ├── HotelHaven_KPI_Analysis.pdf
│   └── figures/
│       ├── kpi_overview.png
│       ├── cancellation_by_segment.png
│       ├── roc_curve_models.png
│       └── feature_importance_top15.png
└── README.md
_________________________________________________________________
🖼 Visuals (for the GitHub page)
You can export key slides/plots from the notebook and reference them here to make the README visually stand out:
![HotelHaven KPI Overview](reports/figures/kpi_overview.png)
*Top‑line booking and cancellation KPIs across 36k+ reservations.*

![Cancellation by Market Segment & Room Type](reports/figures/cancellation_by_segment.png)
*Online + Room Type 1 emerges as the highest combined cancellation risk segment.*

![Model ROC Comparison](reports/figures/roc_curve_models.png)
*ROC curves for baseline vs. Random Forest. Random Forest delivers the best separation.*

![Top 15 Features](reports/figures/feature_importance_top15.png)
*Lead time, average_price, market_segment_type, and special_requests dominate feature importance.*
These image paths assume you’ve placed exported figures under reports/figures/.
_________________________________________________________________
🚀 How to Run the Project
●	Clone the repositorygit clone https://github.com/<your-username>/hotelhaven-booking-intelligence.git cd hotelhaven-booking-intelligence
●	Create and activate a virtual environment (optional but recommended)python -m venv .venv source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
●	Install dependenciespip install -r requirements.txt
●	Explore the analysis
○	Open notebooks/01_eda_kpis.ipynb to see the KPI and EDA work.
○	Open notebooks/02_modelling_baselines.ipynb to review baseline models.
○	Open notebooks/03_random_forest_final.ipynb for the final model and insights.
●	Train the model from scratch (if you provide a script)python src/train_model.py
●	Generate evaluation plots and metricspython src/evaluate.py
_________________________________________________________________
💡 Business Recommendations
(The following are distilled from the KPI and modelling insights; great for recruiters and stakeholders.)
●	Tiered Deposit Policy
Use lead_time to require non‑refundable deposits for bookings 60+ days in advance and for guests with a history of cancellations. This directly targets segments with the highest cancellation risk.
●	Personalised Pre‑Arrival Engagement
Guests with 0 special_requests show markedly higher cancellation rates. Trigger automated pre‑arrival outreach for these bookings (upsell room upgrades, meal plans, or experiences) to increase psychological commitment.
●	Loyalty‑Driven Retention
Leverage repeated and P-C flags to identify high‑value guests and reward them with exclusive benefits tied to non‑cancellable rates. This aligns loyalty perks with more predictable occupancy.
●	Segment‑Specific Pricing & Flexibility
Online market_segment_type shows the highest cancellation share. Introduce a flexible vs. non‑refundable tier structure specifically tuned to this segment, with pricing that compensates for added risk.
●	Ongoing Monitoring & Governance
Treat the model as a decision support tool, not an autopilot:
○	Monitor performance over time for drift as booking patterns change.
○	Regularly retrain using fresh data.
○	Embed clear ownership and review processes to avoid over‑reliance or unintended bias.
_"Every cancellation is a data point — and a prevention opportunity."_
_________________________________________________________________
🧰 Tech Stack
●	Language: Python
●	Analysis: Jupyter Notebooks, pandas, NumPy
●	Modelling: scikit‑learn (RandomForest, LogisticRegression, etc.)
●	Imbalance handling: SMOTE (imbalanced-learn) for comparison
●	Visualisation: Matplotlib, Seaborn (plus presentation slides)
●	Documentation: This README + KPI slide deck
_________________________________________________________________
📬 Contact
If you’d like to talk about this project or similar work:
●	Author: Ugwunwa Stella N
●	Focus: Data Science · Machine Learning · Applied Analytics
Feel free to open an issue, leave a comment, or reach out directly.
_________________________________________________________________
**Summary for recruiters:** This project is a realistic demonstration of how I approach applied machine learning: understanding the business problem, carefully preparing data, testing and comparing models, and—most importantly—translating results into decisions that reduce cancellations and protect revenue.
