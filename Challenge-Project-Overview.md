# No-Show Fraud Detection: Protecting Hotel Inventory from Loyalty Reservation Abuse

**Company / Org:** Wyndham Hotels & Resorts  
**Challenge Advisor:** Danielle Golinski, [danielle.golinski@wyndham.com](mailto:danielle.golinski@wyndham.com)  
**Challenge Advisor:** Seema Yadav, [seema.yadav@wyndham.com](mailto:seema.yadav@wyndham.com)   
**AI Studio Coach:** Deanna DiMonte, [deanna.dimonte@breakthroughtech.org](mailto:deanna.dimonte@breakthroughtech.org)   
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Wyndham Hotels & Resorts
Wyndham Hotels & Resorts is a global leader in the hospitality industry, operating a vast portfolio of hotel brands across various price points and geographic regions. 

---

## 🎯 The Challenge

### Project Summary
In this project, you will use synthetic Wyndham-shaped booking data — including reservation data, stay data, no-show events, and points redemption timing — and supervised classification (XGBoost or similar) with SHAP explainability (or similar) to build a flexible model that identifies members repeatedly booking hotel reservations with no intent to stay, in order to harvest first-night no-show loyalty points and redeem them quickly for value. This will help our company proactively address a fraud pattern that simultaneously ties up hotel inventory and drains loyalty point liability, where bad actors exploit a legitimate member benefit.

### Success Criteria

* A **model** that catches a meaningful share of fraudulent accounts before redemption happens, at a false positive rate low enough that a real analyst queue would be workable.

* A **dashboard** that an analyst could sit down with on Day 1 and understand without additional training.

* A **final presentation** where the fellows can honestly quantify the tradeoff — here's how much fraud we catch, here's the cost in false positives, here's what it would take to deploy this — rather than just reporting that the model performed well on a test set.

A **Red Team exercise** is also part of measuring success. Fellows that can clearly articulate what their model misses and why has understood the problem at a deeper level than one that can only describe what it catches.

### Project Milestones

Fellows should approach the project in phases so they can build a working model first, then improve performance and explainability. 

| Level | Fellow Task | Why It Helps |
|-------|-----------|----------------|
| 1. Baseline	| Logistic Regression or Random Forest | Easy benchmark before using XGBoost |
| 2. Better model	| XGBoost or Gradient Boosting | Handles non-linear behavioral patterns better |
| 3. Explainability	| SHAP summary plot and 3 to 5 example explanations | Turns model output into business-friendly insights |

Use the below phase gates to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

**Phase 1: Understand the Data and Business Problem**
- Review the data dictionary and understand what each field represents.
- Define the target variable clearly: whether a member or reservation pattern should be flagged as potentially fraudulent.
- Conduct exploratory data analysis to understand no-show frequency, booking timing, redemption behavior, cancellation behavior, member tenure, and repeat patterns.
 
**Phase 2: Basic Data Validation and Feature Preparation**
The dataset is synthetic and expected to be mostly clean, so heavy preprocessing is not required. Fellows should still perform basic checks:
- Confirm field types, especially dates, numeric loyalty fields, and categorical fields.
- Join tables where desired.
- Check for missing values, duplicate records, impossible dates, and extreme outliers.
- Review class imbalance between normal and suspicious activity.
- Create behavior-based features such as no-show rate, booking frequency, cancellation ratio, redemption timing, account tenure, and repeated booking patterns.
 
**Phase 3: Baseline Modeling**
- Start with a simple baseline model such as Logistic Regression or Random Forest.
- Use this baseline to understand whether the features have predictive signal.
- Evaluate using Precision, Recall, F1 Score, Precision-Recall AUC, and False Positive Rate.
 
**Phase 4: Model Improvement**
- Train an XGBoost classifier or similar tree-based model.
- Compare performance against the baseline model.
- If class imbalance is significant, test approaches such as class weights or SMOTE.
- Focus on practical tradeoffs: catching suspicious behavior while keeping false positives manageable for business review.
 
**Phase 5: Simplified SHAP Explainability**
SHAP should be used as an explanation layer after the model is working, not as the main technical challenge.
Fellows should focus on:
- A global SHAP feature importance chart showing the top drivers of high-risk predictions.
- A few local SHAP explanations for example flagged accounts or reservations.
- Plain-English interpretation of why the model flagged certain behavior.
- A short discussion of how analysts could use these explanations to review suspicious accounts.

**Phase 6: Red Team Exercise**
- Each team presents their model to a peer team whose job is to defeat it — asking "how would a fraudster adapt to avoid your detection?" This is a lesson in adversarial thinking, which is fundamental to real fraud ML work.
 
**Phase 7: Final Deliverables**
- A reproducible notebook or code workflow.
- A short model evaluation summary.
- A simple dashboard or visual summary for analysts to use.
- Final recommendations explaining possible fraud indicators, business impact, false positive tradeoffs, and what would be needed before any production use.


### Stretch Goals

The most natural first stretch is scoring at the time of booking rather than after the no-show occurs. The base project detects fraud after the fact — the no-show has already happened, the points have already posted. A meaningful extension is to ask: can the model score a reservation at the moment it's made and flag it as high-risk before the member ever no-shows? This shifts the features from historical no-show rates to leading indicators — advance booking window, property tier selected, account age, the number of other reservations booked on the same day. It's a harder problem and a much more valuable one operationally, because it gives the hotel a chance to act before inventory is lost.

The most ambitious stretch, if the fellows are ahead, is simulating model drift. We would generate a small synthetic "next month" dataset where fraudsters have slightly adapted (this is continuously happening in real life!) — they've noticed the model and started spacing their no-shows further apart, targeting different properties, or using different PII patterns to book. Run the existing model against that new data, measure the performance drop, and propose what retraining or feature update would recover it. That exercise teaches something no textbook covers well: a fraud model is never finished, and the question of how to maintain one over time is as important as building it in the first place.


> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

The dataset is synthetic and generated via ChatGPT using 5.6 Sol, using prompts from analysis performed on real fraudulent reservation use cases. 

| Location | File | Description |
|------------|------------|------------------------------------------------------------|
| [data_links.md](data/data_links.md) | reservations_month_2026.csv | Synthetic bookings made at Wyndham hotels |
| [data_links.md](data/data_links.md) | stays_month_2026.csv | Synthetic processed/completed stays sent by hotels to Wyndham for processing |
| [/data folder](data/) | data_dictionary.csv | Data dictionary outlining the description of each field in the files |

To open and work with the files, use Power BI, DuckDB, a database, or Python. In Excel, use Data > Get Data > From Text/CSV and load to the Data Model or create a connection instead of loading all records to a worksheet. Memory and Excel edition still matter at this scale.

Chunked Python example:

```python
import pandas as pd

for chunk in pd.read_csv("reservations_july_2026.csv", chunksize=250_000):
    # Profile or transform the chunk.
    pass
```

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, Regression, Clustering, Recommendation Systems, Time Series Analysis  

**Modeling:**
- Logistic Regression or Random Forest for baseline modeling
- XGBoost or Gradient Boosting for improved model performance
- Class weighting or SMOTE for handling class imbalance, if needed

**Recommended Libraries & Tools:**
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib or Seaborn
- Google Colab
- For Dashboard creation: Streamlit (risk queue UI)

**Evaluation Metrics:**
- Precision
- Recall
- F1 Score
- Precision-Recall AUC
- False Positive Rate
- Confusion Matrix

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Business and Problem Context**
- Hotel Loyalty Programs may offer point earnings for the first night of legitimate no-show stays, given the hotel is typically able to charge the member's credit card for the first night of the stay.
- Hotel stays cancelled out of policy (i.e. after 4pm the day before check-in) also earn points, given the hotel is typically able to charge the member's credit card for the stay.
- The Problem: if the hotel company's website and app do not validate credit cards upon _booking_, fake credit card numbers can be used to secure the inventory.
- In the event of a fraudulently booked no-show or late cancellation, the hotels are instructed to "zero out" the room revenue field on the reservation before the central reservation systems automatically pick it up and send it in for processing, but the hotels have trouble keeping up. The stay then processes with revenue on it and the member earns the points.
- Bad actors book dozens of stays, save up the points, and then quickly redeem them for Free Night Stays at local hotels. They then _sell_ the night on 3rd party booking websites and collect the revenue for the stay.
- Loyalty fraud and abuse overall can include repeated booking behavior, points exploitation, account misuse, and suspicious redemption patterns.
- Fellows should focus on identifying behavioral signals that may indicate no-show or cancellation abuse, while recognizing that model outputs should support analyst review (human in the loop!) rather than automatically label customers as fraudulent.
  

**Background Reading:**
- https://lawstreet.co/vantage-points/japanese-mother-son-duo-nabbed-in-kyoto
- https://www.linkedin.com/pulse/fraud-online-travel-agencies-2026-major-types-impact-how-p-giqfc/
- https://onix-systems.com/blog/online-travel-fraud-prevention

**Technical Tutorials:**
- Microsoft Fabric fraud detection tutorial for an end-to-end fraud detection workflow using data preparation, model training, evaluation, and scoring.
- SHAP documentation for explaining machine learning model outputs.
- Introductory SHAP tutorials that show feature impact and local explanations.
- XGBoost documentation or beginner tutorials for tree-based classification.
- Scikit-learn documentation for classification metrics and train/test validation.


*Feel free to explore beyond these, and share anything interesting you find with us!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to us with questions:** 
* **Email (preferred):** Please copy your teammates and AI Studio Coach! Include "BTT" in the subject line so we can spot your email easily.
* **Discord:** We will check in on Discord throughout the week, but Email communication is preferred.
* **Response times:** We will aim to respond within 48 hours on weekdays. Please reach out to your AI Studio Coach with urgent questions.

**Recommended tools**
* **Coding:** Google Colab
* **Collaboration:** GitHub, Notion
* **Virtual Meetings:** Zoom, Google Meet

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

We're excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
