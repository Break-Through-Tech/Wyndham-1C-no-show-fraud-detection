---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | The project uses a Python-centric tech stack, aligning with students' existing skills and resources. |
| Data Readiness | 🟡 | The dataset is estimated to be between 5 GB and 10 GB. While it is not over the 10GB threshold, the project's success will depend on the complexity and cleanliness of the synthetic data, which may require substantial preprocessing. |
| Resource Check | 🟢 | All required resources, including software and tools, are accessible via Google Colab, which is user-friendly for students and eliminates infrastructure concerns. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
The project addresses a relevant and timely issue in hospitality management—a strong starting point. To enhance student engagement, consider the following suggestions: 1. Simplify the SHAP integration for better understanding among students, and maybe provide additional resources or tutorials. 2. Sharing recommended cleaning and preprocessing of the synthetic data to avoid mismatched expectations. This would help students manage their workload more efficiently throughout the semester. Overall, I encourage you to clarify these gaps in your project outline to better empower students. These steps could help ensure a comprehensive learning experience for the Fellows and a successful project outcome for your team.

---

# No-Show Fraud Detection: Protecting Hotel Inventory from Loyalty Reservation Abuse

**Company / Org:** Wyndham Hotels & Resorts  
**Challenge Advisor:** Seema Yadav, [Email address]     
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Wyndham Hotels & Resorts
Wyndham Hotels & Resorts is a global leader in the hospitality industry, operating a vast portfolio of hotel brands across various price points and geographic regions. The team's objective is to safeguard inventory and maintain the integrity of their loyalty program by identifying and mitigating fraudulent reservation patterns.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use synthetic Wyndham-shaped loyalty data — including reservation history, no-show events, and points redemption timing — and supervised classification (XGBoost) with SHAP explainability to build a flexible model (not limited to rule-based criteria) that identifies members repeatedly booking hotel reservations with no intent to stay, in order to harvest first-night no-show loyalty points and redeem them quickly for value. This will help our company proactively address a fraud pattern that simultaneously ties up hotel inventory and drains loyalty point liability, where bad actors exploit a legitimate member benefit.

Prediction (simple) = Logistic Regression = baseline scoring
Prediction (advanced) = XGBoost = high-accuracy scoring
Explanation = SHAP = explains decisions

### Success Criteria

A model that catches a meaningful share of fraudulent accounts before redemption happens, at a false positive rate low enough that a real analyst queue would be workable.

A SHAP-powered dashboard that an analyst could sit down with on Day 1 and understand without additional training.

A final presentation where the fellows can honestly quantify the tradeoff — here's how much fraud we catch, here's the cost in false positives, here's what it would take to deploy this — rather than just reporting that the model performed well on a test set.

The Red Team exercise in Week 9 is also part of measuring success. Fellows that can clearly articulate what their model misses and why has understood the problem at a deeper level than one that can only describe what it catches.

Below are various milestones plotted for each week, but this is meant to be flexible and provide a guideline only. Some of the techniques / specifics are guidelines as well and can be flexible/discussed.

### Stretch Goals

The most natural first stretch is scoring at the time of booking rather than after the no-show occurs. The base project detects fraud after the fact — the no-show has already happened, the points have already posted. A meaningful extension is to ask: can the model score a reservation at the moment it's made and flag it as high-risk before the member ever no-shows? This shifts the features from historical no-show rates to leading indicators — advance booking window, property tier selected, account age, the number of other reservations booked on the same day. It's a harder problem and a much more valuable one operationally, because it gives the hotel a chance to act before inventory is lost.

The most ambitious stretch, if the fellows are ahead, is simulating model drift. We would generate a small synthetic "next quarter" dataset where fraudsters have slightly adapted (this is continuously happening in real life!) — they've noticed the model and started spacing their no-shows further apart, or targeting mid-tier properties instead of upscale ones. Run the existing model against that new data, measure the performance drop, and propose what retraining or feature update would recover it. That exercise teaches fellows something no textbook covers well: a fraud model is never finished, and the question of how to maintain one over time is as important as building it in the first place.

| Month | Week | Milestone | Key Activities |
|---|---|---|---|
| September | Week 1 | Scope Alignment & Data Setup | Confirm understanding and align to scope. Environment setup. Load all CSVs (data setup). Compute basic stats: row counts, null rates, fraud prevalence. Plot no-show rate distributions: fraud vs. legit. |
| September | Week 2 | Exploratory Data Analysis | Deep Exploratory Data Analysis (EDA). Surface the key behavioral differences between fraudsters and legitimate members. Confirm findings match the signal table. Compile EDA deck. |
| September | Week 3 | Feature Engineering Sprint 1 | Feature engineering sprint 1: no_show_rate, no_show_rate_l90d, noshow_points_ratio, avg_days_to_redemption. Document each feature with target signal. |
| September | Week 4 | Feature Engineering Sprint 2 | Feature engineering sprint 2: max_same_day_noshow_bookings, distinct_markets_booked, upscale_reservation_frac, avg_advance_booking_days. Complete feature store. |
| October | Week 5 | Baseline Modeling & Metric Selection | Baseline logistic regression. XGBoost with class weighting. Compare PR-AUC. Demonstrate why accuracy is the wrong metric. Threshold grid (10 values). |
| October | Week 6 | Tuning & Model Lock | XGBoost hyperparameter tuning. SMOTE experiment — compare to class weighting. Select operating threshold. Lock model version for explainability layer. |
| October | Week 7 | Explainability & Dashboard Scaffold | SHAP TreeExplainer integration. Generate force plots for 3 fraud and 3 legit members. Build dashboard scaffold with placeholder risk queue. |
| October | Week 8 | Dashboard Completion & Business Case | Dashboard completion: live SHAP explanations, risk tier display, filterable alert queue. Usability review with program manager. Business case draft. |
| November | Week 9 | Red Team Exercise & Presentation Build | Red Team exercise (full cohort, 2 hrs). Debrief writeup. Final presentation build. Executive summary draft reviewed by program manager. |
| November | Week 10 | Final Pitch & Retrospective | Presentation rehearsal. Final 20-minute stakeholder pitch. Q&A. Program retrospective. |

Meant to build in some flexibility for some tasks to take longer than one week or iterations on dashboard, project, etc.
> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

[CA to update here - "This is not 100% solidified yet, but we would ideally be providing reservation and redemption data. It would be shared via a secure Sharepoint site (encrypted).  We (Wyndham) may end up needing to generate synthetic Wyndham-shaped data instead of providing real data, depending on what our compliance team comes back with.]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification,Regression,Clustering,Recommendation Systems,Time Series Analysis  

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
