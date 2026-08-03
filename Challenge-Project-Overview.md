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
**Challenge Advisor:** Danielle Golinski, danielle.golinski@wyndham.com  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Wyndham Hotels & Resorts
Wyndham Hotels & Resorts is a global leader in the hospitality industry, operating a vast portfolio of hotel brands across various price points and geographic regions. The team's objective is to safeguard inventory and maintain the integrity of their loyalty program by identifying and mitigating fraudulent reservation patterns.

---

## 🎯 The Challenge
### Project Summary
Students will utilize synthetic loyalty data to build a supervised classification model that identifies fraudulent members who habitually book rooms without the intent to stay to exploit loyalty points. By employing XGBoost and SHAP explainability, the team will develop a system to flag high-risk accounts, effectively protecting hotel inventory and preventing the depletion of loyalty point liability.

### Success Criteria
A model that catches a meaningful share of fraudulent accounts before redemption at a workable false positive rate; a SHAP-powered dashboard for analyst use; and a final presentation quantifying catch rates, costs of false positives, and deployment requirements.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Conduct exploratory data analysis to identify no-show patterns and establish robust data cleaning pipelines for handling synthetic reservation logs. |
| **October** | Feature Engineering & Baseline Modeling | Develop core behavioral features from reservation history and train a baseline logistic regression model to benchmark performance against class imbalances. |
| **November** | Model Optimization & Evaluation | Apply SMOTE to address class imbalance, train an optimized XGBoost classifier, and integrate SHAP TreeExplainer for model interpretability. |
| **December** | Insights, Deliverables & Presentation | Develop an analyst-facing dashboard for risk monitoring and synthesize findings into a business case and final project presentation. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Wyndham Loyalty Reservation Dataset (Internal Synthetic Archive)  
**Format:** CSV, TSV, or Excel (.xlsx)  
**Size:** 5gb to 10gb  
**Location:** Secure internal repository provided by the challenge advisor.  

### Key Details
- Reservation and redemption data provided by the organization (potentially synthetic) in CSV/TSV or Excel (.xlsx) formats. Documentation including a data dictionary is available.
- Features include timestamps of bookings, no-show indicators, point accrual/redemption cycles, and member account tenure. Preprocessing must account for temporal drift and synthetic data noise.

---

## 🛠️ Suggested Approach
**ML Problem Type:** Classification  
**Recommended Libraries:**
- Logistic Regression
- XGBoost
- SHAP (TreeExplainer)
- SMOTE
- Python
- Google Colab
**Evaluation Metrics:** Precision-Recall AUC, False Positive Rate for high-risk flags, and overall classification accuracy on fraudulent member detection.

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- Industry standards on loyalty program fraud detection and hospitality revenue management.
**Technical Tutorials:**
- Documentation for XGBoost and SHAP explainability frameworks in Python.
**Code Examples:**
- Starter notebooks for supervised classification and imbalanced dataset handling (SMOTE).

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Email and internal messaging channels provided by the company.  
**Response time:** 48-72 business hours.  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
