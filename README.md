# Challenge Packaging Industries — Business Intelligence System
> End-to-end data analysis for a real carton box manufacturing business in Chennai — SQL, 6 ML models, and a 7-page Power BI dashboard built on live business data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Power BI](https://img.shields.io/badge/Power%20BI-DAX-F2C811?style=flat-square&logo=powerbi)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?style=flat-square&logo=mysql)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat-square&logo=scikit-learn)

---

## Overview

This is a full-stack business intelligence project built for **Challenge Packaging Industries (CPI)**, a carton box manufacturing company based in Chennai, India. The data source is the company's real financial daybook — not a sample dataset.

The project covers the complete analyst pipeline: raw Excel ingestion → data cleaning → SQL analysis → 6 machine learning models → a 7-page Power BI dashboard with DAX measures — designed to give a small manufacturing business the kind of insight usually reserved for enterprises with dedicated data teams.

---

## Business Context

CPI is a small-to-mid-scale manufacturer serving a customer base of businesses that order custom carton boxes. The core business challenges this project addresses:

- Which customers are most valuable, and which are drifting away?
- Which orders carry the best margins, and which are quietly losing money?
- Which customers are likely to delay or default on payment?
- Are there anomalies in billing or sales that signal a problem?
- What does revenue look like over the next quarter?

---

## Data Source

| File | Description |
|---|---|
| `25-26 Daybook.xls` | Raw financial daybook — real transaction records from CPI's operations |

The daybook contains invoice-level records: customer names, order amounts, payment dates, product types, and billing details. All cleaning, transformation, and feature engineering was performed in Python before modelling.

---

## Project Structure

```
challenge-packaging-analysis/
│
├── data/                          # Raw and cleaned data files
├── ml/                            # ML model notebooks and outputs
├── outputs/                       # Exported CSVs and model result files
├── powerbi/                       # Power BI report file (.pbix)
├── sql/                           # SQL analysis queries
│
├── 25-26 Daybook.xls              # Source data (real business daybook)
├── CPI_ML_Documentation.docx      # ML model documentation
├── CPI_Business_Insights ML.docx  # Business insights from ML results
├── ChallengePackaging_FinalReport.docx  # Full project report
└── cpi_overview_v4.html           # Interactive HTML overview
```

---

## Machine Learning Models

Six models were built to address distinct business questions:

### 1. RFM Segmentation
**Goal:** Segment customers by purchasing behaviour  
**Method:** Recency, Frequency, Monetary scoring — customers ranked and grouped into tiers (Champions, Loyal, At-Risk, Lost)  
**Output:** Customer segment labels used in the Power BI dashboard for targeted strategy

### 2. Revenue Forecasting
**Goal:** Predict future revenue trends  
**Method:** Time-series forecasting on invoice-level monthly aggregates  
**Output:** Forward-looking revenue projections to support inventory and staffing decisions

### 3. Anomaly Detection
**Goal:** Flag unusual invoices or billing patterns  
**Method:** Isolation Forest / statistical outlier detection on order values and payment gaps  
**Output:** Flagged anomalous transactions for manual review

### 4. Margin Prediction
**Goal:** Predict the profit margin of an order before fulfilment  
**Method:** Regression model trained on historical order features (product type, customer, volume, timing)  
**Output:** Predicted margin score per order — enables proactive pricing decisions

### 5. Payment Risk Scoring
**Goal:** Identify customers likely to delay or default on payment  
**Method:** Classification model trained on payment history, order size, and customer tenure  
**Output:** Risk score per customer — High / Medium / Low — surfaced in the dashboard

### 6. Churn Prediction
**Goal:** Identify customers at risk of stopping orders  
**Method:** Binary classification using recency of last order, frequency decline, and payment behaviour  
**Output:** Churn probability per customer with recommended re-engagement priority

---

## Power BI Dashboard — 7 Pages

Built in Power BI with DAX measures, the dashboard gives CPI's management a full operational view:

| Page | Focus |
|---|---|
| Executive Overview | KPI summary — total revenue, orders, top customers, margin snapshot |
| P&L Deep Dive | Profit and loss breakdown — revenue vs costs, margin trends over time |
| Sales Deep Dive | Order-level sales analysis, product mix, customer-wise sales trends |
| Supplier & Raw Material | Procurement analysis — supplier spend, raw material cost tracking |
| Cashflow Health | Payment timelines, outstanding receivables, cashflow patterns |
| Owner Summary | Single-page executive view designed for the business owner |
| ML Insights | Outputs from all 6 ML models — RFM segments, churn risk, payment risk, anomalies, margin prediction, forecasts |

---

## SQL Analysis

Queries in `sql/` answer core operational questions:

- Revenue by customer, by month, by product type
- Average order value and order frequency per customer
- Payment delays — average days to payment by customer
- Top 10 customers by total billing value
- Customers with declining order frequency (churn signals)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Data cleaning, feature engineering, ML modelling |
| Pandas / NumPy | Data manipulation |
| scikit-learn | RFM, classification, regression, anomaly detection |
| MySQL | Business intelligence queries |
| Jupyter Notebook | Modelling and EDA environment |
| Power BI | Dashboard and DAX measures |
| DAX | Custom KPI calculations in Power BI |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/divyabhuvaneshwaran/challenge-packaging-analysis.git
cd challenge-packaging-analysis

# Install Python dependencies
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter

# Launch Jupyter
jupyter notebook
```

Open notebooks in `ml/` sequentially. The Power BI report (`.pbix`) in `powerbi/` requires Power BI Desktop to open.

---

## Key Takeaways

- Built on **real business data** from an active manufacturing company — not a Kaggle dataset
- Covers the full analyst workflow: raw data → cleaning → SQL → ML → dashboard → written report
- Six ML models addressing six distinct business problems across customer analytics, financial risk, and operations
- Designed to be actionable — every model output connects to a decision a business owner can make

---

## About

**Analyst:** Divya Bhuvaneshwaran  
**Business:** Challenge Packaging Industries, Chennai, India  
**Data:** Real financial daybook (FY 2025–26)  
**Stack:** Python · MySQL · scikit-learn · Power BI · DAX  
**Type:** End-to-end BI system | Resume project
