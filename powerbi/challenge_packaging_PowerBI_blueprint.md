
# ============================================================
# CHALLENGE PACKAGING INDUSTRIES
# Power BI Dashboard Blueprint
# Author: Divya | Resume Project
# ============================================================

## 📋 OVERVIEW
This document guides you step-by-step through building a
professional Power BI dashboard for Challenge Packaging Industries.
The dashboard tells a complete business story across 5 report pages.

---

## 🗂️ STEP 1: DATA IMPORT INTO POWER BI

### 1.1 Import the Cleaned CSV Files
In Power BI Desktop:
1. Home → Get Data → Text/CSV
2. Import these files:
   - day_book_clean.csv
   - purchase_register_clean.csv  
   - sales_register_clean.csv
   - rfm_scores.csv
   - margin_analysis.csv
   - sales_forecast.csv

### 1.2 OR Connect directly to the SQLite DB
1. Home → Get Data → ODBC
2. Use "SQLite ODBC Driver" and point to challenge_packaging.db

---

## 🔧 STEP 2: POWER QUERY TRANSFORMATIONS

### Day Book Transformations
```powerquery
let
    Source = Csv.Document(...),
    #"Changed Type" = Table.TransformColumnTypes(Source,{
        {"txn_date", type date},
        {"debit_amount", type number},
        {"credit_amount", type number}
    }),
    #"Added Fiscal Year" = Table.AddColumn(#"Changed Type", "FY_Quarter",
        each if Date.Month([txn_date]) >= 4 then
            "Q" & Text.From(Number.IntegerDivide(Date.Month([txn_date]) - 4, 3) + 1)
        else
            "Q4"),
    #"Filtered Noise" = Table.SelectRows(#"Added Fiscal Year",
        each [debit_amount] > 0 or [credit_amount] > 0)
in
    #"Filtered Noise"
```

---

## 📐 STEP 3: DATA MODEL (RELATIONSHIPS)

```
day_book ─────────────────────────────────
   │ txn_date → Calendar[Date]            
   │ particulars → dim_customer[name]      
   │                                       
purchase_register                          
   │ txn_date → Calendar[Date]            
   │ supplier_name → dim_supplier[name]   
                                           
Calendar Table (mark as Date Table):       
   Date, Year, Month_Num, Month_Name,     
   Quarter, FY, Week_No                   
```

### Create Calendar Table (DAX)
```dax
Calendar = 
ADDCOLUMNS(
    CALENDAR(DATE(2024,4,1), DATE(2026,3,31)),
    "Year",           YEAR([Date]),
    "Month_Num",      MONTH([Date]),
    "Month_Name",     FORMAT([Date], "MMM"),
    "Quarter",        "Q" & FORMAT(CEILING((MONTH([Date])-3)/3,1), "0"),
    "FY",             IF(MONTH([Date])>=4,
                        YEAR([Date]) & "-" & RIGHT(YEAR([Date])+1,2),
                        YEAR([Date])-1 & "-" & RIGHT(YEAR([Date]),2)),
    "Month_Sort",     IF(MONTH([Date])>=4, MONTH([Date])-3, MONTH([Date])+9)
)
```

---

## 📊 STEP 4: KEY DAX MEASURES

```dax
// ── Revenue Measures ──
Total_Sales = 
    CALCULATE(SUM(day_book[debit_amount]), 
              day_book[txn_category] = "Sales")

Sales_LY = 
    CALCULATE([Total_Sales], SAMEPERIODLASTYEAR(Calendar[Date]))

Sales_Growth_Pct = 
    DIVIDE([Total_Sales] - [Sales_LY], [Sales_LY], 0)

// ── Purchase Measures ──
Total_Purchase = 
    CALCULATE(SUM(day_book[credit_amount]), 
              day_book[txn_category] = "Purchase")

// ── Gross Margin ──
Gross_Margin = [Total_Sales] - [Total_Purchase]

Gross_Margin_Pct = 
    DIVIDE([Gross_Margin], [Total_Sales], 0)

// ── Cash Flow ──
Total_Payments = 
    CALCULATE(SUM(day_book[debit_amount]),
              day_book[txn_category] = "Payment")

Total_Receipts = 
    CALCULATE(SUM(day_book[credit_amount]),
              day_book[txn_category] = "Receipt")

Net_Cash_Flow = [Total_Receipts] - [Total_Payments]

// ── Customer Metrics ──
Num_Active_Customers = 
    CALCULATE(DISTINCTCOUNT(day_book[particulars]),
              day_book[txn_category] = "Sales")

Avg_Invoice_Value = 
    DIVIDE([Total_Sales], 
           CALCULATE(COUNT(day_book[vch_no]), day_book[txn_category]="Sales"), 0)

// ── Running Total (for Pareto) ──
Cumulative_Sales = 
    CALCULATE([Total_Sales],
              FILTER(ALL(day_book), 
                     [Total_Sales] >= EARLIER([Total_Sales])))

// ── Top Customer Concentration ──
Top1_Concentration = 
    CALCULATE([Total_Sales],
              TOPN(1, VALUES(day_book[particulars]), [Total_Sales]))
    / CALCULATE([Total_Sales], ALL(day_book[particulars]))
```

---

## 🖥️ STEP 5: DASHBOARD PAGES (5 PAGES)

---

### PAGE 1: EXECUTIVE SUMMARY
**Title:** "Challenge Packaging — FY 2024-25 Snapshot"
**Audience:** Owner / Investor

**KPI Cards (top row):**
| Card | Measure | Icon |
|------|---------|------|
| Total Revenue | Total_Sales | 📈 |
| Gross Margin % | Gross_Margin_Pct | 💹 |
| Net Cash Flow | Net_Cash_Flow | 💰 |
| Active Customers | Num_Active_Customers | 👥 |

**Visuals:**
1. Line Chart — Monthly Revenue Trend (with forecast line)
   - X: Month_Name (sorted), Y: Total_Sales
   - Add reference line at average
2. Clustered Bar — Sales vs Purchase by Month
   - Side-by-side bars, highlighting margin gap
3. Card — "Sundram Fasteners contributes X% of revenue"
   (single number storytelling)
4. Gauge — Gross Margin % (target: 30%, max: 60%)

**Color Theme:** Dark blue (#1B3A6B) + Gold (#F5A623)

---

### PAGE 2: SALES DEEP DIVE
**Title:** "Who Buys From Us & When?"

**Visuals:**
1. Treemap — Revenue by Customer
   - Hierarchy: Customer → Invoice Count
   - Color by RFM Segment (🟢 Champion, 🟡 Growing, 🔴 At Risk)
2. Pareto Chart — Cumulative Revenue %
   - Bar: individual customer revenue
   - Line: cumulative %, add 80% reference line
   - Story: "Top 3 customers = 73% of revenue"
3. Matrix / Heatmap — Customer × Month Revenue
   - Conditional formatting (green = high, red = low)
4. Stacked Column — Sales by Quarter by FY
5. Slicer: Fiscal Year | Quarter | Customer Segment

---

### PAGE 3: PURCHASE & SUPPLIER ANALYSIS
**Title:** "What We Spend & With Whom"

**Visuals:**
1. Horizontal Bar — Top 10 Suppliers by Spend
   - Color by spend bucket
2. Donut Chart — Spend by Supplier Category
   (Paper, Inks, Adhesives, Packaging, Others)
3. Line Chart — Purchase trend vs Sales trend
   - Dual axis: Purchase (left), Sales (right)
   - Story: "Are we over-buying in Q3?"
4. Table — Supplier Detail
   Columns: Supplier | Total Spend | # Bills | Avg Bill | Last Purchase
5. KPI: Purchase-to-Sales Ratio (target < 0.65)

---

### PAGE 4: CASH FLOW & FINANCIAL HEALTH
**Title:** "The Money Story — In vs Out"

**Visuals:**
1. Waterfall Chart — Monthly Net Cash Flow
   - Green bars = positive months, Red = negative
   - "Cash-positive months: Jul, Aug, Jan, Mar"
2. Area Chart — Cumulative Cash In vs Cash Out
   - Fill between lines (green = surplus, red = deficit)
3. Clustered Bar — Payment vs Receipt by Month
4. KPI Cards:
   - Best Cash Month
   - Worst Cash Month  
   - Total Outstanding
5. Tooltip page: hover on a month to see top 5 payors

---

### PAGE 5: ML INSIGHTS — PREDICTIONS & ALERTS
**Title:** "What the Data Predicts"

**Visuals:**
1. Line + Dotted Forecast — Revenue Forecast 6 Months
   - Solid line = actual, Dotted = ML forecast
   - Confidence band (shaded area)
2. Scatter Plot — RFM Segmentation
   - X: Frequency, Y: Monetary, Size: Recency
   - Color: Segment (Champion/Growing/At Risk)
3. Table — Anomalous Transactions (Alert List)
   Columns: Date | Customer | Amount | Flag
   Conditional format: Red rows for Outliers
4. Bar Chart — Gross Margin by Month (color-coded)
   - Green if > 30%, Yellow 15-30%, Red < 15%
5. KPI: Customers At Risk Count + "Action Needed" badge

---

## 🎨 STEP 6: DESIGN GUIDELINES

### Color Palette
| Use | Color | Hex |
|-----|-------|-----|
| Primary Blue | Headers, KPIs | #1B3A6B |
| Gold/Amber | Highlights, CTA | #F5A623 |
| Success Green | Positive metrics | #27AE60 |
| Alert Red | Negative / Risk | #E74C3C |
| Neutral Gray | Background, borders | #F2F3F5 |

### Typography
- Title font: Segoe UI Bold 16pt
- KPI values: Segoe UI Bold 28pt
- Body text: Segoe UI 11pt
- Tooltips: Segoe UI Light 10pt

### Layout Rules
1. KPI cards always on top row
2. Main chart takes 60% of page width
3. Supporting chart on right 40%
4. Slicers on left panel (vertical)
5. Page navigation buttons at bottom
6. Company logo + report title in header banner

---

## 📁 STEP 7: FILE EXPORT & SHARING

1. Save as: ChallengePackaging_DashBoard_FY2024-26.pbix
2. Publish to Power BI Service (free workspace)
3. Export PDF: File → Export → Export to PDF
4. Set up Auto-Refresh: if using CSV, schedule refresh
5. Create mobile layout for each page

---

## 🏆 RESUME TALKING POINTS

Use these bullet points in your resume:

• Built end-to-end data analytics pipeline for a carton box manufacturing SME 
  using Python (pandas, scikit-learn), SQL (SQLite), and Power BI

• Cleaned and structured 2,962 financial transactions across Day Book,
  Purchase Register, and Sales Register spanning FY 2024-26

• Discovered that 1 customer (Sundram Fasteners) contributes 64% of total 
  revenue — flagged as concentration risk and modeled mitigation scenarios

• Built 5 ML models: revenue forecasting (Random Forest + Gradient Boosting),
  RFM customer segmentation, Isolation Forest anomaly detection, 
  gross margin prediction, and payment behavior risk classifier

• Identified 18 anomalous sales transactions worth reviewing for GST compliance
  and revenue leakage using statistical Z-score and Isolation Forest methods

• Delivered a 5-page Power BI dashboard with 20+ visualizations enabling 
  the business owner to monitor cash flow, customer health, and supplier spend

