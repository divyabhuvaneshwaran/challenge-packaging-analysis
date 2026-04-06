-- ============================================================
-- CHALLENGE PACKAGING INDUSTRIES
-- DayBook 2024-26 | SQL Data Cleaning & Analysis
-- Author: Divya | Resume Project
-- ============================================================

-- ============================================================
-- STEP 1: CREATE CLEAN TABLES
-- ============================================================

-- 1.1 Master Day Book table
CREATE TABLE IF NOT EXISTS day_book_clean (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_date        DATE NOT NULL,
    particulars     TEXT NOT NULL,
    vch_type        TEXT,
    vch_no          TEXT,
    debit_amount    DECIMAL(15,2) DEFAULT 0,
    credit_amount   DECIMAL(15,2) DEFAULT 0,
    txn_category    TEXT,        -- Sales / Purchase / Payment / Receipt / Journal / Contra
    fiscal_year     TEXT,        -- 2024-25 or 2025-26
    month_num       INTEGER,
    month_name      TEXT,
    quarter         TEXT         -- Q1, Q2, Q3, Q4
);

-- 1.2 Purchase Register table
CREATE TABLE IF NOT EXISTS purchase_register_clean (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_date        DATE NOT NULL,
    supplier_name   TEXT NOT NULL,
    vch_type        TEXT,
    vch_no          TEXT,
    purchase_amount DECIMAL(15,2) DEFAULT 0,
    fiscal_year     TEXT,
    month_name      TEXT
);

-- 1.3 Sales Register table
CREATE TABLE IF NOT EXISTS sales_register_clean (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_date        DATE NOT NULL,
    buyer_name      TEXT NOT NULL,
    buyer_gstin     TEXT,
    voucher_no      TEXT,
    quantity        DECIMAL(15,2),
    rate            DECIMAL(15,2),
    net_value       DECIMAL(15,2),
    gross_total     DECIMAL(15,2),
    gst_rate        TEXT,         -- 5% or 18%
    fiscal_year     TEXT
);

-- ============================================================
-- STEP 2: DATA CLEANING RULES
-- ============================================================

-- 2.1 Standardize voucher type names (merge legacy + new)
-- Raw values: "GST Sales New", "GST SALES NEW 25-26" → unified: "Sales"
-- Raw values: "Purchase New", "Purchase"             → unified: "Purchase"
-- Raw values: "Payment New", "Payment"               → unified: "Payment"
-- Raw values: "Receipt New", "Receipt"               → unified: "Receipt"

UPDATE day_book_clean
SET txn_category = CASE
    WHEN vch_type LIKE '%Sales%'    THEN 'Sales'
    WHEN vch_type LIKE '%Purchase%' THEN 'Purchase'
    WHEN vch_type LIKE '%Payment%'  THEN 'Payment'
    WHEN vch_type LIKE '%Receipt%'  THEN 'Receipt'
    WHEN vch_type LIKE '%Journal%'  THEN 'Journal'
    WHEN vch_type LIKE '%Contra%'   THEN 'Contra'
    WHEN vch_type LIKE '%Debit Note%' THEN 'Debit Note'
    WHEN vch_type LIKE '%Credit Note%' THEN 'Credit Note'
    ELSE 'Other'
END;

-- 2.2 Derive fiscal year
UPDATE day_book_clean
SET fiscal_year = CASE
    WHEN CAST(strftime('%m', txn_date) AS INTEGER) >= 4
        THEN strftime('%Y', txn_date) || '-' || SUBSTR(strftime('%Y', txn_date),3,2)+1
    ELSE strftime('%Y', txn_date)-1 || '-' || SUBSTR(strftime('%Y', txn_date),3,2)
END;

-- 2.3 Derive quarter (Indian FY: Apr=Q1, Jul=Q2, Oct=Q3, Jan=Q4)
UPDATE day_book_clean
SET quarter = CASE
    WHEN CAST(strftime('%m', txn_date) AS INTEGER) IN (4,5,6)  THEN 'Q1'
    WHEN CAST(strftime('%m', txn_date) AS INTEGER) IN (7,8,9)  THEN 'Q2'
    WHEN CAST(strftime('%m', txn_date) AS INTEGER) IN (10,11,12) THEN 'Q3'
    WHEN CAST(strftime('%m', txn_date) AS INTEGER) IN (1,2,3)  THEN 'Q4'
END;

-- 2.4 Remove rows where both debit and credit are 0 (noise rows)
DELETE FROM day_book_clean
WHERE debit_amount = 0 AND credit_amount = 0;

-- 2.5 Trim whitespace in supplier/buyer names
UPDATE purchase_register_clean
SET supplier_name = TRIM(supplier_name);

UPDATE sales_register_clean
SET buyer_name = TRIM(buyer_name);

-- 2.6 Flag duplicate voucher numbers (for audit)
SELECT vch_no, COUNT(*) as count
FROM day_book_clean
GROUP BY vch_no
HAVING COUNT(*) > 1
ORDER BY count DESC;


-- ============================================================
-- STEP 3: BUSINESS ANALYSIS QUERIES
-- ============================================================

-- ── 3.1 Monthly Revenue Trend (FY 2024-25) ──
SELECT
    strftime('%Y-%m', txn_date)         AS month,
    month_name,
    ROUND(SUM(debit_amount), 2)         AS total_sales,
    COUNT(*)                            AS num_invoices,
    ROUND(AVG(debit_amount), 2)         AS avg_invoice_value
FROM day_book_clean
WHERE txn_category = 'Sales'
  AND fiscal_year  = '2024-25'
GROUP BY month
ORDER BY month;


-- ── 3.2 Customer Revenue Concentration (Pareto / 80-20 Analysis) ──
WITH customer_totals AS (
    SELECT
        particulars                         AS customer,
        SUM(debit_amount)                   AS revenue,
        COUNT(*)                            AS transactions
    FROM day_book_clean
    WHERE txn_category = 'Sales'
    GROUP BY particulars
),
ranked AS (
    SELECT *,
        ROUND(revenue * 100.0 / (SELECT SUM(revenue) FROM customer_totals), 2) AS pct,
        SUM(revenue) OVER (ORDER BY revenue DESC) AS running_total,
        ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank
    FROM customer_totals
)
SELECT
    rank,
    customer,
    ROUND(revenue, 0)      AS revenue_inr,
    pct                    AS revenue_pct,
    ROUND(running_total * 100.0 / (SELECT SUM(revenue) FROM customer_totals), 1) AS cumulative_pct
FROM ranked
ORDER BY rank
LIMIT 15;


-- ── 3.3 Supplier Spend Analysis ──
SELECT
    supplier_name,
    ROUND(SUM(purchase_amount), 0)  AS total_spend,
    COUNT(*)                        AS num_bills,
    ROUND(AVG(purchase_amount), 0)  AS avg_bill_value,
    MIN(txn_date)                   AS first_purchase,
    MAX(txn_date)                   AS last_purchase
FROM purchase_register_clean
GROUP BY supplier_name
ORDER BY total_spend DESC
LIMIT 15;


-- ── 3.4 Monthly Gross Margin ──
WITH monthly_sales AS (
    SELECT
        strftime('%Y-%m', txn_date) AS month,
        SUM(debit_amount)           AS sales
    FROM day_book_clean
    WHERE txn_category = 'Sales'
    GROUP BY month
),
monthly_purchase AS (
    SELECT
        strftime('%Y-%m', txn_date) AS month,
        SUM(credit_amount)          AS purchase
    FROM day_book_clean
    WHERE txn_category = 'Purchase'
    GROUP BY month
)
SELECT
    s.month,
    ROUND(s.sales, 0)                                AS sales_inr,
    ROUND(p.purchase, 0)                             AS purchase_inr,
    ROUND(s.sales - p.purchase, 0)                   AS gross_margin,
    ROUND((s.sales - p.purchase) / s.sales * 100, 1) AS margin_pct
FROM monthly_sales s
JOIN monthly_purchase p ON s.month = p.month
ORDER BY s.month;


-- ── 3.5 Cash Flow Analysis (Payment vs Receipt) ──
SELECT
    strftime('%Y-%m', txn_date)     AS month,
    ROUND(SUM(CASE WHEN txn_category = 'Payment' THEN debit_amount ELSE 0 END), 0)  AS cash_out,
    ROUND(SUM(CASE WHEN txn_category = 'Receipt' THEN credit_amount ELSE 0 END), 0) AS cash_in,
    ROUND(
        SUM(CASE WHEN txn_category = 'Receipt' THEN credit_amount ELSE 0 END) -
        SUM(CASE WHEN txn_category = 'Payment' THEN debit_amount ELSE 0 END), 0
    )                                                                                 AS net_cash_flow
FROM day_book_clean
WHERE txn_category IN ('Payment', 'Receipt')
GROUP BY month
ORDER BY month;


-- ── 3.6 Transaction Volume by Day of Week (Busiest Days) ──
SELECT
    CASE strftime('%w', txn_date)
        WHEN '0' THEN 'Sunday'    WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'   WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'  WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END                         AS day_of_week,
    COUNT(*)                    AS total_txns,
    ROUND(SUM(debit_amount), 0) AS total_debit,
    ROUND(AVG(debit_amount), 0) AS avg_debit
FROM day_book_clean
WHERE txn_category = 'Sales'
GROUP BY strftime('%w', txn_date)
ORDER BY total_txns DESC;


-- ── 3.7 GST Liability Summary (Input vs Output) ──
SELECT
    fiscal_year,
    ROUND(SUM(CASE WHEN txn_category = 'Sales'    THEN debit_amount  ELSE 0 END), 0) AS output_taxable,
    ROUND(SUM(CASE WHEN txn_category = 'Purchase' THEN credit_amount ELSE 0 END), 0) AS input_taxable,
    ROUND(
        SUM(CASE WHEN txn_category = 'Sales'    THEN debit_amount  ELSE 0 END) * 0.05 -
        SUM(CASE WHEN txn_category = 'Purchase' THEN credit_amount ELSE 0 END) * 0.05
    , 0)                                                                               AS net_gst_liability_5pct
FROM day_book_clean
GROUP BY fiscal_year;


-- ── 3.8 Quarterly Performance Comparison ──
SELECT
    fiscal_year,
    quarter,
    ROUND(SUM(CASE WHEN txn_category = 'Sales'    THEN debit_amount  ELSE 0 END), 0) AS sales,
    ROUND(SUM(CASE WHEN txn_category = 'Purchase' THEN credit_amount ELSE 0 END), 0) AS purchases,
    COUNT(CASE WHEN txn_category = 'Sales' THEN 1 END)                                AS sales_invoices
FROM day_book_clean
GROUP BY fiscal_year, quarter
ORDER BY fiscal_year, quarter;


-- ── 3.9 Top Products by Rate × Quantity (Sales Register) ──
SELECT
    buyer_name,
    ROUND(AVG(rate), 2)           AS avg_rate,
    ROUND(SUM(quantity), 0)       AS total_qty,
    ROUND(SUM(gross_total), 0)    AS total_revenue,
    COUNT(*)                      AS num_orders
FROM sales_register_clean
WHERE rate > 0
GROUP BY buyer_name
ORDER BY total_revenue DESC
LIMIT 10;


-- ── 3.10 Anomaly Detection: Unusually Large Transactions ──
WITH stats AS (
    SELECT
        AVG(debit_amount)   AS avg_val,
        AVG(debit_amount * debit_amount) - AVG(debit_amount) * AVG(debit_amount) AS variance
    FROM day_book_clean
    WHERE txn_category = 'Sales' AND debit_amount > 0
)
SELECT
    txn_date,
    particulars,
    ROUND(debit_amount, 0)                                             AS amount,
    ROUND((debit_amount - avg_val) / SQRT(variance), 2)               AS z_score,
    CASE
        WHEN (debit_amount - avg_val) / SQRT(variance) > 3 THEN '🚨 Outlier'
        WHEN (debit_amount - avg_val) / SQRT(variance) > 2 THEN '⚠️  High'
        ELSE 'Normal'
    END                                                                AS flag
FROM day_book_clean, stats
WHERE txn_category = 'Sales'
  AND (debit_amount - avg_val) / SQRT(variance) > 2
ORDER BY z_score DESC
LIMIT 15;
