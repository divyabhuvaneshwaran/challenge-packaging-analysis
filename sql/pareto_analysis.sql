WITH customer_totals AS (
    SELECT 
        Particulars AS Customer,
        ROUND(SUM(Debit_Amount), 0) AS Revenue
    FROM day_book
    WHERE Transaction_Category = 'Sales'
    GROUP BY Particulars
)
SELECT 
    Customer,
    Revenue,
    ROUND(Revenue * 100.0 / SUM(Revenue) OVER(), 1) AS Pct,
    ROUND(SUM(Revenue) OVER (ORDER BY Revenue DESC) * 100.0 / SUM(Revenue) OVER(), 1) AS Cumulative_Pct
FROM customer_totals
ORDER BY Revenue DESC
LIMIT 10;