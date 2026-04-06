--Sales Trend--
SELECT 
    strftime('%Y-%m', Date) AS Month,
    ROUND(SUM(Debit_Amount), 0) AS Total_Sales
FROM day_book
WHERE Transaction_Category = 'Sales'
GROUP BY Month
ORDER BY Month;
