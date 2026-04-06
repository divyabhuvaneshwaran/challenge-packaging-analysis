--Cash flow--

SELECT 
    strftime('%Y-%m', Date) AS Month,
    ROUND(SUM(CASE WHEN Transaction_Category = 'Receipt' 
              THEN Debit_Amount ELSE 0 END), 0) AS Cash_In,
    ROUND(SUM(CASE WHEN Transaction_Category = 'Payment' 
              THEN Debit_Amount ELSE 0 END), 0) AS Cash_Out,
    ROUND(SUM(CASE WHEN Transaction_Category = 'Receipt' 
              THEN Debit_Amount ELSE 0 END) -
          SUM(CASE WHEN Transaction_Category = 'Payment' 
              THEN Debit_Amount ELSE 0 END), 0) AS Net_Cash_Flow
FROM day_book
GROUP BY Month
ORDER BY Month;

SELECT DISTINCT Transaction_Category 
FROM day_book;

SELECT 
    strftime('%Y-%m', Date) AS Month,
    ROUND(SUM(CASE WHEN Transaction_Category = 'Receipt' 
              THEN Credit_Amount ELSE 0 END), 0) AS Cash_In,
    ROUND(SUM(CASE WHEN Transaction_Category = 'Payment' 
              THEN Debit_Amount ELSE 0 END), 0) AS Cash_Out,
    ROUND(SUM(CASE WHEN Transaction_Category = 'Receipt' 
              THEN Credit_Amount ELSE 0 END) -
          SUM(CASE WHEN Transaction_Category = 'Payment' 
              THEN Debit_Amount ELSE 0 END), 0) AS Net_Cash_Flow
FROM day_book
GROUP BY Month
ORDER BY Month;