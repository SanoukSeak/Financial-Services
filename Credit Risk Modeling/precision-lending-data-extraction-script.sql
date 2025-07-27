
# Extracting necessary data for customers as "customer.csv"
SELECT 
    c.CustomerID, 
    c.DateOfBirth, 
    c.Gender, 
    cs.CreditScore, 
    cs.ScoreDate, 
    cs.Source AS CreditSource, 
    cs.CreditUtilization, 
    ra.RiskScore, 
    ra.RiskCategory
FROM bankingb.Customers c
LEFT JOIN bankingb.CreditScores cs ON c.CustomerID = cs.CustomerID
LEFT JOIN bankingb.RiskAssessment ra ON c.CustomerID = ra.CustomerID;

# Data Extraction for "loan.csv"

SELECT 
    L.LoanID, 
    L.CustomerID,
    L.LoanType,
    L.Status, 
    L.RecoveryStatus, 
    L.StartDate, 
    L.EndDate,
    L.DefaultDate, 
    L.PrincipalAmount, 
    L.TotalPaid, 
    L.InterestRate, 
    PH.PaymentDate,
    PH.PaymentAmount, 
    PH.MissedPayment
FROM 
    bankingb.Loans L
LEFT JOIN 
    bankingb.PaymentHistory PH ON L.LoanID = PH.LoanID
ORDER BY 
    L.LoanID;

# Data Extraction for "transaction.csv"

SELECT
    t.TransactionID,
    t.AccountID,
    t.TransactionDate,
    t.Amount,
    t.TransactionType,
    t.Description,
    a.CustomerID,
    a.AccountType,
    a.Balance,
    a.OpenedDate
FROM
    bankingb.Transactions t
JOIN
    bankingb.Accounts a ON t.AccountID = a.AccountID;