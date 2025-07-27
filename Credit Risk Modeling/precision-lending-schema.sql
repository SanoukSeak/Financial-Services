
-- Create Branches Table
CREATE TABLE bankingb.Branches (
    BranchID VARCHAR(10) PRIMARY KEY,
    Street VARCHAR(255),
    City VARCHAR(100),
    StateProvince VARCHAR(100),
    Country VARCHAR(100),
    ZipCode VARCHAR(20)
);

-- Create Customers Table
CREATE TABLE bankingb.Customers (
    CustomerID VARCHAR(10) PRIMARY KEY,
    DateOfBirth DATE,
    Gender ENUM('Male', 'Female'),
    PhoneNumber VARCHAR(20)
);

-- Create Accounts Table
CREATE TABLE bankingb.Accounts (
    AccountID VARCHAR(10) PRIMARY KEY,
    CustomerID VARCHAR(10),
    BranchID VARCHAR(10),
    AccountType ENUM('Savings', 'Checking', 'Business'),
    Balance DECIMAL(15, 2),
    OpenedDate DATE,
    FOREIGN KEY (CustomerID) REFERENCES bankingb.Customers(CustomerID),
    FOREIGN KEY (BranchID) REFERENCES bankingb.Branches(BranchID)
);


-- Create CreditScores Table
CREATE TABLE bankingb.CreditScores (
    ScoreID VARCHAR(10) PRIMARY KEY,
    CustomerID VARCHAR(10),
    CreditScore INT CHECK (CreditScore BETWEEN 300 AND 850),
    ScoreDate DATE,
    Source ENUM('Experian', 'Equifax', 'TransUnion'),
    CreditUtilization DECIMAL(5, 2),
    FOREIGN KEY (CustomerID) REFERENCES bankingb.Customers(CustomerID)
);

-- Create Loans Table
CREATE TABLE bankingb.Loans (
    LoanID VARCHAR(10) PRIMARY KEY,
    CustomerID VARCHAR(10),
    LoanType ENUM('Personal', 'Auto', 'Mortgage', 'Business'),
    PrincipalAmount DECIMAL(15, 2),
    InterestRate DECIMAL(5, 2),
    StartDate DATE,
    EndDate DATE,
    Status ENUM('Active', 'Closed', 'Defaulted'),
    DefaultDate DATE NULL,
    TotalPaid DECIMAL(15, 2),
    RecoveryStatus ENUM('Fully Recovered', 'Partially Recovered', 'Unrecovered'),
    FOREIGN KEY (CustomerID) REFERENCES bankingb.Customers(CustomerID)
);

-- Create PaymentHistory Table
CREATE TABLE bankingb.PaymentHistory (
    PaymentID VARCHAR(10) PRIMARY KEY,
    LoanID VARCHAR(10),
    PaymentDate DATE,
    PaymentAmount DECIMAL(15, 2),
    MissedPayment BOOLEAN,
    PaymentDueDate DATE,
    FOREIGN KEY (LoanID) REFERENCES bankingb.Loans(LoanID)
);

-- Create Transactions Table
CREATE TABLE bankingb.Transactions (
    TransactionID VARCHAR(36) PRIMARY KEY,
    AccountID VARCHAR(10),
    TransactionDate DATE,
    Amount DECIMAL(15, 2),
    TransactionType ENUM('Deposit', 'Withdrawal'),
    Description ENUM(
        'Loan Payment', 'Transfer to Savings', 'Bill Payment', 'Online Purchase',
        'ATM Withdrawal', 'Salary Credit', 'Utility Bill', 'Insurance Premium',
        'Subscription Fee', 'Grocery Shopping', 'Dining', 'Travel Expense',
        'Car Payment', 'Rent', 'Tuition Fee', 'Refund'
    ),
    FOREIGN KEY (AccountID) REFERENCES bankingb.Accounts(AccountID)
);

-- Create RiskAssessment Table
CREATE TABLE bankingb.RiskAssessment (
    RiskID VARCHAR(10) PRIMARY KEY,
    CustomerID VARCHAR(10),
    RiskScore DECIMAL(5,2),
    RiskCategory ENUM('Low', 'Medium', 'High'),
    FOREIGN KEY (CustomerID) REFERENCES bankingb.Customers(CustomerID)
);

-- Create LoanPerformance Table
CREATE TABLE bankingb.LoanPerformance (
    PerformanceID VARCHAR(10) PRIMARY KEY,
    LoanID VARCHAR(10),
    DefaultDate DATE,
    TotalPaid DECIMAL(15, 2),
    RecoveryStatus ENUM('Fully Recovered', 'Partially Recovered', 'Unrecovered'),
    TimeToDefault INT, -- Difference in days between loan start and default date
    PAR_30 DECIMAL(15, 2),
    PAR_60 DECIMAL(15, 2),
    PAR_90 DECIMAL(15, 2),
    FOREIGN KEY (LoanID) REFERENCES bankingb.Loans(LoanID)
);



-- 1. Insert Branches
INSERT INTO bankingb.Branches (BranchID, Street, City, StateProvince, Country, ZipCode)
VALUES 
    ('B0001', '200 West Street', 'New York', 'NY', 'USA', '10282'),
    ('B0002', 'Financial District', 'Dallas', 'TX', 'USA', '75201'),
    ('B0003', 'Finance Hub', 'Salt Lake City', 'UT', 'USA', '84101'),
    ('B0004', 'Downtown Business Center', 'Boston', 'MA', 'USA', '02108'),
    ('B0005', 'Chicago Financial Plaza', 'Chicago', 'IL', 'USA', '60601'),
    ('B0006', 'Market Street Banking', 'San Francisco', 'CA', 'USA', '94103'),
    ('B0007', 'Peachtree Banking Hub', 'Atlanta', 'GA', 'USA', '30303'),
    ('B0008', 'Brickell Financial District', 'Miami', 'FL', 'USA', '33131'),
    ('B0009', 'Downtown Financial Hub', 'Houston', 'TX', 'USA', '77002'),
    ('B0010', 'Wilshire Blvd Business', 'Los Angeles', 'CA', 'USA', '90017'),
    ('B0011', 'Bay Street Financial', 'Toronto', 'Ontario', 'Canada', 'M5J 2J7'),
    ('B0012', 'Infinity Building', 'São Paulo', 'SP', 'Brazil', '01000-000'),
    ('B0013', 'Business District', 'Mexico City', 'CDMX', 'Mexico', '06000'),
    ('B0014', 'Canary Wharf', 'London', 'England', 'UK', 'E14 5AB'),
    ('B0015', 'Finance Tower', 'Frankfurt', 'Hesse', 'Germany', '60311'),
    ('B0016', 'Business Plaza', 'Munich', 'Bavaria', 'Germany', '80331'),
    ('B0017', 'Financial Quarter', 'Paris', 'Île-de-France', 'France', '75001'),
    ('B0018', 'Banking Hub', 'Zurich', 'Zurich', 'Switzerland', '8001'),
    ('B0019', 'Financial District', 'Geneva', 'Geneva', 'Switzerland', '1204'),
    ('B0020', 'Downtown Financial Center', 'Dubai', 'Dubai', 'UAE', '00000'),
    ('B0021', 'Capital Business Park', 'Riyadh', 'Riyadh', 'Saudi Arabia', '11564'),
    ('B0022', 'CBD Financial District', 'Beijing', 'Beijing', 'China', '100020'),
    ('B0023', 'Lujiazui Financial Hub', 'Shanghai', 'Shanghai', 'China', '200120'),
    ('B0024', 'Technology Finance Plaza', 'Shenzhen', 'Guangdong', 'China', '518000'),
    ('B0025', 'UB City', 'Bengaluru', 'Karnataka', 'India', '560001'),
    ('B0026', 'Hitech Business Park', 'Hyderabad', 'Telangana', 'India', '500081'),
    ('B0027', 'Marunouchi Financial Center', 'Tokyo', 'Tokyo', 'Japan', '100-0005'),
    ('B0028', 'Central Banking Hub', 'Hong Kong', 'Hong Kong', 'Hong Kong', '00000'),
    ('B0029', 'Raffles Place Business Center', 'Singapore', 'Singapore', 'Singapore', '048619'),
    ('B0030', 'Martin Place Banking', 'Sydney', 'NSW', 'Australia', '2000'),
    ('B0031', 'Collins Street Finance', 'Melbourne', 'Victoria', 'Australia', '3000'),
    ('B0032', 'Central Business Hub', 'Perth', 'WA', 'Australia', '6000');

-- 2. Insert 3,240 Customers
INSERT IGNORE INTO bankingb.Customers (CustomerID, DateOfBirth, Gender, PhoneNumber)
SELECT 
    CONCAT('C', LPAD(ROW_NUMBER() OVER (), 4, '0')),
    DATE_ADD('1960-01-01', INTERVAL FLOOR(RAND() * 20000) DAY),
    ELT(FLOOR(RAND()*2)+1, 'Male', 'Female'),
    CONCAT('+1', FLOOR(RAND()*9000000000 + 1000000000))
FROM (SELECT 1 FROM information_schema.tables, information_schema.columns LIMIT 3240) AS Temp;

-- 3. Insert ~1.5 Accounts per Customer
INSERT INTO bankingb.Accounts (AccountID, CustomerID, BranchID, AccountType, Balance, OpenedDate)
SELECT 
    CONCAT('A', LPAD(ROW_NUMBER() OVER (ORDER BY C.CustomerID), 4, '0')),  -- Using ROW_NUMBER() for unique AccountID
    C.CustomerID,
    (SELECT BranchID FROM bankingb.Branches ORDER BY RAND() LIMIT 1),
    ELT(FLOOR(RAND()*3)+1, 'Savings', 'Checking', 'Business'),
    ROUND(RAND() * 10000, 2) AS Balance,
    DATE_ADD('2010-01-01', INTERVAL FLOOR(RAND() * 5000) DAY)
FROM bankingb.Customers C;

-- 4. Insert 1 CreditScore per Customer
INSERT INTO bankingb.CreditScores (ScoreID, CustomerID, CreditScore, ScoreDate, Source, CreditUtilization)
SELECT 
    CONCAT('S', LPAD(ROW_NUMBER() OVER (ORDER BY C.CustomerID), 4, '0')),  -- Using ROW_NUMBER() for unique ScoreID
    CustomerID,
    FLOOR(RAND() * 551 + 300),
    DATE_ADD('2023-01-01', INTERVAL FLOOR(RAND() * 365) DAY),
    ELT(FLOOR(RAND()*3)+1, 'Experian', 'Equifax', 'TransUnion'),
    ROUND(RAND() * 90, 2)
FROM bankingb.Customers C;

-- 5. Insert ~2,690 Loans
INSERT INTO bankingb.Loans (LoanID, CustomerID, LoanType, PrincipalAmount, InterestRate, StartDate, EndDate, Status, DefaultDate, TotalPaid, RecoveryStatus)
SELECT 
    CONCAT('L', LPAD(ROW_NUMBER() OVER (ORDER BY C.CustomerID), 4, '0')),  -- Sequential LoanID
    C.CustomerID,
    ELT(FLOOR(RAND()*4)+1, 'Personal', 'Auto', 'Mortgage', 'Business'),
    ROUND(RAND()*95000 + 5000, 2),
    ROUND(RAND()*15 + 2, 2),
    DATE_ADD('2015-01-01', INTERVAL FLOOR(RAND()*3287) DAY),
    DATE_ADD('2030-01-01', INTERVAL FLOOR(RAND()*3287) DAY),
    ELT(FLOOR(RAND()*3)+1, 'Active', 'Closed', 'Defaulted'),
    CASE WHEN RAND() < 0.3 THEN DATE_ADD('2021-01-01', INTERVAL FLOOR(RAND()*700) DAY) ELSE NULL END,
    ROUND(RAND()*90000, 2),
    ELT(FLOOR(RAND()*3)+1, 'Fully Recovered', 'Partially Recovered', 'Unrecovered')
FROM bankingb.Customers C
WHERE RAND() < 0.85
LIMIT 2690;

-- 6. Insert 4 PaymentHistory entries per Loan
SET @payment_counter = 0;

INSERT INTO bankingb.PaymentHistory (PaymentID, LoanID, PaymentDate, PaymentAmount, MissedPayment, PaymentDueDate)
SELECT 
    CONCAT('P', LPAD(@payment_counter := @payment_counter + 1, 4, '0')) AS PaymentID,  -- Generate a unique PaymentID based on a counter
    L.LoanID,
    DATE_ADD('2022-01-01', INTERVAL N.n * 30 + FLOOR(RAND() * 10) DAY) AS PaymentDate,
    ROUND(RAND() * 3500 + 100, 2) AS PaymentAmount,
    IF(RAND() < 0.25, TRUE, FALSE) AS MissedPayment,
    DATE_ADD('2022-01-01', INTERVAL N.n * 30 DAY) AS PaymentDueDate
FROM bankingb.Loans L
JOIN (
    SELECT 0 AS n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3
) N ON RAND() < 0.8;

-- 7 Transactions: Insert ~17 transactions per account to reach ~52,410 records
INSERT INTO bankingb.Transactions (
    TransactionID, 
    AccountID, 
    TransactionDate, 
    Amount, 
    TransactionType, 
    Description
)
SELECT 
    UUID(), 
    A.AccountID,
    DATE_ADD('2018-01-01', INTERVAL FLOOR(RAND() * 1095) DAY),  -- Random date within 5 years
    ROUND(RAND() * 9900 + 100, 2),  -- Amount between 100.00 and 10000.00
    ELT(FLOOR(RAND() * 2) + 1, 'Deposit', 'Withdrawal'),
    ELT(
        FLOOR(RAND() * 16) + 1,
        'Loan Payment', 'Transfer to Savings', 'Bill Payment', 'Online Purchase',
        'ATM Withdrawal', 'Salary Credit', 'Utility Bill', 'Insurance Premium',
        'Subscription Fee', 'Grocery Shopping', 'Dining', 'Travel Expense',
        'Car Payment', 'Rent', 'Tuition Fee', 'Refund'
    )
FROM 
    bankingb.Accounts A
    JOIN (
        SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5
        UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9 UNION ALL SELECT 10
        UNION ALL SELECT 11 UNION ALL SELECT 12 UNION ALL SELECT 13 UNION ALL SELECT 14 UNION ALL SELECT 15
        UNION ALL SELECT 16 UNION ALL SELECT 17
    ) AS X;

-- 8. Insert RiskAssessment
INSERT INTO bankingb.RiskAssessment (RiskID, CustomerID, RiskScore, RiskCategory)
SELECT 
    CONCAT('R', UPPER(SUBSTRING(MD5(RAND()), 1, 7))),  -- Generates 'R' + 7 random alphanumeric characters
    CustomerID,
    ROUND(RAND() * 100, 2),
    ELT(FLOOR(RAND()*3)+1, 'Low', 'Medium', 'High')
FROM bankingb.Customers;

-- 9. Insert LoanPerformance for 25% of Loans
INSERT INTO bankingb.LoanPerformance (PerformanceID, LoanID, DefaultDate, TotalPaid, RecoveryStatus, TimeToDefault, PAR_30, PAR_60, PAR_90)
SELECT 
    CONCAT('P', UPPER(SUBSTRING(MD5(RAND()), 1, 7))),  -- Unique 'P' + 7 random characters
    L.LoanID,
    L.DefaultDate,
    L.TotalPaid,
    L.RecoveryStatus,
    DATEDIFF(L.DefaultDate, L.StartDate),
    ROUND(RAND()*L.PrincipalAmount, 2),
    ROUND(RAND()*L.PrincipalAmount, 2),
    ROUND(RAND()*L.PrincipalAmount, 2)
FROM bankingb.Loans L
WHERE L.Status = 'Defaulted' AND L.DefaultDate IS NOT NULL
LIMIT 650;
