import yfinance as yf
import pandas as pd


ticker = yf.Ticker("PFE")

# 1. Transpose so labels are columns
df_t = ticker.cashflow.T

cash = ticker.balance_sheet.T

# print(cash.index.tolist())

# 2. Fill missing values with 0 so the math doesn't break
# df_t = df_t.fillna(0)

# 3. Calculate (Using + if the values in your DF are already negative)
# Check your DF: if CapEx is -2.6e09, use + to subtract it correctly.

# print(df_t['Operating Cash Flow'])
# print(df_t['Capital Expenditure'])
# print(df_t['Repayment Of Debt'])
print(cash['Cash Cash Equivalents And Short Term Investments'])


lfc = (
    df_t['Operating Cash Flow'] -
    df_t['Capital Expenditure'].abs() -
    df_t['Repayment Of Debt'].abs()
)


# print(lfc)



# lfc = ticker.get_cashflow(freq = 'quarterly')
#
# hello = ticker.get_eps_trend(as_dict=True)
#
# df = ticker.cashflow
# print(df.index.tolist())
#
#
#
# lfc = df['Operating Cash Flow'] - df['Capital Expenditure'] - df['Repayment Of Debt']
# print(lfc)




{'current': 
 {'0q': 0.72751, '+1q': 0.70166, '0y': 2.96126, '+1y': 2.82981}, 
 '7daysAgo': {'0q': 0.72774, '+1q': 0.70189, '0y': 2.96187, '+1y': 2.83123}, 
 '30daysAgo': {'0q': 0.72736, '+1q': 0.71786, '0y': 2.96067, '+1y': 2.81668}, 
 '60daysAgo': {'0q': 0.74818, '+1q': 0.73182, '0y': 2.98981, '+1y': 2.88739}, 
 '90daysAgo': {'0q': 0.77429, '+1q': 0.72857, '0y': 3.09284, '+1y': 2.99231}}

['Free Cash Flow', 'Repurchase Of Capital Stock', 'Repayment Of Debt', 'Issuance Of Debt', 'Capital Expenditure', 'Interest Paid Supplemental Data', 'Income Tax Paid Supplemental Data', 'End Cash Position', 'Beginning Cash Position', 'Effect Of Exchange Rate Changes', 'Changes In Cash', 'Financing Cash Flow', 'Cash From Discontinued Financing Activities', 'Cash Flow From Continuing Financing Activities', 'Net Other Financing Charges', 'Cash Dividends Paid', 'Common Stock Dividend Paid', 'Net Common Stock Issuance', 'Common Stock Payments', 'Net Issuance Payments Of Debt', 'Net Short Term Debt Issuance', 'Short Term Debt Payments', 'Short Term Debt Issuance', 'Net Long Term Debt Issuance', 'Long Term Debt Payments', 'Long Term Debt Issuance', 'Investing Cash Flow', 'Cash From Discontinued Investing Activities', 'Cash Flow From Continuing Investing Activities', 'Net Other Investing Changes', 'Dividends Received Cfi', 'Net Investment Purchase And Sale', 'Sale Of Investment', 'Purchase Of Investment', 'Net Business Purchase And Sale', 'Sale Of Business', 'Purchase Of Business', 'Net PPE Purchase And Sale', 'Purchase Of PPE', 'Operating Cash Flow', 'Cash From Discontinued Operating Activities', 'Cash Flow From Continuing Operating Activities', 'Change In Working Capital', 'Change In Other Working Capital', 'Change In Other Current Liabilities', 'Change In Other Current Assets', 'Change In Payables And Accrued Expense', 'Change In Payable', 'Change In Account Payable', 'Change In Inventory', 'Change In Receivables', 'Changes In Account Receivables', 'Other Non Cash Items', 'Stock Based Compensation', 'Asset Impairment Charge', 'Deferred Tax', 'Deferred Income Tax', 'Depreciation Amortization Depletion', 'Depreciation And Amortization', 'Earnings Losses From Equity Investments', 'Net Income From Continuing Operations']

['Treasury Shares Number', 'Ordinary Shares Number', 'Share Issued', 'Net Debt', 'Total Debt', 'Tangible Book Value', 'Invested Capital', 'Working Capital', 'Net Tangible Assets', 'Common Stock Equity', 'Total Capitalization', 'Total Equity Gross Minority Interest', 'Minority Interest', 'Stockholders Equity', 'Other Equity Interest', 'Gains Losses Not Affecting Retained Earnings', 'Other Equity Adjustments', 'Foreign Currency Translation Adjustments', 'Minimum Pension Liabilities', 'Unrealized Gain Loss', 'Treasury Stock', 'Retained Earnings', 'Additional Paid In Capital', 'Capital Stock', 'Common Stock', 'Preferred Stock', 'Total Liabilities Net Minority Interest', 'Total Non Current Liabilities Net Minority Interest', 'Other Non Current Liabilities', 'Derivative Product Liabilities', 'Employee Benefits', 'Non Current Pension And Other Postretirement Benefit Plans', 'Tradeand Other Payables Non Current', 'Non Current Deferred Liabilities', 'Non Current Deferred Taxes Liabilities', 'Long Term Debt And Capital Lease Obligation', 'Long Term Debt', 'Current Liabilities', 'Other Current Liabilities', 'Current Deferred Liabilities', 'Current Deferred Revenue', 'Current Debt And Capital Lease Obligation', 'Current Debt', 'Other Current Borrowings', 'Commercial Paper', 'Pensionand Other Post Retirement Benefit Plans Current', 'Payables And Accrued Expenses', 'Payables', 'Dividends Payable', 'Total Tax Payable', 'Income Tax Payable', 'Accounts Payable', 'Total Assets', 'Total Non Current Assets', 'Other Non Current Assets', 'Non Current Deferred Assets', 'Non Current Deferred Taxes Assets', 'Investments And Advances', 'Other Investments', 'Investmentin Financial Assets', 'Held To Maturity Securities', 'Available For Sale Securities', 'Long Term Equity Investment', 'Goodwill And Other Intangible Assets', 'Other Intangible Assets', 'Goodwill', 'Net PPE', 'Accumulated Depreciation', 'Gross PPE', 'Construction In Progress', 'Other Properties', 'Machinery Furniture Equipment', 'Buildings And Improvements', 'Land And Improvements', 'Properties', 'Current Assets', 'Other Current Assets', 'Inventory', 'Other Inventories', 'Finished Goods', 'Work In Process', 'Raw Materials', 'Receivables', 'Taxes Receivable', 'Accounts Receivable', 'Allowance For Doubtful Accounts Receivable', 'Gross Accounts Receivable', 'Cash Cash Equivalents And Short Term Investments', 'Other Short Term Investments', 'Cash And Cash Equivalents']
