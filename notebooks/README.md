# Fraud Detection for E-Commerce and Banking Transactions


### Project Overview:

- This project develops robust fraud detection models for e-commerce and credit card transactions using advanced machine learning and geolocation analysis. The system ensures accurate fraud detection, real-time monitoring, and reporting to reduce financial losses and build trust with customers and institutions.
---
Table of Contents:
1. [Datasets](#datasets)
2. [Installation](#installation)
3. [Contributing](#contributing)
4. [License](#license)
---

## Datasets <a name="datasets"></a>

- fraud_data: Contains bank transaction data specifically curated for fraud detection analysis.
- credit_card: Includes e-commerce transaction data aimed at identifying fraudulent activities.
- ip_address: Maps IP addresses to countries

    1. **fraud_data**
        - This dataset contains interesting columns from the bank transactions. Certain important columns are:
            1. `purchase_time`: The time and date when the purchase of an item was made
            2. `purchase_value`: The amount spent on the item
            3. `source`: The source the customer came from (Ads, SEO,...) 
            4. `browser`: The internet browser used when purchasing the item (safari, chrome, opera,...
            5. `class`: Division for the fraudulent activity- 0 for non-fraudulent and 1 for fraudulent
    2. **credit_card**
        - 