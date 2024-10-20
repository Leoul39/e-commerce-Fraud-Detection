# Fraud Detection for E-Commerce and Banking Transactions


### Project Overview:

- This project develops robust fraud detection models for e-commerce and credit card transactions using advanced machine learning and geolocation analysis. The system ensures accurate fraud detection, real-time monitoring, and reporting to reduce financial losses and build trust with customers and institutions.
---
Table of Contents:
1. [Datasets](#datasets)
2. [Installation](#installation)
3. [License](#license)
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
            4. `browser`: The internet browser used when purchasing the item (safari, chrome, opera,...)
            5. `class`: Division for the fraudulent activity- 0 for non-fraudulent and 1 for fraudulent
    2. **credit_card**
        - This dataset contains bank transaction data specifically curated for fraud detection analysis.
            1. `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
            2. `V1 to V28`: These are anonymized features resulting from a PCA transformation. Their exact nature is not disclosed for privacy reasons, but they represent the underlying patterns in the data.
            3. `Amount`: The transaction amount in dollars.
            4. `Class`: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction
    3. **Ip_address**
        - This dataset contains lower and upper bound for the ip_address and also contains contries where those ip addresses are from.
            1. `lower_bound_ip_address`: The lower bound of the IP address range.
            2. `upper_bound_ip_address`: The upper bound of the IP address range.
            3. `country`: The country corresponding to the IP address range.
---
## Installation <a name="installation"></a>
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/fraud_detection.git
    cd fraud_detection
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3. Run the notebooks to see the project
---
## License <a name="license"></a>
- This project is licensed under the MIT License. See the LICENSE file for details.