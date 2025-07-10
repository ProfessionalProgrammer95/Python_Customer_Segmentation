# Customer Segmentation using K-Means Clustering

## Project Overview
This project demonstrates customer segmentation using K-Means clustering on synthetic customer data. It identifies distinct customer groups based on spending habits, income, and online behavior patterns.

## Features
- Synthetic customer data generation with realistic clusters
- Data exploration and cleaning
- Feature scaling and preprocessing
- Optimal cluster determination using elbow method and silhouette scores
- K-Means clustering implementation
- Visualization of customer segments
- Cluster analysis and interpretation

## Tech Stack
- Python 3.x
- Pandas (Data manipulation)
- NumPy (Numerical operations)
- Scikit-learn (Machine learning)
- Matplotlib (Visualization)
- SciPy (Statistical operations)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ProfessionalProgrammer95/Python_Customer_Segmentation.git
   cd customer-segmentation
   
## Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies:

bash
pip install -r requirements.txt


## Run the main script:

bash
python customer_segmentation.py

## The script will:

Generate synthetic customer data

Perform data cleaning

Determine optimal clusters

Segment customers using K-Means

Save results to clustered_customers.csv

Display visualizations

## File Structure

customer-segmentation/
├── customer_segmentation.py    # Main project code
├── customer_segmentation.ipynb # Jupiter Notebook run code and output
├── README.md                   # This file
└── requirements.txt            # Dependencies

## Results Interpretation
The analysis typically identifies 4 customer segments:

High-income spenders: Younger customers with high income and spending

Value-conscious: Middle-aged with moderate income and spending

Frequent shoppers: High online purchase frequency

Retired moderate spenders: Older customers with medium spending

## Customization
To modify the project:

Adjust num_customers in generate_customer_data() for more/less data

Change optimal_clusters to try different segment counts

Modify feature selection in prepare_features()

