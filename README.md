# Hotel Booking Cancellation Prediction Project

## Problem description

This project focuses on predicting whether a hotel booking will be **canceled** by the customer, using booking, guest, and stay features. The model is deployed as a Flask web service that returns the probability of cancellation for a given booking.

## Data Processing

- Exploratory Data Analysis (**EDA**)
- Feature importance analysis (mutual information, correlation, Random Forest feature importances)
- Data preprocessing pipeline with one-hot encoding (`DictVectorizer`)
- Missing value imputation (numerical → 0, categorical → 'NA')

The dataset includes:

- **Features**: `lead_time`, `adr`, `deposit_type`, `customer_type`, `market_segment`, `country`, `previous_cancellations`, `total_of_special_requests`, `required_car_parking_spaces`, `booking_changes`, etc. (11 numerical + 10 categorical).
- **Target**: `is_canceled` (1 = canceled, 0 = honored).

The data is a subsample of 15,000 rows from the original [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset (Antonio, Almeida & Nunes, 2019).

## Model Training

- Logistic Regression (baseline)
- Decision Tree (tuned `max_depth`, `min_samples_leaf`)
- Random Forest (tuned `n_estimators`, `max_depth`)
- XGBoost (tuned `eta`, `max_depth`) — **best model**

5-fold cross-validation was used to validate the selected model. Final metrics on the held-out test set:

- **AUC: 0.9407**
- Accuracy: 0.8630
- Precision: 0.8508
- Recall: 0.7826
- F1: 0.8153

## Deployment

The model is deployed using Flask, providing a REST API to make predictions from booking input data.

### Testing the Service:

1. Clone this repository:

```bash
git clone https://github.com/Bogdan1102/hotel-cancelation-prediction.git
cd hotel-cancelation-prediction
```

2. Install dependencies (see below).

3. Run the Flask service:

```bash
python serve.py
```

4. Send a POST request to the `/predict` endpoint with a JSON payload containing booking features. Example:

```bash
python predict-test.py
```

Or with `curl`:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Dependencies and Environment Setup

- Use `pipenv` to manage the environment:

```bash
pipenv install
pipenv shell
```

- Key files:
  - `Pipfile`: Dependency list.
  - `Dockerfile`: For containerization.
  - `requirements.txt`: Fallback for plain `pip`.

## Containerization

A Docker image has been built to simplify deployment. Steps to build and run:

```bash
docker build -t hotel-cancellation .
docker run -it --rm -p 9696:9696 hotel-cancellation
```

## Cloud Deployment

The Docker container is compatible with **AWS Elastic Beanstalk** (cap. 5.4 of *Machine Learning Bookcamp*). Deployment steps:

```bash
eb init -p docker hotel-cancellation-predictor
eb create hotel-cancellation-env
```

Test the cloud service:

```bash
curl http://hotel-cancellation-env.eba-xxxxxx.eu-west-1.elasticbeanstalk.com/health
curl -X POST http://hotel-cancellation-env.eba-xxxxxx.eu-west-1.elasticbeanstalk.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Teardown when finished:

```bash
eb terminate hotel-cancellation-env --nohangup
```

## Reproducibility

To reproduce this project:

1. The dataset is included in the `data/` folder. If needed, download it from the source:

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/aaqibqadeer/Hotel-booking-demand/master/hotel_bookings.csv \
  -o data/hotel_bookings.csv
```

2. Run the `notebook.ipynb` notebook for data exploration.

3. Train the model using:

```bash
python train.py
```

4. Follow the deployment instructions above.