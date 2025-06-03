import requests

def test_api_predict():
    url = "http://localhost:8000/predict"
    sample = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    response = requests.post(url, json=sample)
    assert response.status_code == 200
    assert "prediction" in response.json()