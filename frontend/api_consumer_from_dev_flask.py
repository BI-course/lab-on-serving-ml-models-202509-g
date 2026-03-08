import requests

url = 'http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions'

payload = {
    'monthly_fee': 60,
    'customer_age': 30,
    'support_calls': 1,
}

headers = {
    'Content-Type': 'application/json', # This header tells the server that the payload is in JSON format
    'Accept': 'application/json',       # This header tells the server that the client expects a JSON response
    'User-Agent': 'ServingMLModels-Client/1.0', # This header identifies the client application making the request (optional but can be useful for logging and debugging in the backend)
    # 'Authorization': 'Bearer <token>',  # This header is used for token-based authentication if the API requires it
}

try:
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status() # This will raise an HTTPError if the response status code is 4xx or 5xx
    result = response.json()
    print('Predicted Class =', result['Predicted Class = '])

except requests.exceptions.HTTPError as e:
    print("HTTP error:", e)
    if e.response is not None:
        print("Response body:", e.response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
