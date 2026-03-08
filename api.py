from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import ast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "model"))

app = Flask(__name__, template_folder="templates")
CORS(
    app,
    supports_credentials=False,
    resources={r"/api/*": {"origins": ["*"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ============================================================
# HELPERS
# ============================================================

def load_artifact(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        if filename == "association_rules.csv":
            df = pd.read_csv(path)
            df["antecedents"] = df["antecedents"].apply(
                lambda x: frozenset(str(item).strip().lower() for item in ast.literal_eval(x))
            )
            df["consequents"] = df["consequents"].apply(
                lambda x: frozenset(str(item).strip().lower() for item in ast.literal_eval(x))
            )
            return df

        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return None


def get_json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None
    return data


def validate_required_fields(data, required_fields):
    if data is None:
        return ["Invalid or missing JSON body."]
    missing = []
    for field in required_fields:
        value = data.get(field)
        if value is None or value == "":
            missing.append(field)
    return missing


def success_response(payload, status_code=200):
    return jsonify({"status": "success", **payload}), status_code


def error_response(message, status_code=400):
    return jsonify({"status": "error", "error": message}), status_code


# ============================================================
# LOAD MODELS / ARTIFACTS
# ============================================================

decisiontree_classifier_baseline = load_artifact("decisiontree_classifier_baseline.pkl")
decisiontree_regressor_optimum = load_artifact("decisiontree_regressor_optimum.pkl")
naive_Bayes_classifier_optimum = load_artifact("naive_Bayes_classifier_optimum.pkl")
knn_classifier_optimum = load_artifact("knn_classifier_optimum.pkl")
support_vector_classifier_optimum = load_artifact("support_vector_classifier_optimum.pkl")
random_forest_classifier_optimum = load_artifact("random_forest_classifier_optimum.pkl")

cluster_classifier_svm = load_artifact("cluster_classifier_svm.pkl")
association_rules = load_artifact("association_rules.csv")

label_encoders_1b = load_artifact("label_encoders_1b.pkl")
label_encoders_2 = load_artifact("label_encoders_2.pkl")
label_encoders_4 = load_artifact("label_encoders_4.pkl")
label_encoders_5 = load_artifact("label_encoders_5.pkl")

onehot_encoder_3 = load_artifact("onehot_encoder_3.pkl")
scaler_3 = load_artifact("scaler_3.pkl")
scaler_4 = load_artifact("scaler_4.pkl")
scaler_5 = load_artifact("scaler_5.pkl")


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api_root():
    return success_response({
        "message": "ML Model API (BBT 4206)",
        "version": "v1",
        "available_endpoints": {
            "health": "/api/health [GET]",
            "decision_tree_classifier": "/api/v1/models/decision-tree-classifier/predictions [POST]",
            "decision_tree_regressor": "/api/v1/models/decision-tree-regressor/predictions [POST]",
            "naive_bayes_classifier": "/api/v1/models/naive-bayes-classifier/predictions [POST]",
            "knn_classifier": "/api/v1/models/knn-classifier/predictions [POST]",
            "svm_classifier": "/api/v1/models/svm-classifier/predictions [POST]",
            "random_forest_classifier": "/api/v1/models/random-forest-classifier/predictions [POST]",
            "cluster_classifier": "/api/v1/models/cluster/predictions [POST]",
            "product_recommender": "/api/v1/recommendations [POST]"
        },
        "notes": [
            "All prediction endpoints accept JSON request bodies.",
            "Use /api/health to verify loaded models and preprocessing artifacts."
        ]
    })


@app.route("/api/health", methods=["GET"])
def health():
    return success_response({
        "alive": True,
        "models_loaded": {
            "decisiontree_classifier_baseline": decisiontree_classifier_baseline is not None,
            "decisiontree_regressor_optimum": decisiontree_regressor_optimum is not None,
            "naive_Bayes_classifier_optimum": naive_Bayes_classifier_optimum is not None,
            "knn_classifier_optimum": knn_classifier_optimum is not None,
            "support_vector_classifier_optimum": support_vector_classifier_optimum is not None,
            "random_forest_classifier_optimum": random_forest_classifier_optimum is not None,
            "cluster_classifier_svm": cluster_classifier_svm is not None,
            "association_rules": association_rules is not None
        },
        "preprocessing_loaded": {
            "label_encoders_1b": label_encoders_1b is not None,
            "label_encoders_2": label_encoders_2 is not None,
            "label_encoders_4": label_encoders_4 is not None,
            "label_encoders_5": label_encoders_5 is not None,
            "onehot_encoder_3": onehot_encoder_3 is not None,
            "scaler_3": scaler_3 is not None,
            "scaler_4": scaler_4 is not None,
            "scaler_5": scaler_5 is not None
        }
    })


# ============================================================
# DECISION TREE CLASSIFIER
# ============================================================

@app.route("/api/v1/models/decision-tree-classifier/predictions", methods=["POST"])
def predict_decision_tree_classifier():
    if decisiontree_classifier_baseline is None:
        return error_response("Model not loaded", 503)

    data = get_json_body()
    required = ["monthly_fee", "customer_age", "support_calls"]
    missing = validate_required_fields(data, required)
    if missing:
        return error_response(f"Missing features: {missing}", 400)

    try:
        X = pd.DataFrame([{
            "monthly_fee": float(data["monthly_fee"]),
            "customer_age": int(data["customer_age"]),
            "support_calls": int(data["support_calls"])
        }])

        pred = decisiontree_classifier_baseline.predict(X)[0]

        return success_response({
            "model_name": "Decision Tree Classifier",
            "prediction": int(pred)
        })
    except Exception as e:
        return error_response(str(e), 500)


# ============================================================
# DECISION TREE REGRESSOR
# ============================================================

@app.route("/api/v1/models/decision-tree-regressor/predictions", methods=["POST"])
def predict_decision_tree_regressor():
    if decisiontree_regressor_optimum is None or label_encoders_1b is None:
        return error_response("Model or encoder not loaded", 503)

    data = get_json_body()
    required = [
        "PaymentDate", "CustomerType", "BranchSubCounty",
        "ProductCategoryName", "QuantityOrdered"
    ]
    missing = validate_required_fields(data, required)
    if missing:
        return error_response(f"Missing features: {missing}", 400)

    try:
        new_data = pd.DataFrame([data])

        for col in ["CustomerType", "BranchSubCounty", "ProductCategoryName"]:
            if col in label_encoders_1b:
                new_data[col] = label_encoders_1b[col].transform(new_data[col])

        new_data["PaymentDate"] = pd.to_datetime(new_data["PaymentDate"])
        new_data["PaymentDate_year"] = new_data["PaymentDate"].dt.year
        new_data["PaymentDate_month"] = new_data["PaymentDate"].dt.month
        new_data["PaymentDate_day"] = new_data["PaymentDate"].dt.day
        new_data["PaymentDate_dayofweek"] = new_data["PaymentDate"].dt.dayofweek
        new_data["QuantityOrdered"] = new_data["QuantityOrdered"].astype(int)

        X = new_data[[
            "CustomerType", "BranchSubCounty", "ProductCategoryName", "QuantityOrdered",
            "PaymentDate_year", "PaymentDate_month", "PaymentDate_day", "PaymentDate_dayofweek"
        ]]

        pred = decisiontree_regressor_optimum.predict(X)[0]

        return success_response({
            "model_name": "Decision Tree Regressor",
            "prediction": float(pred)
        })
    except Exception as e:
        return error_response(str(e), 500)


# ============================================================
# SHARED SESSION-BASED MODELS
# ============================================================

def predict_session_model(model, model_name):
    data = get_json_body()
    required = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
        "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
        "VisitorType", "Weekend"
    ]
    missing = validate_required_fields(data, required)
    if missing:
        return error_response(f"Missing {model_name} features: {missing}", 400)

    try:
        X = pd.DataFrame([{
            "Administrative": int(data["Administrative"]),
            "Administrative_Duration": float(data["Administrative_Duration"]),
            "Informational": int(data["Informational"]),
            "Informational_Duration": float(data["Informational_Duration"]),
            "ProductRelated": int(data["ProductRelated"]),
            "ProductRelated_Duration": float(data["ProductRelated_Duration"]),
            "BounceRates": float(data["BounceRates"]),
            "ExitRates": float(data["ExitRates"]),
            "PageValues": float(data["PageValues"]),
            "SpecialDay": float(data["SpecialDay"]),
            "Month": int(data["Month"]),
            "OperatingSystems": int(data["OperatingSystems"]),
            "Browser": int(data["Browser"]),
            "Region": int(data["Region"]),
            "TrafficType": int(data["TrafficType"]),
            "VisitorType": int(data["VisitorType"]),
            "Weekend": int(data["Weekend"])
        }])

        pred = model.predict(X)[0]

        return success_response({
            "model_name": model_name,
            "prediction": int(pred)
        })
    except Exception as e:
        return error_response(str(e), 500)


@app.route("/api/v1/models/naive-bayes-classifier/predictions", methods=["POST"])
def predict_naive_bayes():
    if naive_Bayes_classifier_optimum is None:
        return error_response("Model not loaded", 503)
    return predict_session_model(naive_Bayes_classifier_optimum, "Naive Bayes Classifier")


@app.route("/api/v1/models/svm-classifier/predictions", methods=["POST"])
def predict_svm():
    if support_vector_classifier_optimum is None:
        return error_response("Model not loaded", 503)
    return predict_session_model(support_vector_classifier_optimum, "SVM Classifier")


@app.route("/api/v1/models/random-forest-classifier/predictions", methods=["POST"])
def predict_rf():
    if random_forest_classifier_optimum is None:
        return error_response("Model not loaded", 503)
    return predict_session_model(random_forest_classifier_optimum, "Random Forest Classifier")


# ============================================================
# KNN
# ============================================================

@app.route("/api/v1/models/knn-classifier/predictions", methods=["POST"])
def predict_knn():
    if knn_classifier_optimum is None:
        return error_response("Model not loaded", 503)

    if onehot_encoder_3 is None or scaler_3 is None:
        return error_response("KNN preprocessing artifacts not loaded", 503)

    data = get_json_body()
    required = [
        "DaysForShippingReal",
        "DaysForShipmentScheduled",
        "OrderItemQuantity",
        "Sales",
        "OrderProfitPerOrder",
        "ShippingMode"
    ]
    missing = validate_required_fields(data, required)
    if missing:
        return error_response(f"Missing KNN features: {missing}", 400)

    try:
        shipping_mode = str(data["ShippingMode"]).strip()

        new_data = pd.DataFrame([{
            "Days for shipping (real)": float(data["DaysForShippingReal"]),
            "Days for shipment (scheduled)": float(data["DaysForShipmentScheduled"]),
            "Order Item Quantity": int(data["OrderItemQuantity"]),
            "Sales": float(data["Sales"]),
            "Order Profit Per Order": float(data["OrderProfitPerOrder"]),
            "Shipping Mode": shipping_mode
        }])

        encoded = onehot_encoder_3.transform(new_data[["Shipping Mode"]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=onehot_encoder_3.get_feature_names_out(["Shipping Mode"]),
            index=new_data.index
        )

        new_data_preprocessed = pd.concat(
            [new_data.drop("Shipping Mode", axis=1), encoded_df],
            axis=1
        )

        new_data_scaled = scaler_3.transform(new_data_preprocessed)

        pred = knn_classifier_optimum.predict(new_data_scaled)[0]

        response_payload = {
            "model_name": "K-Nearest Neighbors Classifier",
            "prediction": int(pred)
        }

        if hasattr(knn_classifier_optimum, "predict_proba"):
            probabilities = knn_classifier_optimum.predict_proba(new_data_scaled)[0]
            response_payload["probability_class_0"] = float(probabilities[0])
            response_payload["probability_class_1"] = float(probabilities[1])

        return success_response(response_payload)
    except Exception as e:
        return error_response(f"KNN prediction failed: {str(e)}", 500)


# ============================================================
# CLUSTER CLASSIFIER
# ============================================================

@app.route("/api/v1/models/cluster/predictions", methods=["POST"])
def predict_cluster():
    if cluster_classifier_svm is None:
        return error_response("Cluster classifier not loaded", 503)

    data = get_json_body()
    required = ["Age", "Annual_Income", "Spending_Score", "Gender_Male"]
    missing = validate_required_fields(data, required)
    if missing:
        return error_response(f"Missing Cluster features: {missing}", 400)

    try:
        artifact = cluster_classifier_svm
        model = artifact["model"]
        scaler = artifact["scaler"]
        feature_columns = artifact["feature_columns"]

        X = pd.DataFrame([{
            "Age": float(data["Age"]),
            "Annual Income (k$)": float(data["Annual_Income"]),
            "Spending Score (1-100)": float(data["Spending_Score"]),
            "Gender_Male": int(data["Gender_Male"])
        }])

        X = X[feature_columns]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        return success_response({
            "model_name": "Cluster Classifier",
            "prediction": int(pred)
        })
    except Exception as e:
        return error_response(str(e), 500)


# ============================================================
# RECOMMENDER
# ============================================================

@app.route("/api/v1/recommendations", methods=["POST"])
def recommend():
    data = get_json_body()
    if data is None:
        return error_response("Invalid or missing JSON body", 400)

    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        return error_response("No items provided.", 400)

    items = [str(item).strip().lower() for item in items if str(item).strip()]
    if not items:
        return error_response("No valid items provided.", 400)

    if association_rules is None:
        return error_response("Association rules not loaded", 503)

    try:
        basket = frozenset(items)
        recommendation_scores = {}

        for _, rule in association_rules.iterrows():
            antecedent = rule["antecedents"]
            consequent = rule["consequents"]

            if antecedent.issubset(basket):
                for item in consequent:
                    if item not in basket:
                        score = float(rule["confidence"])
                        if item in recommendation_scores:
                            recommendation_scores[item] = max(recommendation_scores[item], score)
                        else:
                            recommendation_scores[item] = score

        sorted_recommendations = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return success_response({
            "input_items": items,
            "recommended_products": [item for item, _ in sorted_recommendations[:5]]
        })
    except Exception as e:
        return error_response(str(e), 500)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("FLASK_PORT", 5000)))