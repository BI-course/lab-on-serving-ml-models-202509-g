const API_BASE = window.location.origin;

const modelEndpointMap = {
    "decision-tree-classifier": "/api/v1/models/decision-tree-classifier/predictions",
    "decision-tree-regressor": "/api/v1/models/decision-tree-regressor/predictions",
    "naive-bayes": "/api/v1/models/naive-bayes-classifier/predictions",
    "knn": "/api/v1/models/knn-classifier/predictions",
    "svm": "/api/v1/models/svm-classifier/predictions",
    "random-forest": "/api/v1/models/random-forest-classifier/predictions",
    "cluster": "/api/v1/models/cluster/predictions"
};

function showResult(elementId, data, isError = false) {
    const el = document.getElementById(elementId);
    el.textContent = JSON.stringify(data, null, 2);
    el.classList.add("visible");
    el.classList.remove("error", "success");
    el.classList.add(isError ? "error" : "success");
}

function hideAllModelInputs() {
    [
        "dtcInputs",
        "dtrInputs",
        "nbInputs",
        "knnInputs",
        "clusterInputs"
    ].forEach(id => document.getElementById(id).classList.add("hidden"));
}

function updateModelInputs() {
    hideAllModelInputs();

    const model = document.getElementById("modelSelect").value;

    if (model === "decision-tree-classifier") {
        document.getElementById("dtcInputs").classList.remove("hidden");
    } else if (model === "decision-tree-regressor") {
        document.getElementById("dtrInputs").classList.remove("hidden");
    } else if (["naive-bayes", "svm", "random-forest"].includes(model)) {
        document.getElementById("nbInputs").classList.remove("hidden");
    } else if (model === "knn") {
        document.getElementById("knnInputs").classList.remove("hidden");
    } else if (model === "cluster") {
        document.getElementById("clusterInputs").classList.remove("hidden");
    }
}

function readNumber(id, label, isInteger = false) {
    const raw = document.getElementById(id).value.trim();
    if (raw === "") {
        throw new Error(`${label} is required.`);
    }

    const value = isInteger ? parseInt(raw, 10) : parseFloat(raw);
    if (Number.isNaN(value)) {
        throw new Error(`${label} must be a valid number.`);
    }

    return value;
}

function readText(id, label) {
    const value = document.getElementById(id).value.trim();
    if (!value) {
        throw new Error(`${label} is required.`);
    }
    return value;
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        showResult("healthResult", data, !response.ok);
    } catch (error) {
        showResult("healthResult", { error: `Failed to connect to API: ${error.message}` }, true);
    }
}

async function predict() {
    const model = document.getElementById("modelSelect").value;
    const endpointPath = modelEndpointMap[model];

    if (!endpointPath) {
        showResult("predictionResult", { error: "Invalid model selection." }, true);
        return;
    }

    let features = {};

    try {
        if (model === "decision-tree-classifier") {
            features = {
                monthly_fee: readNumber("monthly_fee", "Monthly Fee"),
                customer_age: readNumber("customer_age", "Customer Age", true),
                support_calls: readNumber("support_calls", "Support Calls", true)
            };
        } else if (model === "decision-tree-regressor") {
            features = {
                PaymentDate: readText("PaymentDate", "Payment Date"),
                CustomerType: readText("CustomerType", "Customer Type"),
                BranchSubCounty: readText("BranchSubCounty", "Branch SubCounty"),
                ProductCategoryName: readText("ProductCategoryName", "Product Category Name"),
                QuantityOrdered: readNumber("QuantityOrdered", "Quantity Ordered", true)
            };
        } else if (["naive-bayes", "svm", "random-forest"].includes(model)) {
            features = {
                Administrative: readNumber("Administrative", "Administrative", true),
                Administrative_Duration: readNumber("Administrative_Duration", "Administrative Duration"),
                Informational: readNumber("Informational", "Informational", true),
                Informational_Duration: readNumber("Informational_Duration", "Informational Duration"),
                ProductRelated: readNumber("ProductRelated", "ProductRelated", true),
                ProductRelated_Duration: readNumber("ProductRelated_Duration", "ProductRelated Duration"),
                BounceRates: readNumber("BounceRates", "BounceRates"),
                ExitRates: readNumber("ExitRates", "ExitRates"),
                PageValues: readNumber("PageValues", "PageValues"),
                SpecialDay: readNumber("SpecialDay", "SpecialDay"),
                Month: readNumber("Month", "Month", true),
                OperatingSystems: readNumber("OperatingSystems", "OperatingSystems", true),
                Browser: readNumber("Browser", "Browser", true),
                Region: readNumber("Region", "Region", true),
                TrafficType: readNumber("TrafficType", "TrafficType", true),
                VisitorType: readNumber("VisitorType", "VisitorType", true),
                Weekend: readNumber("Weekend", "Weekend", true)
            };
        } else if (model === "knn") {
            features = {
                DaysForShippingReal: readNumber("DaysForShippingReal", "Days For Shipping Real"),
                DaysForShipmentScheduled: readNumber("DaysForShipmentScheduled", "Days For Shipment Scheduled"),
                OrderItemQuantity: readNumber("OrderItemQuantity", "Order Item Quantity", true),
                Sales: readNumber("Sales", "Sales"),
                OrderProfitPerOrder: readNumber("OrderProfitPerOrder", "Order Profit Per Order"),
                ShippingMode: document.getElementById("ShippingMode").value
            };
        } else if (model === "cluster") {
            features = {
                Age: readNumber("Age", "Age"),
                Annual_Income: readNumber("Annual_Income", "Annual Income"),
                Spending_Score: readNumber("Spending_Score", "Spending Score"),
                Gender_Male: readNumber("Gender_Male", "Gender", true)
            };
        }
    } catch (error) {
        showResult("predictionResult", { error: error.message }, true);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}${endpointPath}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(features)
        });

        const data = await response.json();
        showResult("predictionResult", data, !response.ok);
    } catch (error) {
        showResult("predictionResult", { error: `Failed to connect to API: ${error.message}` }, true);
    }
}

async function recommend() {
    const itemsText = document.getElementById("itemsInput").value.trim();

    if (!itemsText) {
        showResult("recommendResult", { error: "Please enter at least one item." }, true);
        return;
    }

    const items = itemsText.split(",").map(item => item.trim()).filter(Boolean);

    if (items.length === 0) {
        showResult("recommendResult", { error: "No valid items found." }, true);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/v1/recommendations`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ items })
        });

        const data = await response.json();
        showResult("recommendResult", data, !response.ok);
    } catch (error) {
        showResult("recommendResult", { error: `Failed to connect to API: ${error.message}` }, true);
    }
}

window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("modelSelect").addEventListener("change", updateModelInputs);
    document.getElementById("healthBtn").addEventListener("click", checkHealth);
    document.getElementById("predictBtn").addEventListener("click", predict);
    document.getElementById("recommendBtn").addEventListener("click", recommend);
    updateModelInputs();
});