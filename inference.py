import time
import os
try:
    import psutil
except ImportError:
    psutil = None
from flask import Flask, request, jsonify
import mlflow.sklearn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import prometheus_exporter as metrics

app = Flask(__name__)

# Load model
MODEL_PATH = "best_model_local"
try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Start time for uptime metric
START_TIME = time.time()

# Set model info
metrics.MODEL_INFO.labels(version='1.0', algorithm='LogisticRegression').set(1)

@app.route('/predict', methods=['POST'])
@metrics.CONCURRENT_REQUESTS.track_inprogress()
def predict():
    start_time = time.time()
    metrics.INFERENCE_COUNT.inc()
    
    try:
        data = request.get_json()
        if not data:
            metrics.REQUEST_ERRORS.inc()
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')
        
        if not text:
            metrics.REQUEST_ERRORS.inc()
            return jsonify({'error': 'No text provided'}), 400
        
        metrics.INPUT_TEXT_LENGTH.observe(len(text))
        
        # Inference
        prediction = model.predict([text])[0]
        
        # Update metrics
        latency = time.time() - start_time
        metrics.INFERENCE_LATENCY.observe(latency)
        metrics.PREDICTION_LABEL.labels(label=str(prediction)).inc()
        metrics.SUCCESS_COUNT.inc()
        
        return jsonify({
            'prediction': int(prediction),
            'latency': latency
        })
        
    except Exception as e:
        metrics.REQUEST_ERRORS.inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    # Update system metrics before returning
    metrics.UPTIME.set(time.time() - START_TIME)
    try:
        metrics.MEMORY_USAGE.set(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/')
def home():
    return "Inference Service is running. Use /predict for predictions and /metrics for Prometheus metrics."

if __name__ == '__main__':
    # Running on 5001 to avoid conflicts
    app.run(host='0.0.0.0', port=5001)
