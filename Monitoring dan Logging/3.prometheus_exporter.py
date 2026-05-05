from prometheus_client import Counter, Histogram, Gauge, Summary

# 1. Total inference count
INFERENCE_COUNT = Counter(
    'model_inference_total', 
    'Total number of model inferences'
)

# 2. Inference latency
INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds', 
    'Time spent performing inference'
)

# 3. Prediction results (labels)
PREDICTION_LABEL = Counter(
    'model_prediction_total', 
    'Total number of predictions by label',
    ['label']
)

# 4. Request error count
REQUEST_ERRORS = Counter(
    'model_request_errors_total', 
    'Total number of request errors'
)

# 5. Input text length
INPUT_TEXT_LENGTH = Summary(
    'model_input_text_length', 
    'Length of input text for inference'
)

# 6. Memory usage (Gauge)
MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes', 
    'Current memory usage of the inference service'
)

# 7. Service uptime
UPTIME = Gauge(
    'model_service_uptime_seconds', 
    'Uptime of the inference service in seconds'
)

# 8. Concurrent requests
CONCURRENT_REQUESTS = Gauge(
    'model_concurrent_requests', 
    'Number of concurrent inference requests'
)

# 9. Model version (can use a constant label)
MODEL_INFO = Gauge(
    'model_info', 
    'Model information',
    ['version', 'algorithm']
)

# 10. Success rate (calculated as success/total, but we can track success count specifically)
SUCCESS_COUNT = Counter(
    'model_inference_success_total',
    'Total number of successful model inferences'
)
