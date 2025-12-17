"""Performance tests for batch inference endpoint."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.performance
@patch("src.api.routers.inference.InferenceService")
def test_batch_parallel_vs_sequential_performance(mock_service_class, client, api_key):
    """Benchmark parallel vs sequential batch processing. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import PredictionResult

    mock_model = MagicMock()
    mock_service = MagicMock()
    mock_service.load_model.return_value = mock_model

    batch_size = 50
    delay_per_item = 0.01

    def mock_predict_batch_parallel(model, features_list, concurrency=10, timeout=None):
        """Mock parallel batch processing with delay."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        def predict_single(index, features):
            time.sleep(delay_per_item)
            return (
                index,
                PredictionResult(predictions={"BaseSalary": {"p50": 100000.0}}, metadata={}),
            )

        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            from concurrent.futures import as_completed

            future_to_index = {
                executor.submit(predict_single, i, f): i for i, f in enumerate(features_list)
            }
            for future in as_completed(future_to_index):
                results.append(future.result())

        return sorted(results, key=lambda x: x[0])

    mock_service.predict_batch_parallel.side_effect = mock_predict_batch_parallel
    mock_service_class.return_value = mock_service

    features = [{"Level": "L4", "YearsOfExperience": i} for i in range(batch_size)]

    start_time = time.time()
    response = client.post(
        "/api/v1/models/test123/predict/batch",
        json={"features": features, "concurrency": 10},
        headers={"X-API-Key": api_key},
    )
    parallel_time = time.time() - start_time

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == batch_size
    assert data["success_count"] == batch_size

    expected_sequential_time = batch_size * delay_per_item
    expected_parallel_time = (batch_size / 10) * delay_per_item

    assert parallel_time < expected_sequential_time
    assert parallel_time < expected_parallel_time * 2


@pytest.mark.performance
@patch("src.api.routers.inference.InferenceService")
def test_batch_throughput_various_sizes(mock_service_class, client, api_key):
    """Test throughput with various batch sizes. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import PredictionResult

    mock_model = MagicMock()
    mock_service = MagicMock()
    mock_service.load_model.return_value = mock_model

    batch_sizes = [10, 50, 100, 500]
    results = {}

    for batch_size in batch_sizes:
        mock_service.predict_batch_parallel.return_value = [
            (i, PredictionResult(predictions={"BaseSalary": {"p50": 100000.0}}, metadata={}))
            for i in range(batch_size)
        ]

        start_time = time.time()
        response = client.post(
            "/api/v1/models/test123/predict/batch",
            json={"features": [{"Level": "L4"} for _ in range(batch_size)]},
            headers={"X-API-Key": api_key},
        )
        elapsed = time.time() - start_time

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == batch_size

        throughput = batch_size / elapsed if elapsed > 0 else float("inf")
        results[batch_size] = throughput

    assert all(throughput > 0 for throughput in results.values())


@pytest.mark.performance
@patch("src.api.routers.inference.InferenceService")
def test_batch_throughput_various_concurrency_levels(mock_service_class, client, api_key):
    """Test throughput with various concurrency levels. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.api.rate_limiting import BATCH_INFERENCE_CONCURRENCY
    from src.services.inference_service import PredictionResult

    mock_model = MagicMock()
    mock_service = MagicMock()
    mock_service.load_model.return_value = mock_model
    mock_service_class.return_value = mock_service

    batch_size = 100
    concurrency_levels = [1, 5, 10, 20]
    results = {}

    for concurrency in concurrency_levels:
        mock_service.predict_batch_parallel.return_value = [
            (i, PredictionResult(predictions={"BaseSalary": {"p50": 100000.0}}, metadata={}))
            for i in range(batch_size)
        ]

        start_time = time.time()
        response = client.post(
            "/api/v1/models/test123/predict/batch",
            json={
                "features": [{"Level": "L4"} for _ in range(batch_size)],
                "concurrency": concurrency,
            },
            headers={"X-API-Key": api_key},
        )
        elapsed = time.time() - start_time

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == batch_size

        throughput = batch_size / elapsed if elapsed > 0 else float("inf")
        results[concurrency] = throughput

        assert mock_service.predict_batch_parallel.called
        call_args = mock_service.predict_batch_parallel.call_args
        assert call_args is not None
        expected_concurrency = min(concurrency, BATCH_INFERENCE_CONCURRENCY)
        assert call_args[1]["concurrency"] == expected_concurrency

    assert all(throughput > 0 for throughput in results.values())


@pytest.mark.performance
@patch("src.api.routers.inference.InferenceService")
def test_batch_large_batch_performance(mock_service_class, client, api_key):
    """Test performance with large batch size. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import PredictionResult

    mock_model = MagicMock()
    mock_service = MagicMock()
    mock_service.load_model.return_value = mock_model

    batch_size = 1000
    mock_service.predict_batch_parallel.return_value = [
        (i, PredictionResult(predictions={"BaseSalary": {"p50": 100000.0}}, metadata={}))
        for i in range(batch_size)
    ]
    mock_service_class.return_value = mock_service

    start_time = time.time()
    response = client.post(
        "/api/v1/models/test123/predict/batch",
        json={"features": [{"Level": "L4"} for _ in range(batch_size)]},
        headers={"X-API-Key": api_key},
    )
    elapsed = time.time() - start_time

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == batch_size
    assert data["success_count"] == batch_size

    assert elapsed < 60
