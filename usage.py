import ray
from src.config import AIRecordProcessorConfig
from src.processor import build_ai_record_processor
from src.registry import ServiceRegistry
from typing import Any, Dict, Optional

@ray.remote
class InferenceActor:
    def ping(self):
        return "alive"

    def infer(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        response = f"Processed: {data.get('prompt', '')} with max_tokens={kwargs.get('max_tokens', 0)}"
        return {"response": response, **data}

if __name__ == "__main__":
    ray.init()

    registry = ServiceRegistry.remote()

    actor1 = InferenceActor.remote()
    actor2 = InferenceActor.remote()
    ray.get(registry.register_service.remote("ai_inference", actor1))
    ray.get(registry.register_service.remote("ai_inference", actor2))

    config = AIRecordProcessorConfig(
        batch_size=32,
        concurrency=2,
        service_type="ai_inference",
        registry_handle=registry,
        max_concurrent_requests=5,
        request_timeout=10.0,
        retry_count=2,
        load_balance_strategy="random",
        min_services_required=1,
        service_check_interval=0.5,
        max_service_wait_time=30.0,
        service_cache_ttl=10.0,
        backoff_factor=2.0,
        inference_kwargs={"max_tokens": 100},
    )

    processor = build_ai_record_processor(config)

    ds = ray.data.range(100).map(lambda x: {"prompt": f"Query {x}"})

    result_ds = processor(ds)

    for row in result_ds.iter_rows():
        print(row)

    ray.get(registry.unregister_service.remote("ai_inference", actor1))
    ray.get(registry.unregister_service.remote("ai_inference", actor2))
    print("Finish test!!")