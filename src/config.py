from typing import Optional, Dict, Any

from ray.llm._internal.batch.processor.base import ProcessorConfig
from pydantic import Field
from ray.actor import ActorHandle


class AIRecordProcessorConfig(ProcessorConfig):
    """The configuration for the AI Record Processor."""

    service_type: str = Field(default="ai_inference", description="The type of service to use for RPC inference requests.")
    registry_handle: Optional[ActorHandle] = Field(default=None, description="Handle to the global ServiceRegistry actor. If None, a new one will be created.")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent RPC requests per processing actor.")
    request_timeout: Optional[float] = Field(default=None, description="Timeout for each RPC request in seconds.")
    inference_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional kwargs to pass to the inference RPC method.")
    retry_count: int = Field(default=3, description="Number of retries for failed RPC requests.")
    load_balance_strategy: str = Field(default="round_robin", description="Load balancing strategy: 'round_robin' or 'random'.")
    min_services_required: int = Field(default=1, description="Minimum number of services required to proceed. If fewer, will wait and retry.")
    service_check_interval: float = Field(default=1.0, description="Interval in seconds to wait before rechecking services if insufficient.")
    max_service_wait_time: float = Field(default=60.0, description="Maximum time in seconds to wait for sufficient services before raising error.")
    service_cache_ttl: float = Field(default=5.0, description="TTL for caching service list in seconds for performance.")
    backoff_factor: float = Field(default=1.5, description="Exponential backoff factor for retries and waits.")
    max_concurrent_batches: int = Field(
        default=8,
        description="The maximum number of concurrent batches in the engine."
    )
    runtime_env: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The runtime environment to use for the processing."
    )