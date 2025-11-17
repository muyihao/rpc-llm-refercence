# src/processor.py
"""
Processor 构造器，完全对齐官方 vLLMEngineProcessor 的写法。
"""

from typing import Any, Dict, Optional

import ray
from ray.data.block import UserDefinedFunction
from ray.llm._internal.batch.processor.base import Processor, ProcessorBuilder, DEFAULT_MAX_TASKS_IN_FLIGHT

from .config import AIRecordProcessorConfig
from .stage import RPCInferenceStage
from .registry import ServiceRegistry


def build_ai_record_processor(
    config: AIRecordProcessorConfig,
    preprocess: Optional[UserDefinedFunction] = None,
    postprocess: Optional[UserDefinedFunction] = None,
    preprocess_map_kwargs: Optional[Dict[str, Any]] = None,
    postprocess_map_kwargs: Optional[Dict[str, Any]] = None,
) -> Processor:
    """构造一个完全容错的 RPC 推理 Processor"""

    ray.init(ignore_reinit_error=True)

    # 创建或复用全局注册中心
    registry = config.registry_handle or ServiceRegistry.remote()
    if config.registry_handle is None:
        config.registry_handle = registry  # 方便外部获取

    stages = []

    # 核心 RPC 推理 Stage
    stages.append(
        RPCInferenceStage(
            fn_constructor_kwargs={
                "registry": registry,
                "service_type": config.service_type,
                "inference_kwargs": config.inference_kwargs,
                "request_timeout": config.request_timeout,
                "retry_count": config.retry_count,
                "load_balance_strategy": config.load_balance_strategy,
                "min_services_required": config.min_services_required,
                "service_check_interval": config.service_check_interval,
                "max_service_wait_time": config.max_service_wait_time,
                "service_cache_ttl": config.service_cache_ttl,
                "backoff_factor": config.backoff_factor,
            },
            map_batches_kwargs={
                "zero_copy_batch": True,
                "compute": ray.data.ActorPoolStrategy(
                    min_size=config.get_concurrency(autoscaling_enabled=False)[0],
                    max_size=config.get_concurrency(autoscaling_enabled=False)[1],
                    max_tasks_in_flight_per_actor=config.experimental.get(
                        "max_tasks_in_flight_per_actor", DEFAULT_MAX_TASKS_IN_FLIGHT
                    ),
                ),
                "concurrency": config.get_concurrency(),
                "batch_size": config.batch_size,
                "max_concurrency": config.max_concurrent_requests,
                "runtime_env": config.runtime_env,
            },
        )
    )

    return Processor(
        config=config,
        stages=stages,
        preprocess=preprocess,
        postprocess=postprocess,
        # preprocess_map_kwargs=preprocess_map_kwargs,
        # postprocess_map_kwargs=postprocess_map_kwargs,
    )


# 注册到 ProcessorBuilder，之后可以直接 ProcessorBuilder.build(AIRecordProcessorConfig(...))
ProcessorBuilder.register(AIRecordProcessorConfig, build_ai_record_processor)