# src/stage.py
"""
RPC-based inference stage.
完全模仿 ray.llm._internal.batch.stages.vllm_engine_stage 的实现方式，
但适配自定义 RPC 配置（通过 **kwargs 捕获用户参数）。
"""

from typing import Any, Dict, List, AsyncIterator, Optional, Type
import asyncio
import itertools
import random
import time
from ray.llm._internal.batch.stages.base import StatefulStageUDF, StatefulStage
from ray.actor import ActorHandle
import ray  # 确保导入 ray 以使用 ray.get


class RPCInferenceStageUDF(StatefulStageUDF):
    """
    真正的 UDF 实现，所有推理逻辑都在这里。
    通过 **kwargs 捕获所有用户配置（registry 等），忽略 Ray 自动注入的参数。
    """

    def __init__(
        self,
        data_column: str,
        expected_input_keys: Optional[List[str]] = None,
        **kwargs  # 捕获所有 fn_constructor_kwargs（如 registry, timeout 等）
    ):
        # 调用父类，只传必需参数，忽略多余的
        super().__init__(data_column, expected_input_keys)
        # 存储用户自定义配置，用于 udf
        self.user_config = kwargs

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Async UDF：处理一批数据，通过 RPC 分发到注册的 Actor。
        支持动态服务发现、重试、缓存、指数退避、负载均衡。
        """
        # ---------- 从 user_config 取出配置 ----------
        registry: ActorHandle = self.user_config["registry"]
        service_type: str = self.user_config["service_type"]
        inference_kwargs: Dict[str, Any] = self.user_config["inference_kwargs"]
        request_timeout: float = self.user_config.get("request_timeout", None)
        retry_count: int = self.user_config.get("retry_count", 3)
        load_balance: str = self.user_config.get("load_balance_strategy", "round_robin")
        min_services: int = self.user_config.get("min_services_required", 1)
        check_interval: float = self.user_config.get("service_check_interval", 1.0)
        max_wait: float = self.user_config.get("max_service_wait_time", 60.0)
        cache_ttl: float = self.user_config.get("service_cache_ttl", 5.0)
        backoff_factor: float = self.user_config.get("backoff_factor", 1.5)

        # ---------- 服务发现 + 缓存 + 健康检查 ----------
        _cached_services: List[ActorHandle] = []
        _cache_ts = 0.0

        def get_live_services() -> List[ActorHandle]:
            nonlocal _cached_services, _cache_ts
            now = time.time()
            if now - _cache_ts < cache_ttl and _cached_services:
                return _cached_services
            # 从注册中心获取（带健康检查）
            services = ray.get(registry.get_services.remote(service_type))
            # 简单健康检查：尝试 ping（假设 Actor 有 ping 方法）
            live_services = []
            for service in services:
                try:
                    ray.get(service.ping.remote() if hasattr(service, 'ping') else True, timeout=1.0)
                    live_services.append(service)
                except Exception:
                    # 自动移除死 Actor（注册中心会处理 unregister）
                    ray.get(registry.unregister_service.remote(service_type, service))
            _cached_services = live_services
            _cache_ts = now
            return live_services

        # 等待至少 min_services 可用（指数退避）
        wait_time = check_interval
        start_time = time.time()
        services = get_live_services()
        while len(services) < min_services:
            if time.time() - start_time > max_wait:
                raise RuntimeError(
                    f"Timeout waiting for >= {min_services} live services (got {len(services)}) "
                    f"for type '{service_type}' after {max_wait}s"
                )
            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * backoff_factor, max_wait / 2)  # 上限避免无限等待
            services = get_live_services()

        if not services:
            raise RuntimeError(f"No live services available for '{service_type}'")

        # ---------- 负载均衡提交 RPC ----------
        futures = []
        if load_balance == "round_robin":
            cycle = itertools.cycle(services)
            for row in batch:
                service = next(cycle)
                future = service.infer.remote(row, **inference_kwargs)
                futures.append(future)
        elif load_balance == "random":
            for row in batch:
                service = random.choice(services)
                future = service.infer.remote(row, **inference_kwargs)
                futures.append(future)
        else:
            raise ValueError(f"Unknown load_balance: {load_balance}")

        # ---------- 收集结果：per-task 重试 + 动态刷新服务 ----------
        for idx, future in enumerate(futures):
            row = batch[idx]
            attempts = retry_count + 1
            current_future = future
            wait_time = check_interval

            while attempts > 0:
                try:
                    result = ray.get(current_future, timeout=request_timeout)
                    yield result  # 异步 yield，保持顺序
                    break
                except Exception as exc:
                    attempts -= 1
                    if attempts == 0:
                        raise RuntimeError(f"Row {idx} exhausted {retry_count} retries: {exc}") from exc

                    # 重试前：短暂等待 + 刷新服务（处理动态增减/故障）
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * backoff_factor, max_wait / 2)
                    services = get_live_services()
                    if not services:
                        raise RuntimeError("All services failed/disappeared during retry")

                    # 重新选择服务
                    chosen = random.choice(services) if load_balance == "random" else next(itertools.cycle(services))
                    current_future = chosen.infer.remote(row, **inference_kwargs)


class RPCInferenceStage(StatefulStage):
    """
    Stage 配置：fn 指向自定义 UDF。
    """
    fn: Type[StatefulStageUDF] = RPCInferenceStageUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        """定义必需输入列（类似 vLLM）。"""
        return {"prompt": "The text prompt for inference (str)."}

    def get_optional_input_keys(self) -> Dict[str, str]:
        """可选输入列。"""
        return {
            "context": "Additional context for the inference request.",
            "metadata": "Any metadata to pass through to the RPC actor.",
        }