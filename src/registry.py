from typing import Dict, List
import ray
from ray.actor import ActorHandle
import time

@ray.remote
class ServiceRegistry:
    """A global Ray Actor serving as a registry for AI inference services with health checks."""

    def __init__(self):
        self._services: Dict[str, List[ActorHandle]] = {}  # Key: service_type, Value: List of ActorHandles
        self._health: Dict[ActorHandle, float] = {}  # Last health check time

    def register_service(self, service_type: str, actor_handle: ActorHandle):
        """Register an inference service actor."""
        if service_type not in self._services:
            self._services[service_type] = []
        self._services[service_type].append(actor_handle)
        self._health[actor_handle] = time.time()

    def get_services(self, service_type: str) -> List[ActorHandle]:
        """Get list of available inference actors, removing dead ones."""
        if service_type not in self._services:
            return []
        live_services = []
        for actor in self._services[service_type][:]:
            try:
                ray.get(actor.ping.remote(), timeout=1.0)  # Assume actors have a ping method
                live_services.append(actor)
            except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                self.unregister_service(service_type, actor)
        return live_services

    def unregister_service(self, service_type: str, actor_handle: ActorHandle):
        """Unregister an inference service actor."""
        if service_type in self._services:
            self._services[service_type] = [a for a in self._services[service_type] if a != actor_handle]
        self._health.pop(actor_handle, None)