"""
SpeakAI — Service Registry

Maps session_id → AIService. Thread-safe via asyncio.
Extracted for clean separation from the AIService itself.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("speakai.registry")


class ServiceRegistry:
    """Maps session_id → AIService. Thread-safe via asyncio."""

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}

    def create(
        self,
        session_id: str,
        on_metrics: Optional[Callable] = None,
        on_feedback: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
        on_chat: Optional[Callable] = None,
        on_transcript: Optional[Callable] = None,
        on_conversation_state: Optional[Callable] = None,
    ) -> Any:
        from .ai_service import AIService

        service = AIService(
            session_id=session_id,
            on_metrics=on_metrics,
            on_feedback=on_feedback,
            on_status=on_status,
            on_chat=on_chat,
            on_transcript=on_transcript,
            on_conversation_state=on_conversation_state,
        )
        self._services[session_id] = service
        logger.info(f"ServiceRegistry: created {session_id} (total: {len(self._services)})")
        return service

    async def stop_service(self, session_id: str) -> Optional[Dict[str, Any]]:
        service = self._services.pop(session_id, None)
        if service:
            summary = await service.stop()
            logger.info(f"ServiceRegistry: removed {session_id} (total: {len(self._services)})")
            return summary
        return None

    async def stop_all(self) -> None:
        for sid in list(self._services.keys()):
            await self.stop_service(sid)

    def get(self, session_id: str) -> Optional[Any]:
        return self._services.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._services)

    @property
    def all_services(self) -> Dict[str, Any]:
        return dict(self._services)
