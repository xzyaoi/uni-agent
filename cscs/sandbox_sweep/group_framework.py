"""Bound entire GRPO groups while retaining the existing episode runner."""
import asyncio
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from uni_agent.framework import framework as _framework

_BaseFramework = getattr(_framework, 'GatewayAgentFramework', None)
if _BaseFramework is None:
    _BaseFramework = _framework.OpenAICompatibleAgentFramework
if not any(hasattr(_BaseFramework, name) for name in ('_run_prompt_rollouts', '_run_prompt_sessions_to_tq')):
    raise ImportError('Unsupported framework group-dispatch API')


class GroupLimitedGatewayAgentFramework(_BaseFramework):
    @classmethod
    def from_config(cls, *, config, **kwargs):
        instance = super().from_config(config=config, **kwargs)
        instance._group_limit = int(OmegaConf.select(
            config, 'actor_rollout_ref.rollout.custom.agent_framework.max_concurrent_groups'))
        if instance._group_limit <= 0:
            raise ValueError('max_concurrent_groups must be positive')
        instance._group_semaphore = None
        instance._groups_active = instance._groups_terminal = 0
        return instance

    def _record_group(self, event, uid):
        path = Path(os.environ['UNI_AGENT_TRACE_DIR']) / ('group-status-%s.jsonl' % os.getpid())
        record = {'kind': 'pipeline', 'schema_version': 1, 'timestamp': time.time(), 'event': event, 'uid': str(uid),
                  'active_groups': self._groups_active, 'terminal_groups': self._groups_terminal,
                  'group_limit': self._group_limit}
        with path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    async def _run_prompt_rollouts(self, *, sample_fields, **kwargs):
        return await self._run_limited_group('_run_prompt_rollouts', sample_fields, kwargs)

    async def _run_prompt_sessions_to_tq(self, *, sample_fields, **kwargs):
        return await self._run_limited_group('_run_prompt_sessions_to_tq', sample_fields, kwargs)

    async def _run_limited_group(self, method, sample_fields, kwargs):
        if self._group_semaphore is None:
            self._group_semaphore = asyncio.Semaphore(self._group_limit)
        async with self._group_semaphore:
            self._groups_active += 1
            self._record_group('group_begin', sample_fields['uid'])
            settled = False
            try:
                result = await getattr(super(), method)(sample_fields=sample_fields, **kwargs)
                settled = True
                return result
            finally:
                self._groups_active -= 1
                self._groups_terminal += int(settled)
                self._record_group('group_terminal' if settled else 'group_interrupted', sample_fields['uid'])
