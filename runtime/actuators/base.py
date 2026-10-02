"""Base class every actuator implements. The dispatcher only calls handle(); device specifics stay in the subclass."""
from __future__ import annotations


class Actuator:
    name = "actuator"
    kind = "emulated"        # "real" | "emulated"

    async def handle(self, cmd, state):
        """cmd: runtime.command.Command, state: runtime.state.State.
        Mutate state.data[<own key>] and return a runtime.command.Response."""
        raise NotImplementedError

    def status(self) -> dict:
        return {"name": self.name, "kind": self.kind}

    def cancel(self) -> None:
        """stop background tasks (shutdown / tests)"""
