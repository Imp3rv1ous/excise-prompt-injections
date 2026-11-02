"""LlmClient that keeps the token usage and error of every call, so each run record can report them."""

from excise.agent.framework.llm import LlmClient


class CountingClient(LlmClient):
    def __init__(self, model, **config):
        super().__init__(model, **config)
        self.usage = []
        self.errors = []

    async def generate(self, request):
        response = await super().generate(request)
        self.usage.append(response.usage_metadata)
        if response.error_message:
            self.errors.append(response.error_message)
        return response
