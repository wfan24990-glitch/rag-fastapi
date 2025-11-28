import os
import aiohttp
import asyncio

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "doubao")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.model = os.getenv("LLM_MODEL", "doubao-seed-1-6-251015")

    async def generate(self, prompt: str, max_tokens: int = 512):
        if self.provider.lower() == "doubao":
            return await self._doubao_chat(prompt, max_tokens)
        else:
            raise ValueError("Unsupported LLM provider: " + self.provider)

    async def _doubao_chat(self, prompt: str, max_tokens: int):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "stream": False
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers, timeout=60) as resp:
                resp.raise_for_status()
                r = await resp.json()
                return r["choices"][0]["message"]["content"]
