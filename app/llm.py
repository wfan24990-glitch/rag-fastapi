import os
import aiohttp
from app.llm_client import LLMClient
from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

_client = LLMClient()

async def generate_local(system_prompt: str, user_prompt: str) -> str:
    """
    Generate text using the configured local (or primary) LLM provider.
    """
    # Concatenate system and user prompt as the current client interface 
    # primarily handles a single prompt string.
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    return await _client.generate(full_prompt)

async def generate_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Fallback generation using OpenAI.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in configuration.")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers, timeout=60) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"OpenAI API Error: {resp.status} - {error_text}")
            r = await resp.json()
            return r["choices"][0]["message"]["content"]
