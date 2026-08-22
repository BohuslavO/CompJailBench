import os
import random

from config import (
    LLM_PROVIDER,
    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_DEPLOYMENT, validate_azure_config,
    GEMINI_API_KEY, GEMINI_MODEL, validate_gemini_config,
)


class LLMClient:
    """Azure OpenAI backend."""

    def __init__(self):
        validate_azure_config()

        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
        self.model = AZURE_DEPLOYMENT

    def chat(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class GeminiClient:
    """Google AI Studio (Gemini) backend — free tier, no credit card
    required. Uses the current `google-genai` SDK
    (`pip install google-genai`, `from google import genai`).
    """

    def __init__(self):
        validate_gemini_config()

        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def chat(self, system_prompt, user_prompt):
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text


class MockLLMClient:
    """
    Stand-in for a real client that returns canned/deterministic responses
    instead of calling any API. Lets the whole pipeline (planner parsing,
    worker fan-out, aggregation, graph construction, both monitors, CSV
    output) be exercised and debugged with no network, no cost, and no key
    at all.

    It inspects the system_prompt to figure out which "role" is calling
    it (planner / worker / aggregator / grader) and returns a plausible
    fake response shaped the way that role expects, so downstream parsing
    code (e.g. PlannerAgent._parse_subtasks, the JSON graders in
    monitors.py) has something realistic to chew on.
    """

    def __init__(self):
        pass

    def chat(self, system_prompt, user_prompt):
        role = self._infer_role(system_prompt)

        if role == "planner":
            return self._mock_plan(system_prompt)
        if role == "worker":
            return self._mock_worker_output(user_prompt)
        if role == "aggregator":
            return self._mock_aggregation()
        if role == "grader":
            return self._mock_grader_json()
        return "This is a mock response. No real model was called."

    @staticmethod
    def _infer_role(system_prompt):
        p = system_prompt.lower()
        if "planning agent" in p:
            return "planner"
        if "worker agent" in p:
            return "worker"
        if "aggregation agent" in p:
            return "aggregator"
        if "safety evaluator" in p:
            return "grader"
        return "generic"

    @staticmethod
    def _mock_plan(user_prompt):
        import re
        match = re.search(r"exactly (\d+) independent subtasks", user_prompt)
        n = int(match.group(1)) if match else 2
        lines = [f"{i+1}. Mock subtask {i+1} derived from the task" for i in range(n)]
        return "\n".join(lines)

    @staticmethod
    def _mock_worker_output(user_prompt):
        return f"[MOCK WORKER OUTPUT] Addressed subtask: {user_prompt[:60]}..."

    @staticmethod
    def _mock_aggregation():
        return "[MOCK AGGREGATED OUTPUT] Combined response drawing on all worker outputs."

    @staticmethod
    def _mock_grader_json():
        completed = random.choice([True, False])
        return (
            '{"completed": %s, "refused": %s, "leaked": %s, '
            '"confidence": %.2f, "reasoning": "mock grader output"}'
            % (str(completed).lower(), str(not completed).lower(),
               str(completed).lower(), random.uniform(0.4, 0.9))
        )


def get_client():
    """Factory used everywhere instead of instantiating a client class
    directly. Routes based on LLM_PROVIDER (set in .env):
        COMPJAILBENCH_MOCK=1  -> always wins, forces MockLLMClient
        LLM_PROVIDER=gemini   -> GeminiClient (free, no card)
        LLM_PROVIDER=azure    -> LLMClient (Azure OpenAI)
    """
    if os.environ.get("COMPJAILBENCH_MOCK") == "1":
        return MockLLMClient()

    provider = LLM_PROVIDER.lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "azure":
        return LLMClient()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Expected 'gemini' or 'azure' "
        "(or set COMPJAILBENCH_MOCK=1 to skip real calls entirely)."
    )