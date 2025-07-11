import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, AsyncIterator
from tqdm.asyncio import tqdm as async_tqdm

from ..dataset_utils import UnifiedDatasetInterface

logger = logging.getLogger(__name__)


class OpenAIExecutor:
    """
    Executor for running evaluations on OpenAI-compatible APIs.
    """
    def __init__(self,
                 model: str,
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 concurrency: int = 10,
                 timeout: int = 120,
                 max_retries: int = 3):
        self.model_config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        self.api_key = api_key
        self.base_url = base_url
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.openai_url = f"{base_url.rstrip('/')}/chat/completions"

    async def run(self, dataset_interface: UnifiedDatasetInterface, dataset_path: str, topk: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generates prompts, sends requests asynchronously, and returns parsed results.
        """
        logger.info("Loading dataset...")
        all_data = dataset_interface.load_dataset_shard(dataset_path, 0, 1)
        logger.info(f"Loaded {len(all_data)} entries")

        all_requests, original_questions_map = self._prepare_requests(
            all_data, dataset_interface, batch_size, topk
        )
        
        logger.info(f"Generated {len(all_requests)} prompts. Sending requests to API...")
        
        raw_results = []
        async for result in async_tqdm(
            self._process_batch_requests(all_requests),
            total=len(all_requests),
            desc="Processing API requests"
        ):
            raw_results.append(self._parse_openai_response(result))
            
        logger.info(f"Completed {len(raw_results)} requests")
        
        processed_results = self._post_process(raw_results, original_questions_map)
        logger.info(f"Successfully processed {len(processed_results)} entries")
        
        return processed_results, raw_results

    def _prepare_requests(self, all_data: List[Dict], dataset_interface: UnifiedDatasetInterface, batch_size: int, topk: int):
        all_requests = []
        original_questions_map = {}
        
        logger.info("Generating prompts in batches...")
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            for entry in batch:
                entry_id = entry["id"]
                prompt = dataset_interface.dataset.apply_template(entry["data"], dataset_interface.template_provider)
                
                all_requests.append({
                    "id": entry_id,
                    "payload": self._build_openai_request_payload(prompt)
                })
                original_questions_map[entry_id] = entry["data"]
        return all_requests, original_questions_map

    def _build_openai_request_payload(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model_config["model"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.model_config["temperature"],
            "max_tokens": self.model_config["max_tokens"]
        }

    async def _process_batch_requests(self, requests_data: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        headers = { "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json" }
        connector = aiohttp.TCPConnector(limit=self.concurrency + 50, limit_per_host=self.concurrency)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async with aiohttp.ClientSession(connector=connector, headers=headers, timeout=timeout) as session:
            tasks = [
                self._single_request(session, req["payload"], req["id"], semaphore)
                for req in requests_data
            ]
            for f in asyncio.as_completed(tasks):
                yield await f

    async def _single_request(self, session: aiohttp.ClientSession, payload: Dict[str, Any], request_id: Any, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        initial_backoff = 1.0
        async with semaphore:
            for attempt in range(self.max_retries + 1):
                last_err, status_code_for_error = None, None
                try:
                    async with session.post(self.openai_url, json=payload) as response:
                        if response.status == 200:
                            return {"id": request_id, "success": True, "response": await response.json(), "status_code": response.status}
                        
                        status_code_for_error = response.status
                        if 400 <= response.status < 500 and response.status != 429:
                            response_data = await response.json()
                            logger.warning(f"Request {request_id} failed with non-retryable status {response.status}: {response_data.get('error', {}).get('message', 'Unknown error')}")
                            return {"id": request_id, "success": False, "error": response_data, "status_code": response.status}
                        
                        last_err = f"HTTP status {response.status}: {await response.text()}"
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                    last_err = f"Connection error: {str(e)}"
                except Exception as e:
                    logger.error(f"Request {request_id} encountered an unexpected error: {str(e)}")
                    return {"id": request_id, "success": False, "error": f"Unexpected error: {str(e)}", "status_code": None}

                if attempt < self.max_retries:
                    backoff = initial_backoff * (2 ** attempt)
                    logger.warning(f"Request {request_id} failed (attempt {attempt + 1}/{self.max_retries + 1}): {last_err}. Retrying in {backoff:.2f}s.")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Request {request_id} failed after {self.max_retries + 1} attempts. Last error: {last_err}")
                    return {"id": request_id, "success": False, "error": f"Exhausted all retries. Last error: {last_err}", "status_code": status_code_for_error}
        return {"id": request_id, "success": False, "error": "Exited retry loop unexpectedly", "status_code": None}

    def _parse_openai_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        output_item = {"id": api_response["id"], "success": api_response["success"]}
        if not api_response["success"]:
            output_item["error_details"] = api_response.get("error", "Unknown error")
            output_item["status_code"] = api_response.get("status_code")
            output_item["domains"] = None
        else:
            try:
                content = api_response["response"]["choices"][0]["message"]["content"].strip()
                output_item["domains"] = [d.strip() for d in content.split(",") if d.strip()]
                output_item["raw_api_response"] = api_response["response"]
            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Error parsing successful response for ID {api_response['id']}: {e}. API Response: {api_response.get('response')}")
                output_item["success"] = False
                output_item["error_details"] = f"Parsing error: {str(e)}"
                output_item["domains"] = None
        return output_item

    def _post_process(self, results: List[Dict[str, Any]], original_questions_map: Dict[Any, str]) -> List[Dict[str, Any]]:
        processed = []
        for r in results:
            question = original_questions_map.get(r.get("id"))
            if question is None:
                logger.warning(f"Could not find original question for ID {r.get('id')}. Skipping.")
                continue
            if r.get("success"):
                processed.append({"id": r["id"], "question": question, "domains": r.get("domains")})
        processed.sort(key=lambda x: x["id"])
        return processed 