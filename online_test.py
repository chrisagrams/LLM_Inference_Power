import argparse
import asyncio
import ast
import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

@dataclass
class TimingInfo:
    """Container for timing information from the API response"""
    prompt_per_token_ms: Optional[float] = None
    predicted_ms: Optional[float] = None
    prompt_ms: Optional[float] = None
    prompt_per_second: Optional[float] = None
    predicted_n: Optional[int] = None
    predicted_per_token_ms: Optional[float] = None
    predicted_per_second: Optional[float] = None
    prompt_n: Optional[int] = None

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    idx: int
    time_s: float
    prompt_tokens: int
    completion_tokens: int
    timing_info: TimingInfo
    error: Optional[str] = None
    
    # Computed fields
    @property
    def computed_tokens_per_sec(self) -> Optional[float]:
        """Compute tokens/second based on our measured time"""
        if self.completion_tokens and self.time_s > 0:
            return self.completion_tokens / self.time_s
        return None

def load_dataset(path: str) -> List[List[Dict[str, str]]]:
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line looks like: [{'content': '...', 'role': 'user'}]
            item = ast.literal_eval(line)
            if not isinstance(item, list):
                raise ValueError("Each dataset line must be a list of message dicts.")
            msgs.append(item)
    return msgs

def load_lengths(path: Optional[str], n: int) -> List[Optional[int]]:
    if not path:
        return [None] * n
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(int(s))
    if len(out) < n:
        out.extend([None] * (n - len(out)))
    return out[:n]

def extract_timing_info(resp: ChatCompletion) -> TimingInfo:
    """Extract timing information from the API response"""
    # Try to get timings from the response
    timings_data = {}
    
    # Check if the response has timing information in the expected format
    if hasattr(resp, 'timings') and resp.timings:
        timings_data = resp.timings
    elif isinstance(resp.model_extra, dict) and 'timings' in resp.model_extra:
        timings_data = resp.model_extra['timings']
    
    return TimingInfo(
        prompt_per_token_ms=timings_data.get('prompt_per_token_ms'),
        predicted_ms=timings_data.get('predicted_ms'),
        prompt_ms=timings_data.get('prompt_ms'),
        prompt_per_second=timings_data.get('prompt_per_second'),
        predicted_n=timings_data.get('predicted_n'),
        predicted_per_token_ms=timings_data.get('predicted_per_token_ms'),
        predicted_per_second=timings_data.get('predicted_per_second'),
        prompt_n=timings_data.get('prompt_n'),
    )

async def do_one(
    client: AsyncOpenAI,
    idx: int,
    messages: List[Dict[str, str]],
    model: str,
    timeout: float,
    max_output_tokens: Optional[int],
    temperature: float,
    base_system: Optional[str],
) -> BenchmarkResult:
    req_messages = []
    if base_system:
        req_messages.append({"role": "system", "content": base_system})
    req_messages.extend(messages)

    params: Dict[str, Any] = {
        "model": model,
        "messages": req_messages,
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        params["max_tokens"] = max_output_tokens

    t0 = time.perf_counter()
    try:
        resp: ChatCompletion = await asyncio.wait_for(
            client.chat.completions.create(**params), timeout=timeout
        )
    except Exception as e:
        return BenchmarkResult(
            idx=idx,
            time_s=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            timing_info=TimingInfo(),
            error=f"{type(e).__name__}: {e}"
        )
    
    dt = time.perf_counter() - t0

    prompt_tokens = getattr(resp.usage, "prompt_tokens", None) or 0
    completion_tokens = getattr(resp.usage, "completion_tokens", None) or 0
    timing_info = extract_timing_info(resp)
    
    return BenchmarkResult(
        idx=idx,
        time_s=dt,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        timing_info=timing_info,
        error=None
    )

def get_csv_headers() -> List[str]:
    """Get the CSV headers for all fields we want to export"""
    return [
        "index",
        "time_s",
        "prompt_tokens", 
        "completion_tokens",
        "computed_tokens_per_sec",
        "prompt_per_token_ms",
        "predicted_ms", 
        "prompt_ms",
        "prompt_per_second",
        "predicted_n",
        "predicted_per_token_ms",
        "predicted_per_second", 
        "prompt_n",
        "error"
    ]

def result_to_csv_row(result: BenchmarkResult) -> List[str]:
    """Convert a BenchmarkResult to a CSV row"""
    def format_float(val: Optional[float]) -> str:
        return f"{val:.6f}" if val is not None else ""
    
    def format_int(val: Optional[int]) -> str:
        return str(val) if val is not None else ""
    
    return [
        str(result.idx),
        f"{result.time_s:.6f}",
        str(result.prompt_tokens),
        str(result.completion_tokens),
        format_float(result.computed_tokens_per_sec),
        format_float(result.timing_info.prompt_per_token_ms),
        format_float(result.timing_info.predicted_ms),
        format_float(result.timing_info.prompt_ms),
        format_float(result.timing_info.prompt_per_second),
        format_int(result.timing_info.predicted_n),
        format_float(result.timing_info.predicted_per_token_ms),
        format_float(result.timing_info.predicted_per_second),
        format_int(result.timing_info.prompt_n),
        result.error or ""
    ]

async def run(
    dataset_path: str,
    lengths_path: Optional[str],
    model: str,
    batch_size: int,
    timeout: float,
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    csv_out: Optional[str],
    base_system: Optional[str],
) -> None:
    client = AsyncOpenAI(base_url=base_url or None, api_key=api_key or os.getenv("OPENAI_API_KEY"))
    dataset = load_dataset(dataset_path)
    lengths = load_lengths(lengths_path, len(dataset))

    sem = asyncio.Semaphore(batch_size)
    results: List[BenchmarkResult] = []
    
    # Initialize CSV file if output path is provided
    csv_file = None
    csv_writer = None
    if csv_out:
        csv_file = open(csv_out, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(get_csv_headers())
        csv_file.flush()  # Ensure headers are written immediately

    async def wrapped(i: int):
        async with sem:
            result = await do_one(
                client=client,
                idx=i,
                messages=dataset[i],
                model=model,
                timeout=timeout,
                max_output_tokens=lengths[i],
                temperature=temperature,
                base_system=base_system,
            )
            results.append(result)
            
            # Write result to CSV immediately if writer is available
            if csv_writer:
                csv_writer.writerow(result_to_csv_row(result))
                csv_file.flush()  # Ensure data is written to disk immediately
            
            tps_str = f"{result.computed_tokens_per_sec:.2f}" if result.computed_tokens_per_sec else "N/A"
            print(f"[{result.idx}] Time: {result.time_s:.3f}s, Tokens: {result.completion_tokens}, "
                  f"TPS: {tps_str}, Error: {result.error or 'None'}")

    try:
        tasks = [asyncio.create_task(wrapped(i)) for i in range(len(dataset))]
        await asyncio.gather(*tasks)
    finally:
        # Close CSV file if it was opened
        if csv_file:
            csv_file.close()

    results.sort(key=lambda x: x.idx)
    
    # Print summary
    headers = get_csv_headers()
    print("\nResults Summary:")
    print("\t".join(headers))
    
    total_completion_tokens = 0
    total_time = 0.0
    successful_results = [r for r in results if r.error is None]
    
    for result in results:
        if result.error is None:
            total_completion_tokens += result.completion_tokens
            total_time += result.time_s
    
    if total_time > 0 and successful_results:
        overall_tps = total_completion_tokens / total_time
        print(f"\nOverall Stats:")
        print(f"  Successful requests: {len(successful_results)}/{len(results)}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Overall tokens/s: {overall_tps:.3f}")
        
        # Show API-reported vs computed metrics comparison
        api_tps_values = [r.timing_info.predicted_per_second for r in successful_results if r.timing_info.predicted_per_second]
        if api_tps_values:
            avg_api_tps = sum(api_tps_values) / len(api_tps_values)
            print(f"  Average API-reported tokens/s: {avg_api_tps:.3f}")
    else:
        print("\nNo successful timings recorded.")

    # CSV was already written during execution
    if csv_out:
        print(f"\nResults written to: {csv_out}")

def main():
    p = argparse.ArgumentParser(description="Concurrent Chat benchmarking with OpenAI-compatible API.")
    p.add_argument("--dataset", default="inputs_chat_format.txt", help="Path to dataset .txt (one list of {role,content} per line).")
    p.add_argument("--lengths", default="out_lenghts.txt", help="Path to expected output lengths; one integer per line used as max_tokens.")
    p.add_argument("--model", required=True, help="Model name.")
    p.add_argument("--batch-size", type=int, default=1, help="Number of concurrent requests.")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds.")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--base-url", help="OpenAI-compatible base URL (e.g., http://localhost:11434/v1).")
    p.add_argument("--api-key", default="1234", help="API key; defaults to OPENAI_API_KEY env.")
    p.add_argument("--csv-out", help="Optional CSV output path.")
    p.add_argument("--system", help="Optional system message to prepend.")
    args = p.parse_args()

    asyncio.run(
        run(
            dataset_path=args.dataset,
            lengths_path=args.lengths,
            model=args.model,
            batch_size=args.batch_size,
            timeout=args.timeout,
            temperature=args.temperature,
            base_url=args.base_url,
            api_key=args.api_key,
            csv_out=args.csv_out,
            base_system=args.system,
        )
    )

if __name__ == "__main__":
    main()