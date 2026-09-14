import re
import io
import json
import urllib.request
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

MENTION_REGEX = re.compile(r"@[A-Za-z0-9_]+")
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_tweet_text(text: str) -> Dict[str, Any]:
    """
    Clean tweet text by stripping @mentions and URLs, but keep them logged
    for traceability.
    """
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""

    raw_text = text.strip()
    mentions = MENTION_REGEX.findall(raw_text)
    urls = URL_REGEX.findall(raw_text)

    # Clean text: remove mentions and URLs, normalize whitespace
    cleaned = MENTION_REGEX.sub("", raw_text)
    cleaned = URL_REGEX.sub("", cleaned)
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    return {
        "raw_text": raw_text,
        "clean_text": cleaned,
        "mentions": mentions,
        "urls": urls,
    }


def download_subsample_stream(
    brand_handle: str = config.BRAND_HANDLE,
    output_path: Path = config.RAW_DATA_PATH,
    chunk_bytes: int = 50_000_000,
) -> Path:
    """
    Stream a chunk of the Kaggle TWCS dataset directly from the public mirror,
    filtering for tweets where author is the brand or tweets that reference the brand.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Raw subsample already exists at {output_path}")
        return output_path

    url = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
    print(f"Streaming {chunk_bytes // (1024*1024)}MB of twcs.csv to extract @{brand_handle} subsample...")
    
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{chunk_bytes}"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    text = content.decode("utf-8", errors="ignore")
    # Truncate at last newline to avoid partial row parsing errors
    last_nl = text.rfind("\n")
    if last_nl != -1:
        text = text[:last_nl]

    df = pd.read_csv(io.StringIO(text), low_memory=False)
    
    # Filter for brand tweets and customer inbound tweets addressing the brand
    brand_mask = df["author_id"].astype(str).str.lower() == brand_handle.lower()
    inbound_mask = df["text"].astype(str).str.contains(f"@{brand_handle}", case=False, na=False)
    
    subsample_df = df[brand_mask | inbound_mask].copy()
    
    # Also include any parent/child tweets referenced by these tweets if present in the chunk
    referenced_ids = set()
    for col in ["in_response_to_tweet_id", "response_tweet_id"]:
        for val in subsample_df[col].dropna():
            for tid in str(val).split(","):
                tid = tid.strip()
                if tid:
                    referenced_ids.add(tid)
                    
    extended_mask = df["tweet_id"].astype(str).isin(referenced_ids)
    final_df = df[brand_mask | inbound_mask | extended_mask].drop_duplicates(subset=["tweet_id"])
    
    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_df)} tweets for brand @{brand_handle} to {output_path}")
    return output_path


def reconstruct_threads(
    df: pd.DataFrame,
    brand_handle: str = config.BRAND_HANDLE
) -> List[Dict[str, Any]]:
    """
    Reconstruct conversation threads using reply-chain fields
    (in_response_to_tweet_id and response_tweet_id).
    
    Splits each thread into customer messages, brand replies, and classifies
    thread outcome:
      - brand_replied: Did the brand reply?
      - subsequent_customer_complaint: Was there customer follow-up after brand reply?
      - went_quiet_after_brand: Did the thread go quiet after brand reply? (Noisy resolution proxy)
    """
    brand_lower = brand_handle.lower()
    tweet_dict = {}

    # Standardize string representations
    for _, row in df.iterrows():
        tid = str(row["tweet_id"]).strip()
        author = str(row.get("author_id", "")).strip()
        inbound = bool(row.get("inbound", False))
        text = str(row.get("text", ""))
        in_reply_to = str(row.get("in_response_to_tweet_id", "")).strip()
        in_reply_to = in_reply_to if in_reply_to not in ["", "nan", "None"] else None
        
        resp_raw = str(row.get("response_tweet_id", "")).strip()
        responses = [r.strip() for r in resp_raw.split(",") if r.strip() and r.strip() not in ["nan", "None"]]
        
        cleaned_info = clean_tweet_text(text)
        
        tweet_dict[tid] = {
            "tweet_id": tid,
            "author_id": author,
            "inbound": inbound,
            "is_brand": (author.lower() == brand_lower),
            "raw_text": cleaned_info["raw_text"],
            "clean_text": cleaned_info["clean_text"],
            "mentions": cleaned_info["mentions"],
            "urls": cleaned_info["urls"],
            "created_at": str(row.get("created_at", "")),
            "in_reply_to": in_reply_to,
            "responses": responses,
        }

    # Find root tweets (inbound customer messages that are either starters or whose parent isn't in slice)
    root_tweet_ids = []
    for tid, t in tweet_dict.items():
        if t["inbound"] and not t["is_brand"]:
            if t["in_reply_to"] is None or t["in_reply_to"] not in tweet_dict:
                root_tweet_ids.append(tid)

    threads = []
    visited_tweets = set()

    for root_id in root_tweet_ids:
        thread_turns = []
        queue = [root_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id not in tweet_dict or curr_id in visited_tweets:
                continue
            visited_tweets.add(curr_id)
            node = tweet_dict[curr_id]
            thread_turns.append(node)

            for child_id in node["responses"]:
                if child_id in tweet_dict and child_id not in visited_tweets:
                    queue.append(child_id)

        if not thread_turns:
            continue

        # Split thread into customer turns and brand turns
        customer_turns = [turn for turn in thread_turns if not turn["is_brand"]]
        brand_turns = [turn for turn in thread_turns if turn["is_brand"]]

        # Determine thread outcome & resolution proxy
        brand_replied = len(brand_turns) > 0
        
        # Check turn order to see if customer spoke AFTER the brand's final reply
        further_customer_complaint = False
        went_quiet_after_brand = False
        
        if brand_replied:
            # Find index of the first and last brand turn
            brand_turn_indices = [i for i, turn in enumerate(thread_turns) if turn["is_brand"]]
            last_brand_idx = brand_turn_indices[-1]
            
            # Any customer turn after the brand's last reply?
            customer_after_last_brand = [
                turn for i, turn in enumerate(thread_turns) 
                if i > last_brand_idx and not turn["is_brand"]
            ]
            
            if len(customer_after_last_brand) > 0:
                further_customer_complaint = True
                went_quiet_after_brand = False
            else:
                further_customer_complaint = False
                went_quiet_after_brand = True  # Noisy resolution proxy!
        
        # Initial customer message (the query that initiated the thread)
        initial_customer_msg = customer_turns[0]["clean_text"] if customer_turns else ""
        initial_customer_raw = customer_turns[0]["raw_text"] if customer_turns else ""
        
        # Brand's resolution reply
        brand_final_reply = brand_turns[-1]["clean_text"] if brand_turns else ""
        brand_final_raw = brand_turns[-1]["raw_text"] if brand_turns else ""

        threads.append({
            "thread_id": root_id,
            "total_turns": len(thread_turns),
            "turns": thread_turns,
            "customer_turns": customer_turns,
            "brand_turns": brand_turns,
            "brand_replied": brand_replied,
            "further_customer_complaint": further_customer_complaint,
            "went_quiet_after_brand": went_quiet_after_brand,
            "resolution_proxy": went_quiet_after_brand,  # Explicitly labeled as noisy resolution proxy
            "initial_customer_text": initial_customer_msg,
            "initial_customer_raw": initial_customer_raw,
            "brand_final_reply": brand_final_reply,
            "brand_final_raw": brand_final_raw,
        })

    return threads


def extract_resolution_pairs(threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter threads where the noisy resolution proxy holds (went_quiet_after_brand)
    and extract (customer message, brand reply) pairs for historical RAG grounding.
    """
    pairs = []
    for thread in threads:
        if (
            thread["resolution_proxy"]
            and len(thread["initial_customer_text"]) > 10
            and len(thread["brand_final_reply"]) > 10
        ):
            pairs.append({
                "thread_id": thread["thread_id"],
                "customer_text": thread["initial_customer_text"],
                "customer_raw": thread["initial_customer_raw"],
                "brand_reply": thread["brand_final_reply"],
                "brand_raw": thread["brand_final_raw"],
            })
    return pairs


def run_ingestion_and_reconstruction(
    brand_handle: str = config.BRAND_HANDLE,
    force_download: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    High-level orchestrator for Step 1:
    1. Ingest raw tweets subsample.
    2. Reconstruct threads with turn tracking and noisy resolution proxy.
    3. Save processed threads and resolution pairs to jsonl.
    """
    csv_path = config.RAW_DATA_PATH
    if force_download or not csv_path.exists():
        download_subsample_stream(brand_handle=brand_handle, output_path=csv_path)

    print(f"Loading raw subsample from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} tweets.")

    threads = reconstruct_threads(df, brand_handle=brand_handle)
    print(f"Reconstructed {len(threads)} threads.")

    # Write processed threads
    with open(config.PROCESSED_THREADS_PATH, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved processed threads to {config.PROCESSED_THREADS_PATH}")

    # Extract resolution pairs
    resolution_pairs = extract_resolution_pairs(threads)
    with open(config.RESOLUTION_PAIRS_PATH, "w", encoding="utf-8") as f:
        for p in resolution_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(resolution_pairs)} historical resolution pairs to {config.RESOLUTION_PAIRS_PATH}")

    return threads, resolution_pairs


if __name__ == "__main__":
    run_ingestion_and_reconstruction()
