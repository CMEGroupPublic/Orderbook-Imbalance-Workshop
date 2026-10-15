# Copyright 2026 CME Group Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import re
import os
import yaml
import logging
import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("orderbook_pipeline.utils")

# ---------------------------------------------------------------------------
# Centralized Configuration Management
# ---------------------------------------------------------------------------

CONFIG_PATH = os.getenv("ORDERBOOK_CONFIG_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))

def load_config() -> Dict[str, Any]:
    """
    Loads pipeline parameters and constants from a centralized config.yaml file.
    Forces reading from config.yaml only; raises FileNotFoundError or ValueError if missing/malformed.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Configuration Error: Centralized configuration file was not found at '{CONFIG_PATH}'. "
            "The pipeline is configured to run ONLY with a valid config.yaml file."
        )
        
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Configuration Error: The file at '{CONFIG_PATH}' is malformed or not a valid dictionary. "
                "The pipeline is configured to run ONLY with a valid config.yaml file."
            )
            
        # Dynamically resolve relative output_dir to absolute path relative to utils.py location (parent directory)
        if "paths" in cfg and "output_dir" in cfg["paths"]:
            out_dir = cfg["paths"]["output_dir"]
            if out_dir is not None:
                if not os.path.isabs(out_dir):
                    cfg["paths"]["output_dir"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", out_dir))
                
        # Dynamically resolve placeholder prefix, bucket, and project fallbacks using environment variables
        if "paths" in cfg and cfg["paths"].get("gcs_bucket") == "your-gcs-bucket-name":
            cfg["paths"]["gcs_bucket"] = os.getenv("GCS_BUCKET", "your-gcs-bucket-name")
        if "paths" in cfg and cfg["paths"].get("gcs_prefix") == "your-gcs-prefix":
            cfg["paths"]["gcs_prefix"] = os.getenv("GCS_PREFIX", "your-gcs-prefix")
        if "model" in cfg and cfg["model"].get("project_fallback") == "your-gcp-project-id":
            cfg["model"]["project_fallback"] = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "your-gcp-project-id"
            
        return cfg
    except Exception as e:
        logger.error(f"Failed to load centralized configuration from '{CONFIG_PATH}': {e}")
        raise

config = load_config()

# ---------------------------------------------------------------------------
# Zero-Trust Input Validation & Sanitization
# ---------------------------------------------------------------------------

def validate_symbol(symbol: Any) -> str:
    """
    Validates the contract symbol using a strict zero-trust approach.
    Only allows uppercase alphanumeric strings of length 1 to 10.
    """
    if not isinstance(symbol, str):
        raise TypeError(f"Invalid symbol type: expected str, got {type(symbol).__name__}")
        
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        raise ValueError("Symbol cannot be empty or whitespace only.")
        
    # Strict regex validation to prevent prompt injection or directory traversal
    if not re.match(r"^[A-Z0-9]{1,10}$", symbol_clean):
        raise ValueError(
            f"Security Violation: Symbol '{symbol_clean}' contains unauthorized characters. "
            "Only uppercase letters and numbers of length 1-10 are allowed."
        )
    return symbol_clean

def validate_date(date_str: Any) -> str:
    """
    Validates that the provided date matches standard formats (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    and represents a chronologically valid date/time.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"Invalid date type: expected str, got {type(date_str).__name__}")
        
    date_clean = date_str.strip()
    if not date_clean:
        raise ValueError("Date string cannot be empty.")
        
    # Allowed date/time formats
    formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats:
        try:
            # Parse to verify chronological correctness
            parsed_dt = datetime.datetime.strptime(date_clean, fmt)
            return parsed_dt.strftime(fmt)
        except ValueError:
            continue
            
    raise ValueError(
        f"Security Violation: Date '{date_clean}' is not in a valid format. "
        "Allowed formats are YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."
    )

def validate_inputs(symbol: Any, start_date: Any, end_date: Any) -> tuple:
    """
    Runs security-first, zero-trust validations on all dynamic graph inputs.
    """
    valid_symbol = validate_symbol(symbol)
    valid_start = validate_date(start_date)
    valid_end = validate_date(end_date)
    
    # Chronological validation
    formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]
    def to_datetime(s):
        for fmt in formats:
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse '{s}' as datetime.")
        
    start_dt = to_datetime(valid_start)
    end_dt = to_datetime(valid_end)
    
    if start_dt > end_dt:
        raise ValueError(f"Chronological Error: Start date '{valid_start}' cannot be after end date '{valid_end}'.")
        
    return valid_symbol, valid_start, valid_end

# ---------------------------------------------------------------------------
# LLM Instantiation (Gemini 2.5 Pro via ChatGoogleGenerativeAI with Vertex AI)
# ---------------------------------------------------------------------------

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the real model using ChatGoogleGenerativeAI with Vertex AI,
    configured dynamically via config.yaml. No mocks are used.
    """
    import google.auth
    creds, proj = google.auth.default()
    if not proj:
        # Default project fallback if not resolved automatically from credentials
        proj = config.get("model", {}).get("project_fallback", "your-gcp-project-id")
        
    return ChatGoogleGenerativeAI(
        model=config.get("model", {}).get("name", "gemini-2.5-pro"),
        vertexai=config.get("model", {}).get("vertexai", True),
        project=proj,
        credentials=creds,
        temperature=config.get("model", {}).get("temperature", 0.0)
    )

# ---------------------------------------------------------------------------
# Data Loading from GCS (Essential Production Only)
# ---------------------------------------------------------------------------

def _load_real_data_from_gcs(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scans the configured GCS bucket,
    identifies partitions containing relevant asset and date range,
    loads, filters, renames, scales prices, and downsamples the data.
    """
    import io
    import pandas as pd
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError, RefreshError
    
    target = symbol.strip().upper()
    gcs_bucket = config.get("paths", {}).get("gcs_bucket", "your-gcs-bucket-name")
    gcs_prefix = config.get("paths", {}).get("gcs_prefix", "your-gcs-prefix").strip("/")
    
    logger.info(f"Connecting to GCS bucket '{gcs_bucket}' with prefix '{gcs_prefix}' for asset '{target}'...")
    
    try:
        client = storage.Client()
    except (DefaultCredentialsError, RefreshError) as auth_err:
        err_msg = (
            f"Authentication failed while trying to connect to GCS: {auth_err}. "
            "Please run 'gcloud auth login' in your standard terminal and restart the Gemini CLI to sync credentials."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg) from auth_err
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        raise
        
    # Generate dates in range [start_date, end_date]
    try:
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
    except Exception as e:
        logger.error(f"Error parsing start/end dates: {e}")
        return pd.DataFrame()
        
    p_start = pd.to_datetime(start_date)
    p_end = pd.to_datetime(end_date)
    # If end_date is a bare date without time, make the upper bound inclusive of the whole day
    if " " not in str(end_date) and "T" not in str(end_date) and ":" not in str(end_date):
        p_end = p_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        
    dates = []
    curr = start_dt
    while curr <= end_dt:
        dates.append(curr)
        curr += pd.Timedelta(days=1)
        
    # Construct targeted prefixes to avoid scanning the entire bucket.
    candidate_prefixes = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        date_str_flat = d.strftime("%Y%m%d")
        
        for d_str in [date_str, date_str_flat]:
            candidate_prefixes.extend([
                f"{gcs_prefix}/date={d_str}/asset={target}",
                f"{gcs_prefix}/{d_str}/{target}",
                f"{gcs_prefix}/asset={target}/date={d_str}",
                f"{gcs_prefix}/{target}/{d_str}",
            ])
            
    # De-duplicate candidate prefixes preserving order
    candidate_prefixes = list(dict.fromkeys(candidate_prefixes))
    
    # List and download blobs matching the dynamic prefixes
    logger.info(f"Scanning GCS using {len(candidate_prefixes)} candidate partition prefixes...")
    blobs_to_read = []
    
    for prefix in candidate_prefixes:
        try:
            blobs = list(client.list_blobs(gcs_bucket, prefix=prefix))
            for b in blobs:
                if b.name.endswith(".parquet"):
                    blobs_to_read.append(b)
        except Exception as e:
            logger.warning(f"Error listing blobs for prefix '{prefix}': {e}")
            continue
            
    if not blobs_to_read:
        logger.warning(
            f"No partitioned Parquet files found in GCS bucket '{gcs_bucket}' "
            f"matching symbol '{target}' and date range [{start_date}, {end_date}]."
        )
        return pd.DataFrame()
        
    # Sort blobs to read them chronologically by name
    blobs_to_read.sort(key=lambda x: x.name)
    
    dfs = []
    cols_to_use = [
        'symbol', 'asset', 'transactTime',
        'level_1_bid_price', 'level_1_bid_quantity', 'level_1_ask_price', 'level_1_ask_quantity',
        'level_2_bid_price', 'level_2_bid_quantity', 'level_2_ask_price', 'level_2_ask_quantity',
        'level_3_bid_price', 'level_3_bid_quantity', 'level_3_ask_price', 'level_3_ask_quantity',
        'level_4_bid_price', 'level_4_bid_quantity', 'level_4_ask_price', 'level_4_ask_quantity',
        'level_5_bid_price', 'level_5_bid_quantity', 'level_5_ask_price', 'level_5_ask_quantity'
    ]
    
    for b in blobs_to_read:
        try:
            logger.info(f"Reading Parquet data from GCS blob: gs://{gcs_bucket}/{b.name}")
            content = b.download_as_bytes()
            import pyarrow as pa
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(io.BytesIO(content))
            table = pf.read(columns=cols_to_use)
            clean_fields = [pa.field(name, table.schema.field(name).type) for name in cols_to_use]
            clean_schema = pa.schema(clean_fields)
            clean_table = pa.Table.from_arrays(table.columns, schema=clean_schema)
            df_chunk = clean_table.to_pandas()
            
            # Filter rows by symbol/asset match
            df_chunk = df_chunk[(df_chunk['asset'].str.upper() == target) | (df_chunk['symbol'].str.upper() == target)]
            
            # Filter rows by start/end date range
            df_chunk['datetime_parsed'] = pd.to_datetime(df_chunk['transactTime'].str[:14], format='%Y%m%d%H%M%S')
            df_chunk = df_chunk[(df_chunk['datetime_parsed'] >= p_start) & 
                                (df_chunk['datetime_parsed'] <= p_end)]
            
            if not df_chunk.empty:
                dfs.append(df_chunk)
        except Exception as e:
            logger.warning(f"Error processing GCS blob gs://{gcs_bucket}/{b.name}: {e}")
            continue
            
    if not dfs:
        return pd.DataFrame()
        
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(by='datetime_parsed').reset_index(drop=True)
    
    # De-duplicate ticks within the same second, then resample and forward-fill on continuous 1-second grid
    df_all = df_all.drop_duplicates(subset=['datetime_parsed'], keep='last')
    df_all = df_all.set_index('datetime_parsed')
    df_all = df_all.resample('1s').ffill()
    df_all = df_all.reset_index()
    
    # Standardize timestamp to human-readable format YYYY-MM-DD HH:MM:SS
    df_all['timestamp'] = df_all['datetime_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Map columns to standard schema
    rename_dict = {}
    for i in range(1, 6):
        rename_dict[f'level_{i}_bid_price'] = f'bid_price_{i}'
        rename_dict[f'level_{i}_bid_quantity'] = f'bid_size_{i}'
        rename_dict[f'level_{i}_ask_price'] = f'ask_price_{i}'
        rename_dict[f'level_{i}_ask_quantity'] = f'ask_size_{i}'
        
    df_all = df_all.rename(columns=rename_dict)
    
    # Scale prices (divide by 100.0 to convert tick-integers to real dollar values)
    for i in range(1, 6):
        df_all[f'bid_price_{i}'] = df_all[f'bid_price_{i}'] / 100.0
        df_all[f'ask_price_{i}'] = df_all[f'ask_price_{i}'] / 100.0
        
    df_all = df_all.drop(columns=['datetime_parsed', 'transactTime'], errors='ignore')
    
    # Keep only the columns expected in the standard schema
    final_cols = ['timestamp', 'symbol']
    for i in range(1, 6):
        final_cols.extend([f'bid_price_{i}', f'bid_size_{i}', f'ask_price_{i}', f'ask_size_{i}'])
        
    df_all = df_all[[c for c in final_cols if c in df_all.columns]]
    
    # Systematic downsampling to configured target ticks
    target_ticks = config.get("parameters", {}).get("downsample_target_ticks", 120)
    if len(df_all) > target_ticks:
        step = len(df_all) // target_ticks
        df_all = df_all.iloc[::step].head(target_ticks).reset_index(drop=True)
        
    logger.info(f"Successfully processed GCS order book data. Retained {len(df_all)} rows.")
    return df_all


def generate_l2_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves Level 2 (Market-by-Price) order book data with 5 depth levels.
    Loads and processes real CME L2 Parquet logs directly from the GCS bucket.
    """
    logger.info(f"Retrieving L2 order book data for symbol '{symbol}' from {start_date} to {end_date}...")
    
    # Ingestion directly from GCS with zero dummy fallbacks
    df_real = _load_real_data_from_gcs(symbol, start_date, end_date)
    if df_real.empty:
        err_msg = (
            f"No real Level 2 order book data found. "
            f"GCS Data Ingestion Error: No Level 2 order book records found for contract symbol '{symbol}' "
            f"within the requested date range [{start_date}, {end_date}] in the GCS bucket."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
        
    return df_real


def compute_l2_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs core mathematical calculations:
    1. Mid-Price (Top-of-Book) = (Best Bid Price1 + Best Ask Price1) / 2
    2. NOBI (L1 Imbalance) = (Bid Size1 - Ask Size1) / (Bid Size1 + Ask Size1)
    3. WOBI (L5 Weighted Imbalance) = [Sum(Bid Size_i / i) - Sum(Ask Size_i / i)] / [Sum(Bid Size_i / i) + Sum(Ask Size_i / i)]
    4. 20-tick EMA Smoothing of NOBI and WOBI
    """
    df_calc = df.copy()
    
    # 1. Mid-Price
    df_calc['mid_price'] = (df_calc['bid_price_1'] + df_calc['ask_price_1']) / 2.0
    
    # 2. Top-of-Book Imbalance (NOBI L1)
    nobi_denom = df_calc['bid_size_1'] + df_calc['ask_size_1']
    df_calc['NOBI'] = np.where(nobi_denom == 0, 0.0, (df_calc['bid_size_1'] - df_calc['ask_size_1']) / nobi_denom)
    
    # 3. L2 Weighted Deep-Book Imbalance (WOBI k=5 levels)
    weighted_bid = 0.0
    weighted_ask = 0.0
    for i in range(1, 6):
        weighted_bid += df_calc[f'bid_size_{i}'] / i
        weighted_ask += df_calc[f'ask_size_{i}'] / i
        
    wobi_denom = weighted_bid + weighted_ask
    df_calc['WOBI'] = np.where(wobi_denom == 0, 0.0, (weighted_bid - weighted_ask) / wobi_denom)
    
    # 4. 20-tick Exponential Moving Average (EMA) smoothing to filter quote flickering
    smoothing_span = config.get("parameters", {}).get("smoothing_span", 20)
    df_calc['NOBI_smooth'] = df_calc['NOBI'].ewm(span=smoothing_span, adjust=False).mean()
    df_calc['WOBI_smooth'] = df_calc['WOBI'].ewm(span=smoothing_span, adjust=False).mean()
    
    return df_calc


def parse_nl_query(query: str) -> tuple:
    """
    Parses a natural language query using Gemini to extract symbol, start_date, and end_date.
    Returns:
        tuple: (symbol, start_date, end_date)
    """
    import json
    # Use reference date from centralized configuration
    today_str = config.get("pipeline_defaults", {}).get("reference_today_date", "2026-08-20")
    default_start = config.get("pipeline_defaults", {}).get("start_date", "2026-08-18")
    default_end = config.get("pipeline_defaults", {}).get("end_date", "2026-08-19")
    default_symbol = config.get("pipeline_defaults", {}).get("symbol", "CL")
    
    prompt = f"""You are an advanced natural language query processor for a CME order book pipeline.
Your job is to parse the user's conversational request and extract the following parameters as a valid JSON object:
- "symbol": The standardized CME futures symbol. Map common names:
  - "crude", "oil", "crude oil", "cl" -> "CL"
  - "gold", "gc" -> "GC"
  - "s&p", "s&p 500", "sp500", "e-mini", "es" -> "ES"
  - "nasdaq", "nq", "nq100" -> "NQ"
- "start_date": Format as "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS".
- "end_date": Format as "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS".

System Reference Date Context:
Today's date is {today_str} (Thursday).
Use this reference date to resolve relative phrases like "yesterday" (which would be "2026-08-19"), "today" (which would be "2026-08-20"), "last week", etc.
If only a single day is specified (e.g. "for August 18, 2026"), "start_date" should be "2026-08-18" and "end_date" should be "2026-08-19" (representing the interval of that day).
If no start/end dates are found, default to "start_date": "{default_start}" and "end_date": "{default_end}".

Respond ONLY with a valid JSON object of the following format, with no markdown code blocks, no backticks, and no extra conversational text:
{{
  "symbol": "CL",
  "start_date": "2026-08-18",
  "end_date": "2026-08-19"
}}

User Conversational Query:
"{query}"
"""
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        text = response.content.strip()
        
        # Clean any accidental markdown backticks formatting from LLM output
        text_cleaned = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text_cleaned = re.sub(r"\s*```$", "", text_cleaned)
        text_cleaned = text_cleaned.strip()
        
        parsed = json.loads(text_cleaned)
        
        symbol_extracted = parsed.get("symbol", default_symbol)
        start_extracted = parsed.get("start_date", default_start)
        end_extracted = parsed.get("end_date", default_end)
        
        return symbol_extracted, start_extracted, end_extracted
    except Exception as e:
        logger.error(f"Failed to parse LLM JSON response: '{text if 'text' in locals() else ''}'. Error: {e}")
        return default_symbol, default_start, default_end
