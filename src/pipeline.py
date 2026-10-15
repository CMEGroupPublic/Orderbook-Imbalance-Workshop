# Copyright 2026 CME Group Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import logging
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from .agents import data_retrieval_agent, visualizer_agent
from .utils import validate_inputs, parse_nl_query, config


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("orderbook_pipeline")

# Define the LangGraph State representation matching our pipeline data schema
class PipelineState(TypedDict):
    symbol: str
    start_date: str
    end_date: str
    raw_data: Optional[List[Dict[str, Any]]]
    processed_data: Optional[List[Dict[str, Any]]]
    chart_path: Optional[str]
    messages: List[Dict[str, str]]

def build_pipeline_graph() -> StateGraph:
    """
    Constructs and compiles the 100% LangGraph workflow.
    Executes sequentially: data_retrieval_agent -> visualizer_agent -> END.
    """
    # 1. Initialize StateGraph with the PipelineState schema
    workflow = StateGraph(PipelineState)
    
    # 2. Register Node functions
    workflow.add_node("data_retrieval", data_retrieval_agent)
    workflow.add_node("visualizer", visualizer_agent)
    
    # 3. Establish execution flow edges
    workflow.add_edge(START, "data_retrieval")
    workflow.add_edge("data_retrieval", "visualizer")
    workflow.add_edge("visualizer", END)
    
    # 4. Compile the state graph into a runnable workflow
    return workflow.compile()

def run_pipeline(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Top-level API entrypoint to execute the multi-agent reconstructed L2 order book pipeline.
    Ensures zero-trust input validation before starting graph execution.
    """
    logger.info(f"Triggering L2 Order Book Pipeline for {symbol} ({start_date} to {end_date})")
    
    # Zero-Trust Input Validation before passing parameters to graph execution
    v_symbol, v_start, v_end = validate_inputs(symbol, start_date, end_date)
    
    initial_state: PipelineState = {
        "symbol": v_symbol,
        "start_date": v_start,
        "end_date": v_end,
        "raw_data": None,
        "processed_data": None,
        "chart_path": None,
        "messages": []
    }
    
    # Build, compile, and invoke the LangGraph workflow
    compiled_workflow = build_pipeline_graph()
    final_state = compiled_workflow.invoke(initial_state)
    
    return final_state

if __name__ == "__main__":
    import sys
    
    # Defaults from config.yaml
    sym = config.get("pipeline_defaults", {}).get("symbol", "CL")
    start = config.get("pipeline_defaults", {}).get("start_date", "2026-08-18")
    end = config.get("pipeline_defaults", {}).get("end_date", "2026-08-19")
    
    query_used = None
    
    if len(sys.argv) == 2:
        query_used = sys.argv[1]
        print(f"Parsing Natural Language Query: '{query_used}'...")
        sym, start, end = parse_nl_query(query_used)
        print(f"Extracted parameters: Symbol={sym}, Start Date={start}, End Date={end}\n")
    elif len(sys.argv) >= 4:
        sym = sys.argv[1]
        start = sys.argv[2]
        end = sys.argv[3]
    else:
        try:
            print("="*80)
            print("CME LEVEL 2 ORDER BOOK PIPELINE - INTERACTIVE NATURAL LANGUAGE MODE")
            print("="*80)
            print("Example queries:")
            print("  - 'Show me the crude oil chart for yesterday'")
            print("  - 'Give me gold metrics from August 18, 2026 to August 19, 2026'")
            print("  - 'Analyze Nasdaq on 2026-08-18'")
            print("="*80)
            query_used = input("Enter your query: ").strip()
            if query_used:
                print(f"\nParsing query: '{query_used}'...")
                sym, start, end = parse_nl_query(query_used)
                print(f"Extracted parameters: Symbol={sym}, Start Date={start}, End Date={end}\n")
            else:
                print("No query entered. Using defaults.")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user. Using defaults.")
            
    try:
        result = run_pipeline(sym, start, end)
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUCCESS")
        print(f"Interactive Dashboard Saved To: {result['chart_path']}")
        if query_used:
            print(f"Processed Query: {query_used}")
        print("="*80)
        for msg in result["messages"]:
            print(f"[{msg['role'].upper()}]\n{msg['content']}\n" + "-"*80)
    except FileNotFoundError as e:
        print("\n" + "="*80)
        print("PIPELINE EXECUTION FAILED GRACEFULLY")
        print(f"Status:   NO DATA AVAILABLE")
        print(f"Details:  {e}")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        err_str = str(e)
        if "reauthenticate" in err_str.lower() or "RefreshError" in err_str or "credentials" in err_str.lower():
            print("\n" + "="*80)
            print("PIPELINE EXECUTION FAILED GRACEFULLY")
            print("Status:   AUTHENTICATION EXPIRED")
            print("Details:  Your Google Application Default Credentials (ADC) require reauthentication.")
            print("\nPlease execute the following in your unrestricted terminal:")
            print("  gcloud auth application-default login")
            print("And restart the Gemini CLI session to sync the refreshed credentials.")
            print("="*80)
            sys.exit(1)
        else:
            logger.error(f"Pipeline execution encountered an unexpected fatal error: {err_str}", exc_info=True)
            sys.exit(1)
