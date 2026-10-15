# Copyright 2026 CME Group Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List
from markdown_it import MarkdownIt

from .utils import (
    validate_inputs,
    get_llm,
    generate_l2_data,
    compute_l2_metrics,
    config,
)

logger = logging.getLogger("orderbook_pipeline.agents")

# ---------------------------------------------------------------------------
# Node 1: data_retrieval_agent
# ---------------------------------------------------------------------------

def data_retrieval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent Node: data_retrieval_agent
    Validates dynamic graph inputs, invokes gemini-2.5-pro for retrieval planning,
    generates L2 MBP reconstructed order book logs, and computes L2 metrics.
    """
    symbol = state.get("symbol")
    start_date = state.get("start_date")
    end_date = state.get("end_date")
    
    logger.info(f"[{symbol}] Entering data_retrieval_agent node...")
    
    # Security-First Zero-Trust Validation of dynamic inputs
    v_symbol, v_start, v_end = validate_inputs(symbol, start_date, end_date)
    logger.info(f"Inputs validated. symbol={v_symbol}, start={v_start}, end={v_end}")
    
    # Initialize the real Gemini 2.5 Pro model for planning
    llm = get_llm()
    plan_prompt = (
        f"You are the data_retrieval_agent in a security-hardened CME L2 order book analysis pipeline.\n"
        f"Formulate a brief 2-sentence planning strategy for retrieving and metrics-processing Level 2 "
        f"MBP data for futures contract symbol '{v_symbol}' from {v_start} to {v_end}.\n"
        f"Do not include any greeting or polite filler."
    )
    response = llm.invoke(plan_prompt)
    plan_content = response.content.strip()
    logger.info(f"data_retrieval_agent LLM response: {plan_content}")
    
    # Retrieve high-fidelity depth logs
    df_raw = generate_l2_data(v_symbol, v_start, v_end)
    raw_list = df_raw.to_dict(orient="records")
    
    # Compute L2 metrics (Mid-Price, NOBI L1, WOBI L5, 20-tick EMA smoothing)
    df_processed = compute_l2_metrics(df_raw)
    processed_list = df_processed.to_dict(orient="records")
    
    # State update
    messages = list(state.get("messages", []))
    messages.append({
        "role": "data_retrieval_agent",
        "content": plan_content
    })
    
    return {
        **state,
        "symbol": v_symbol,
        "start_date": v_start,
        "end_date": v_end,
        "raw_data": raw_list,
        "processed_data": processed_list,
        "messages": messages
    }

# ---------------------------------------------------------------------------
# Node 2: visualizer_agent
# ---------------------------------------------------------------------------

def visualizer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent Node: visualizer_agent
    Consumes calculated metrics, invokes gemini-2.5-pro for advanced depth analysis,
    and exports a professional dual-axis dark-themed Plotly dashboard.
    """
    symbol = state.get("symbol")
    start_date = state.get("start_date")
    end_date = state.get("end_date")
    processed_data = state.get("processed_data")
    
    logger.info(f"[{symbol}] Entering visualizer_agent node...")
    
    if not processed_data:
        raise ValueError("Security/Operational Error: No processed data found in graph state.")
        
    df = pd.DataFrame(processed_data)
    
    # Initialize real Gemini 2.5 Pro model for deep order book pattern analysis
    llm = get_llm()
    analysis_prompt = (
        f"You are the visualizer_agent in an elite CME Level 2 order book pipeline.\n"
        f"Perform an advanced microstructural analysis of contract '{symbol}' from {start_date} to {end_date}.\n"
        f"Key metrics calculated over this interval:\n"
        f"- Mean Mid-Price: {df['mid_price'].mean():.4f}\n"
        f"- Max L2 Weighted Deep-Book Imbalance (WOBI): {df['WOBI'].max():.4f}\n"
        f"- Min L2 Weighted Deep-Book Imbalance (WOBI): {df['WOBI'].min():.4f}\n\n"
        f"Provide a highly professional 3-point executive bulleted list explaining "
        f"the market microstructural patterns (e.g., correlation between deep book WOBI and price movement, "
        f"smoothing out transient spoofing attempts via decaying weights, or buyer/seller pressure)."
    )
    response = llm.invoke(analysis_prompt)
    analysis_content = response.content.strip()
    logger.info(f"visualizer_agent LLM response:\n{analysis_content}")
    
    # Construct the Plotly subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Left Y-Axis: Mid-Price (Gold line)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name=f"{symbol} Mid-Price",
            line=dict(color="#F2A900", width=1.5),
            mode='lines'
        ),
        secondary_y=False
    )
    
    # Right Y-Axis: L2 Weighted Imbalance (Cyan line)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['WOBI_smooth'],
            name="L2 Imbalance Ratio (EMA)",
            line=dict(color="#00E5FF", width=1.5),
            mode='lines'
        ),
        secondary_y=True
    )
    
    # Update layout using exact visual specifications
    fig.update_layout(
        title=dict(
            text=f"{symbol} Order Book Size Imbalance & Mid-Price ({start_date} to {end_date})",
            font=dict(color="#FFFFFF", size=15, family="monospace")
        ),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#161B22",
        height=750,
        xaxis=dict(
            showgrid=True,
            gridcolor="#21262D",
            tickfont=dict(color="#C2CACE", size=11),
            title=dict(text="Timestamp", font=dict(color="#FFFFFF", size=12)),
            rangeslider=dict(visible=True)  # Range slider enabled on X-axis
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#21262D",
            tickfont=dict(color="#F2A900", size=11),
            title=dict(text=f"{symbol} Mid-Price ($)", font=dict(color="#F2A900", size=12))
        ),
        yaxis2=dict(
            showgrid=False,
            tickfont=dict(color="#00E5FF", size=11),
            title=dict(text="L2 Imbalance Ratio", font=dict(color="#00E5FF", size=12)),
            range=[-1.1, 1.1]  # Bounded strictly to [-1.1, +1.1]
        ),
        legend=dict(
            font=dict(color="#FFFFFF", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=60, r=60, t=100, b=100)
    )
    
    # Incorporate LLM analysis directly below the Plotly chart
    model_name = config.get("model", {}).get("name", "gemini-2.5-pro")
    md = MarkdownIt()
    analysis_html = md.render(analysis_content)
    
    # Calculate key statistics dynamically for KPI callout cards
    mean_price = df['mid_price'].mean()
    max_wobi = df['WOBI'].max()
    min_wobi = df['WOBI'].min()
    
    commentary_section = f"""
    <div style="background-color: #161B22; color: #FFFFFF; font-family: monospace; padding: 30px; border-top: 1px solid #21262D; font-size: 13px;">
        <h3 style="color: #00E5FF; margin-top: 0; margin-bottom: 20px; font-size: 16px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 8px; height: 8px; background-color: #00E5FF; border-radius: 50%;"></span>
            Executive Microstructural Analysis ({model_name})
        </h3>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-label">CONTRACT SYMBOL</span>
                <span class="metric-value text-gold">{symbol}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">MEAN MID-PRICE</span>
                <span class="metric-value">${mean_price:,.4f}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">MAX WOBI (L5)</span>
                <span class="metric-value text-cyan">+{max_wobi:.4f}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">MIN WOBI (L5)</span>
                <span class="metric-value text-red">-{abs(min_wobi):.4f}</span>
            </div>
        </div>

        <div class="analysis-content">{analysis_html}</div>
    </div>
    """
    
    # Define output workspace path
    output_dir = config.get("paths", {}).get("output_dir", os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)
    
    dashboard_template = config.get("paths", {}).get("pipeline_dashboard_template", "{symbol}_orderbook_imbalance.html")
    chart_filename = dashboard_template.format(symbol=symbol.lower())
    chart_path = os.path.join(output_dir, chart_filename)
    
    # Save complete dashboard HTML
    plotly_chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{symbol} Reconstructed L2 Order Book Dashboard</title>
        <style>
            body {{
                background-color: #0E1117;
                margin: 0;
                padding: 30px 25px 50px 25px;
                font-family: monospace;
                color: #C2CACE;
            }}
            .report-header {{
                max-width: 1200px;
                margin: 0 auto 25px auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #21262D;
                padding-bottom: 15px;
            }}
            .header-title {{
                color: #FFFFFF;
                margin: 0;
                font-size: 20px;
                font-weight: bold;
                letter-spacing: -0.5px;
            }}
            .header-subtitle {{
                color: #8B949E;
                margin: 5px 0 0 0;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .badge {{
                background-color: #21262D;
                color: #00E5FF;
                padding: 5px 12px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #30363D;
                letter-spacing: 0.5px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: #161B22;
                border: 1px solid #21262D;
                border-radius: 8px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                overflow: hidden;
            }}
            .chart-header {{
                background-color: #161B22;
                padding: 15px 25px;
                border-bottom: 1px solid #21262D;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .chart-title {{
                color: #8B949E;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background-color: #0E1117;
                border: 1px solid #21262D;
                border-radius: 6px;
                padding: 16px 20px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                box-shadow: inset 0 1px 4px rgba(0,0,0,0.2);
            }}
            .metric-label {{
                font-size: 10px;
                color: #8B949E;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 600;
            }}
            .metric-value {{
                font-size: 18px;
                font-weight: bold;
                color: #FFFFFF;
                font-family: monospace;
            }}
            .text-gold {{
                color: #F2A900 !important;
            }}
            .text-cyan {{
                color: #00E5FF !important;
            }}
            .text-red {{
                color: #FF5555 !important;
            }}
            .analysis-content {{
                line-height: 1.6;
                color: #C2CACE;
                font-family: monospace;
            }}
            .analysis-content p {{
                margin-top: 0;
                margin-bottom: 18px;
                line-height: 1.6;
            }}
            /* Metadata Paragraph Box Styling */
            .analysis-content p:first-of-type {{
                background-color: #0E1117;
                border: 1px dashed #21262D;
                border-radius: 6px;
                padding: 18px 22px;
                margin-bottom: 25px;
                font-size: 12px;
                color: #8B949E;
                line-height: 1.7;
            }}
            .analysis-content p:first-of-type strong {{
                color: #00E5FF;
                font-weight: bold;
            }}
            .analysis-content ul, .analysis-content ol {{
                margin-top: 0;
                margin-bottom: 20px;
                padding-left: 0;
                list-style: none;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }}
            .analysis-content li {{
                background-color: #0E1117;
                border: 1px solid #21262D;
                border-left: 4px solid #00E5FF;
                border-radius: 6px;
                padding: 18px 22px;
                margin: 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                line-height: 1.6;
            }}
            .analysis-content li strong {{
                color: #FFFFFF;
                display: block;
                font-size: 14px;
                margin-bottom: 8px;
                letter-spacing: -0.2px;
            }}
            .analysis-content strong {{
                color: #FFFFFF;
            }}
            .analysis-content h1, .analysis-content h2, .analysis-content h3, .analysis-content h4 {{
                color: #00E5FF;
                margin-top: 20px;
                margin-bottom: 10px;
                letter-spacing: -0.3px;
            }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <div>
                <h1 class="header-title">CME L2 MICROSTRUCTURAL PIPELINE</h1>
                <p class="header-subtitle">Institutional Order Book Analytics</p>
            </div>
            <div>
                <span class="badge">OPERATIONAL INTEL</span>
            </div>
        </div>
        
        <div class="container">
            <div class="chart-header">
                <span class="chart-title">Interactive Time-Series Visualization</span>
            </div>
            {plotly_chart_html}
            {commentary_section}
        </div>
    </body>
    </html>
    """
    
    with open(chart_path, "w") as f:
        f.write(full_html)
        
    logger.info(f"Interactive order book dashboard exported to: {chart_path}")
    
    messages = list(state.get("messages", []))
    messages.append({
        "role": "visualizer_agent",
        "content": analysis_content
    })
    
    return {
        **state,
        "chart_path": chart_path,
        "messages": messages
    }
