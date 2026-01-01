#!/usr/bin/env bash
pip install uv
uv run --with streamlit streamlit run main.py --server.port=8501 --server.address=0.0.0.0 --server.enableWebsocketCompression=false --server.enableCORS=false
