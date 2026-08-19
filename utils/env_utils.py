"""Environment and API key resolution helpers for Streamlit Cloud and local runs."""
import os


def get_api_key(key_name: str) -> str:
    """Retrieve API key from OS environment variable or Streamlit secrets dictionary."""
    # 1. Check OS Environment variable
    val = os.getenv(key_name)
    if val:
        return val.strip()

    # 2. Check Streamlit Secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            if key_name in st.secrets:
                return str(st.secrets[key_name]).strip()
            # Case-insensitive lookup
            for k, v in st.secrets.items():
                if k.upper() == key_name.upper():
                    return str(v).strip()
    except Exception:
        pass

    return ""
