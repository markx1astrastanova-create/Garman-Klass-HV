from pathlib import Path
import threading

import streamlit as st
import streamlit.components.v1 as components

from streamlit_backend import dispatch

st.set_page_config(page_title="GKHV", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stMainBlockContainer {padding: 0; max-width: 100%;}
.stMainBlockContainer [data-testid="stVerticalBlock"] {gap: 0;}
iframe[title="streamlit_app.gkhv"] {display: block; height: 100dvh;}
header[data-testid="stHeader"] {display: none;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_backend():
    from BACKEND import main
    return main, threading.Lock()


component = components.declare_component(
    "gkhv", path=str(Path(__file__).parent / "FRONTEND" / "streamlit_dist")
)
request = component(response=st.session_state.get("response"), key="gkhv", default=None)
if isinstance(request, dict) and request.get("id") != st.session_state.get("request_id"):
    backend, lock = load_backend()
    # Serialize upstream data requests across browser sessions.
    with lock:
        st.session_state.response = dispatch(request, backend)
    st.session_state.request_id = request["id"]
    st.rerun()
