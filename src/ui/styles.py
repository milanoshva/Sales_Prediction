import streamlit as st
import plotly.graph_objects as go # Added: Import as go

def set_custom_ui():
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --bg-color: #0e1117;
            --text-color: #E6E6E6;
            --accent-color: #4169E1;
            --accent-hover: #3457D1;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --card-bg: #1E222A;
            --border-color: #30363d;
        }
        
        /* Main app background and text */
        .stApp { 
            background-color: var(--bg-color); 
            color: var(--text-color); 
        }
        
        /* Text styling */
        .stMarkdown, .stTitle, .stHeader, .stSubheader { 
            color: var(--text-color); 
        }
        
        h1, h2, h3, h4 {
            font-weight: 600 !important;
            color: white !important;
            margin: 0.5rem 0 !important;
        }
        
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        
        /* Buttons styling */
        .stButton>button { 
            background-color: var(--accent-color);
            color: white; 
            border-radius: 6px; 
            font-weight: 600; 
            padding: 0.3rem 0.8rem;
            border: none;
            transition: all 0.3s ease;
            font-size: 0.85rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        
        .stButton>button:hover {
            background-color: var(--accent-hover);
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }
        
        /* Download button styling */
        .stDownloadButton>button { 
            background-color: var(--success-color);
            color: white; 
            border-radius: 6px; 
            font-weight: 600; 
            padding: 0.3rem 0.8rem;
            border: none;
            transition: all 0.3s ease;
            font-size: 0.85rem;
        }
        
        .stDownloadButton>button:hover {
            filter: brightness(110%);
            transform: translateY(-1px);
        }
        
        /* File uploader styling (specific to Home.py) */
        .stFileUploader {
            background-color: var(--card-bg);
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            color: var(--text-color);
        }
        
        .stFileUploader label {
            color: var(--text-color) !important;
            font-size: 0.9rem;
        }
        
        .stFileUploader div[role="button"] {
            background-color: var(--accent-color);
            color: white;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.3rem 0.8rem;
        }
        
        .stFileUploader div[role="button"]:hover {
            background-color: var(--accent-hover);
        }
        
        /* Form elements styling (general) */
        section[data-testid="stSelectbox"], 
        section[data-testid="stNumberInput"],
        section[data-testid="stMultiselect"],
        section[data-testid="stSlider"],
        section[data-testid="stDateInput"],
        section[data-testid="stTextInput"],
        .stRadio { /* Apply base styling to stRadio */
            background-color: var(--card-bg);
            border-radius: 6px;
            padding: 0.3rem;
            border: 1px solid var(--border-color);
            margin-bottom: 0.3rem;
        }
        
        div[data-baseweb="select"] {
            background-color: var(--card-bg) !important;
            border-radius: 4px;
            color: var(--text-color);
            border: 1px solid var(--border-color) !important;
            box-shadow: none !important;
            outline: none !important;
            max-width: 100%;
            appearance: none !important;
            padding-right: 0 !important;
            background: transparent !important;
            font-size: 0.85rem;
        }
        
        div[data-baseweb="select"]:hover {
            border-color: var(--accent-color) !important;
        }
        
        div[data-baseweb="select"] > div {
            border: none !important;
            background: transparent !important;
        }
        
        div[data-baseweb="select"]::after {
            display: none !important;
        }
        
        input {
            background-color: var(--card-bg) !important;
            color: var(--text-color) !important;
            border-radius: 4px !important;
            border: 1px solid var(--border-color) !important;
            padding: 0.3rem !important;
            font-size: 0.85rem !important;
        }
        
        input:focus {
            border-color: var(--accent-color) !important;
        }
        
        /* DataFrames styling */
        .stDataFrame { 
            background-color: var(--card-bg); 
            border-radius: 6px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            overflow: hidden;
            font-size: 0.85rem;
        }
        
        /* Cards styling */
        .card {
            background: linear-gradient(145deg, #1E222A, #252A34);
            padding: 0.8rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            margin-bottom: 0.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            overflow: hidden;
        }
        
        .card:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        }
        
        /* Info box styling */
        .info-box { 
            background-color: var(--card-bg); 
            padding: 0.8rem; 
            border-radius: 8px; 
            border-left: 4px solid var(--accent-color); 
            margin-bottom: 0.5rem; 
            font-size: 0.85rem; 
            color: var(--text-color);
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            line-height: 1.4;
        }
        
        .info-box h3 {
            margin-top: 0;
            color: white;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(145deg, #1E222A, #252A34);
            padding: 0.8rem;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            text-align: center;
            margin: 0.3rem;
            font-size: 0.85rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            height: 100%;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        }
        
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: white;
            margin: 0.3rem 0;
        }
        
        .metric-label {
            color: #afb5c0;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        /* Radio button styling */
        .stRadio {
            display: flex;
            flex-wrap: nowrap;
            gap: 0.5rem;
            align-items: center;
            white-space: nowrap;
            position: relative;
        }
        
        .stRadio label {
            margin: 0 0.3rem;
            font-size: 0.85rem;
            color: var(--text-color);
            cursor: pointer;
            position: relative;
        }
        
        .stRadio > div > label > span {
            margin-right: 0.3rem;
        }
        
        .stRadio input[type="radio"] {
            cursor: pointer;
        }
        
        /* Disable interaction for category and payment radios when toggled off */
        .stRadio.disabled label {
            cursor: not-allowed;
            opacity: 0.5;
        }
        .stRadio.disabled input[type="radio"] {
            cursor: not-allowed;
        }
        
        /* Advanced badge styling */
        .advanced-badge { 
            background-color: var(--accent-color); 
            color: white; 
            padding: 0.3rem 0.6rem; 
            border-radius: 5px; 
            font-size: 0.85rem; 
            font-weight: 600;
            display: inline-block;
            margin-left: 0.5rem;
        }
        
        /* Alert styling */
        .st-alert {
            padding: 0.5rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            font-size: 0.85rem;
        }
        
        .st-alert-info {
            background-color: rgba(65, 105, 225, 0.1);
            border-left: 4px solid var(--accent-color);
        }
        
        .st-alert-success {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid var(--success-color);
        }
        
        .st-alert-warning {
            background-color: rgba(255, 193, 7, 0.1);
            border-left: 4px solid var(--warning-color);
        }
        
        .st-alert-error {
            background-color: rgba(220, 53, 69, 0.1);
            border-left: 4px solid var(--danger-color);
        }
        
        /* Result card styling */
        .result-card {
            background-color: var(--card-bg);
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            margin-top: 0.5rem;
        }
        
        .result-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.3rem;
        }
        
        .result-metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.3rem;
            font-size: 0.85rem;
        }
        
        .result-metric-label {
            color: #afb5c0;
        }
        
        .result-metric-value {
            font-weight: 600;
            color: white;
        }
        
        /* Expander styling */
        .stExpander {
            background-color: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            margin-bottom: 0.5rem;
            border: 1px solid var(--border-color);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .card {
                margin-bottom: 0.5rem;
            }
            div[data-baseweb="select"] {
                max-width: 100% !important;
            }
            .stRadio {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.2rem;
            }
        }
        
        /* Reduce spacing */
        .st-emotion-cache-1wmy9hl {
            margin: 0 !important;
            padding: 0.5rem !important;
        }
        
        .st-emotion-cache-1dp5vir {
            padding: 0.5rem !important;
        }
        
        /* Smaller charts */
        .stPlotlyChart {
            margin: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def get_plotly_template():
    # Plotly theme definition, moved here for centralization
    return go.layout.Template(
        layout=dict(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#1E222A',
            font=dict(color='#E6E6E6', size=10),
            xaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
            margin=dict(l=30, r=30, t=30, b=30),
            height=300
        )
    )