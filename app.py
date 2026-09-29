"""The Royal Athenæum: Harvard Library & HOLLIS Digital Archives.

A museum-grade digital archives and retrieval-augmented research repository
styled with the authoritative, distinguished aesthetic of Harvard University Libraries
and the HOLLIS Digital Archives collection.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_TREATISE_PATH = Path(__file__).parent / "document_loaders" / "philosophical_treatise.txt"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Royal Athenæum | Harvard Library & HOLLIS Archives",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Harvard Library & HOLLIS Digital Archives Design System
# ---------------------------------------------------------------------------
HOLLIS_ARCHIVAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ==========================================================================
   CSS CUSTOM PROPERTIES: Harvard Crimson, Archival Grounds & 8px Grid
   ========================================================================== */
:root {
  /* Primary Harvard Crimson & Variants */
  --crimson-primary: #A51C30;
  --crimson-hover: #861626;
  --crimson-active: #6B111E;
  --crimson-light: #F9EBEF;
  --crimson-border: #D17684;
  --crimson-glow: rgba(165, 28, 48, 0.25);

  /* Academic Brass & Archival Metal Accents */
  --brass-accent: #C5A059;
  --brass-bright: #D8B573;
  --brass-muted: #9E7D3C;
  --brass-subtle: #F3EBDD;
  --brass-border: #D6BD8A;
  --brass-glow: rgba(197, 160, 89, 0.28);

  /* Archival Grounds & Reading Surfaces */
  --bg-archival: #F9F8F6;
  --bg-sheet: #FFFFFF;
  --bg-surface-subtle: #F3F1EC;
  --bg-surface-inset: #ECE8E0;
  --bg-paper-card: #FFFFFF;

  /* High-Contrast Dark Slate & Charcoal Typography */
  --text-primary: #1C1D1F;
  --text-body: #2B2D31;
  --text-muted: #565961;
  --text-faint: #767A83;
  --text-white: #FFFFFF;

  /* Structural Rules & Borders */
  --border-rule-subtle: #E3DFD7;
  --border-rule-strong: #CDC7BC;
  --border-double-brass: 3px double #C5A059;

  /* Deep Dark Oak Sidebar Environment */
  --oak-sidebar-bg: #1E1E1E;
  --oak-surface-card: #272727;
  --oak-surface-hover: #323232;
  --oak-border: #3A3A3A;
  --oak-border-strong: #505050;
  --oak-input-bg: #141414;
  --oak-text-primary: #F7F7F5;
  --oak-text-muted: #B5B5B5;

  /* Shadows */
  --shadow-card: 0 2px 10px rgba(28, 29, 31, 0.06);
  --shadow-frame: 0 4px 24px rgba(28, 29, 31, 0.08);
  --shadow-button: 0 2px 6px rgba(165, 28, 48, 0.25);
  --shadow-sidebar-button: 0 2px 8px rgba(0, 0, 0, 0.35);

  /* Typography */
  --font-display: 'Cinzel', Georgia, 'Times New Roman', serif;
  --font-serif: 'EB Garamond', Garamond, Georgia, serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;

  /* 8px Spatial Grid System */
  --space-8: 8px;
  --space-16: 16px;
  --space-24: 24px;
  --space-32: 32px;
  --space-40: 40px;
  --space-48: 48px;
  --space-64: 64px;
}

/* ==========================================================================
   Base Viewport & Backgrounds
   ========================================================================== */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background-color: var(--bg-archival) !important;
}

.main .block-container {
  max-width: 1080px;
  padding-top: var(--space-24);
  padding-bottom: var(--space-64);
}

/* Base Typography */
p, label, span {
  font-family: var(--font-serif);
  color: var(--text-body);
}

/* Protect Material Symbols & Icons from typography overrides */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons,
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stTextInput"] button *,
[data-testid="stSidebar"] [data-baseweb="input"] button * {
  font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
  font-style: normal !important;
  font-weight: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  word-wrap: normal !important;
  white-space: nowrap !important;
  direction: ltr !important;
}

/* ==========================================================================
   Master Frontispiece: Harvard Library / HOLLIS Digital Archives
   ========================================================================== */
.hollis-frontispiece {
  background: var(--bg-sheet);
  border: 1px solid var(--border-rule-strong);
  border-top: 5px solid var(--crimson-primary);
  border-radius: 4px;
  padding: var(--space-32) var(--space-48) var(--space-24) var(--space-48);
  text-align: center;
  margin-bottom: var(--space-32);
  box-shadow: var(--shadow-frame);
  position: relative;
}

.frontispiece-masthead {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-16);
  font-family: var(--font-display);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 2.2px;
  color: var(--crimson-primary);
  text-transform: uppercase;
  margin-bottom: var(--space-16);
}

.frontispiece-masthead .masthead-divider {
  color: var(--brass-accent);
}

.frontispiece-title {
  font-family: var(--font-display);
  font-size: 38px;
  font-weight: 800;
  letter-spacing: 3px;
  color: var(--text-primary);
  margin: 0 0 var(--space-12) 0;
  text-transform: uppercase;
}

/* Crisp double-line architectural borders */
.double-line-rule {
  height: 6px;
  border-top: 1px solid var(--brass-accent);
  border-bottom: 2px solid var(--brass-accent);
  margin: var(--space-16) auto;
  max-width: 640px;
}

.frontispiece-motto {
  font-family: var(--font-serif);
  font-style: italic;
  font-size: 19px;
  font-weight: 500;
  color: var(--crimson-hover);
  margin: 0 0 var(--space-12) 0;
  letter-spacing: 0.5px;
}

.frontispiece-preamble {
  font-family: var(--font-serif);
  font-size: 17.5px;
  line-height: 1.75;
  max-width: 840px;
  margin: 0 auto;
  color: var(--text-body);
}

/* ==========================================================================
   Archival Section Headers (Clean Double-Line Architectural Style)
   ========================================================================== */
.hollis-section-header {
  margin-top: var(--space-32);
  margin-bottom: var(--space-16);
  padding-bottom: var(--space-8);
  border-bottom: 3px double var(--brass-accent);
}

.section-header-content {
  display: flex;
  align-items: center;
  gap: var(--space-16);
}

.section-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 14px;
  background-color: var(--crimson-primary);
  color: var(--text-white);
  border: 1px solid var(--brass-accent);
  border-radius: 3px;
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  box-shadow: var(--shadow-button);
}

.section-title-group {
  flex: 1;
}

.section-title {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 1.2px;
  color: var(--text-primary);
  text-transform: uppercase;
  margin: 0;
}

.section-subtitle {
  font-family: var(--font-serif);
  font-size: 15.5px;
  color: var(--text-muted);
  font-style: italic;
  margin: 2px 0 0 0;
}

/* ==========================================================================
   Bordered Container Cards
   ========================================================================== */
[data-testid="stVerticalBlockBorderWrapper"] {
  background-color: var(--bg-sheet) !important;
  border: 1px solid var(--border-rule-strong) !important;
  border-radius: 4px !important;
  box-shadow: var(--shadow-card) !important;
}

/* ==========================================================================
   Accession Voucher & Registry Docket (Section I)
   ========================================================================== */
.accession-docket {
  background: var(--bg-surface-subtle);
  border: 1px solid var(--border-rule-strong);
  border-left: 4px solid var(--crimson-primary);
  border-radius: 4px;
  padding: var(--space-16);
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.accession-docket-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border-rule-strong);
  padding-bottom: var(--space-8);
  margin-bottom: var(--space-12);
}

.docket-title {
  font-family: var(--font-display);
  font-size: 12.5px;
  font-weight: 800;
  letter-spacing: 1.2px;
  color: var(--crimson-primary);
  text-transform: uppercase;
}

.docket-stamp {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  color: var(--brass-muted);
  background: #FFFFFF;
  border: 1px solid var(--brass-border);
  padding: 2px 8px;
  border-radius: 2px;
}

.docket-row {
  display: flex;
  justify-content: space-between;
  font-size: 14.5px;
  margin-bottom: var(--space-8);
}

.docket-label {
  font-family: var(--font-display);
  font-size: 12px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
}

.docket-value {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  max-width: 175px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.docket-prompt {
  font-family: var(--font-serif);
  font-size: 14.5px;
  color: var(--text-muted);
  line-height: 1.6;
  margin: 0;
}

/* ==========================================================================
   Buttons: Harvard Crimson Primary & Crisp Secondary
   ========================================================================== */
/* Main page primary buttons */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  background-color: var(--crimson-primary) !important;
  background-image: none !important;
  color: var(--text-white) !important;
  border: 1px solid var(--crimson-hover) !important;
  border-radius: 4px !important;
  font-family: var(--font-display) !important;
  font-size: 13.5px !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  padding: 10px 20px !important;
  box-shadow: var(--shadow-button) !important;
  transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] *,
.stButton > button[data-testid="baseButton-primary"] *,
.stButton > button[data-testid="stBaseButton-primary"] * {
  color: var(--text-white) !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  background-color: var(--crimson-hover) !important;
  border-color: var(--brass-accent) !important;
  box-shadow: 0 4px 12px var(--crimson-glow) !important;
  transform: translateY(-1px) !important;
}

/* Main page secondary buttons (Suggestions, etc.) */
.main .stButton > button,
.main .stButton > button[kind="secondary"],
.main [data-testid="stButton"] > button {
  background-color: #FFFFFF !important;
  background-image: none !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-rule-strong) !important;
  border-radius: 4px !important;
  font-family: var(--font-serif) !important;
  font-size: 14.5px !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  padding: 8px 16px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
  transition: all 0.2s ease !important;
}

.main .stButton > button:hover,
.main [data-testid="stButton"] > button:hover {
  background-color: var(--crimson-light) !important;
  border-color: var(--crimson-primary) !important;
  color: var(--crimson-primary) !important;
  box-shadow: 0 2px 6px var(--crimson-glow) !important;
  transform: translateY(-1px) !important;
}

/* ==========================================================================
   File Ingestion & Text Inputs
   ========================================================================== */
[data-testid="stFileUploader"] {
  background-color: var(--bg-sheet) !important;
  border: 2px dashed var(--border-rule-strong) !important;
  border-radius: 4px !important;
  padding: var(--space-16) !important;
}

[data-testid="stFileUploader"] section {
  background-color: transparent !important;
  border: none !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
  background-color: var(--crimson-primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--crimson-hover) !important;
  border-radius: 4px !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  box-shadow: var(--shadow-button) !important;
}

[data-testid="stFileUploader"] button:hover {
  background-color: var(--crimson-hover) !important;
  border-color: var(--brass-accent) !important;
}

[data-testid="stFileUploader"] button * {
  color: #FFFFFF !important;
}

.main [data-testid="stTextInput"] input {
  background-color: #FFFFFF !important;
  color: var(--text-primary) !important;
  border: 1.5px solid var(--border-rule-strong) !important;
  border-radius: 4px !important;
  font-family: var(--font-serif) !important;
  font-size: 16.5px !important;
  padding: 10px 14px !important;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

.main [data-testid="stTextInput"] input:focus {
  border-color: var(--crimson-primary) !important;
  box-shadow: 0 0 0 3px var(--crimson-glow) !important;
  outline: none !important;
}

/* ==========================================================================
   Deep Dark Oak Sidebar (#1E1E1E), Brass Accents (#C5A059) & Crisp Inputs
   ========================================================================== */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background-color: var(--oak-sidebar-bg) !important;
  border-right: 1px solid var(--oak-border) !important;
}

[data-testid="stSidebarCollapseButton"] button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--brass-accent) !important;
  padding: 4px !important;
  border-radius: 4px !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
  background: rgba(197, 160, 89, 0.12) !important;
  color: #FFFFFF !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] span {
  fill: var(--brass-accent) !important;
  color: var(--brass-accent) !important;
}

/* Sidebar Headings & Labels */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: var(--brass-bright) !important;
  font-family: var(--font-display) !important;
  letter-spacing: 0.8px !important;
  font-size: 13.5px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small {
  color: var(--oak-text-muted) !important;
  font-family: var(--font-serif) !important;
  font-size: 14.5px !important;
  line-height: 1.6 !important;
}

/* Sidebar Curator's Header Plaque */
.hollis-sidebar-header {
  background: var(--oak-surface-card);
  border: 1px solid var(--oak-border);
  border-top: 3px solid var(--crimson-primary);
  border-radius: 4px;
  padding: var(--space-16);
  margin-bottom: var(--space-16);
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

.hollis-shield-tag {
  display: inline-block;
  font-family: var(--font-display);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 2px;
  color: var(--brass-bright);
  border: 1px solid var(--brass-accent);
  padding: 2px 8px;
  border-radius: 2px;
  margin-bottom: var(--space-8);
  text-transform: uppercase;
}

.sidebar-plaque-title {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 1.5px;
  color: var(--oak-text-primary) !important;
  text-transform: uppercase;
  margin: 0 0 4px 0;
}

.sidebar-plaque-sub {
  font-family: var(--font-serif);
  font-size: 13px;
  font-style: italic;
  color: var(--oak-text-muted) !important;
  margin: 0;
}

/* Crisp Dark Oak Sidebar Input Fields */
[data-testid="stSidebar"] [data-testid="stTextInput"] > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] div,
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="input"] {
  background-color: var(--oak-input-bg) !important;
  border-radius: 4px !important;
}

[data-testid="stSidebar"] [data-baseweb="base-input"] {
  border: 1px solid var(--oak-border-strong) !important;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.5) !important;
  overflow: hidden !important;
  transition: border-color 0.2s ease !important;
}

[data-testid="stSidebar"] [data-baseweb="base-input"]:focus-within {
  border-color: var(--brass-accent) !important;
  box-shadow: 0 0 0 1px var(--brass-accent), inset 0 1px 3px rgba(0, 0, 0, 0.5) !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] {
  border: none !important;
  box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background-color: transparent !important;
  color: #FFFFFF !important;
  border: none !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
  padding: 8px 12px !important;
  box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
  color: #707070 !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--brass-bright) !important;
  padding: 0 10px !important;
  height: 100% !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] button:hover {
  background: rgba(197, 160, 89, 0.12) !important;
  color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] button svg {
  fill: var(--brass-bright) !important;
  width: 17px !important;
  height: 17px !important;
}

/* Sidebar Status Badges (Harvard Crimson Accented) */
.status-pill {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: var(--space-8) !important;
  padding: 10px 14px !important;
  border-radius: 4px !important;
  font-family: var(--font-display) !important;
  font-size: 12.5px !important;
  font-weight: 800 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  box-sizing: border-box !important;
  width: 100% !important;
}

[data-testid="stSidebar"] .status-pill-ready {
  background-color: var(--oak-surface-card) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--brass-accent) !important;
  border-left: 4px solid var(--crimson-primary) !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stSidebar"] .status-pill-ready span {
  color: #FFFFFF !important;
  font-family: var(--font-display) !important;
  font-weight: 800 !important;
}

[data-testid="stSidebar"] .status-pill-empty {
  background-color: var(--oak-surface-card) !important;
  color: var(--brass-bright) !important;
  border: 1px solid var(--oak-border-strong) !important;
  border-left: 4px solid var(--brass-muted) !important;
}

[data-testid="stSidebar"] .status-pill-empty span {
  color: var(--brass-bright) !important;
  font-family: var(--font-display) !important;
  font-weight: 800 !important;
}

/* Sidebar Curator's Ledger Card */
.curator-ledger-card {
  margin-top: var(--space-16);
  padding: var(--space-16);
  background: var(--oak-surface-card);
  border-radius: 4px;
  border: 1px solid var(--oak-border);
}

.ledger-header {
  margin: 0 0 var(--space-8) 0;
  font-family: var(--font-display);
  font-size: 11.5px;
  color: var(--brass-bright) !important;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.ledger-list {
  margin: 0;
  padding-left: var(--space-16);
  font-size: 13px;
  color: var(--oak-text-muted) !important;
  line-height: 1.7;
}

.ledger-list strong {
  color: var(--oak-text-primary) !important;
  font-weight: 600;
}

/* Sidebar Action Buttons: Harvard Crimson (#A51C30) */
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
  background-color: var(--crimson-primary) !important;
  background-image: none !important;
  color: #FFFFFF !important;
  border: 1px solid var(--brass-accent) !important;
  border-radius: 4px !important;
  font-family: var(--font-display) !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  box-shadow: var(--shadow-sidebar-button) !important;
  transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stButton button *,
[data-testid="stSidebar"] [data-testid="stButton"] button *,
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] * {
  color: #FFFFFF !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
}

[data-testid="stSidebar"] .stButton button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
  background-color: var(--crimson-hover) !important;
  border-color: var(--brass-bright) !important;
  box-shadow: 0 0 10px var(--brass-glow) !important;
  color: #FFFFFF !important;
  transform: translateY(-1px) !important;
}

/* ==========================================================================
   Reading Room Portfolio / Unified Dispatch Container (Section II)
   ========================================================================== */
.st-key-dispatch_portfolio [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-sheet) !important;
  border: 1px solid var(--border-rule-strong) !important;
  border-top: 4px solid var(--crimson-primary) !important;
  border-radius: 4px !important;
  padding: var(--space-32) !important;
  margin-top: var(--space-24) !important;
  margin-bottom: var(--space-24) !important;
  box-shadow: var(--shadow-frame) !important;
}

.dispatch-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 3px double var(--brass-accent);
  padding-bottom: var(--space-16);
  margin-bottom: var(--space-24);
}

.dispatch-title {
  font-family: var(--font-display);
  font-size: 21px;
  font-weight: 800;
  color: var(--crimson-primary);
  letter-spacing: 1px;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: var(--space-12);
}

.dispatch-reference {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 1.2px;
  font-weight: 600;
  color: var(--brass-muted);
  text-transform: uppercase;
}

.dispatch-reading {
  font-family: var(--font-serif) !important;
  font-size: 18px !important;
  line-height: 1.9 !important;
  color: var(--text-primary) !important;
  padding: 0 var(--space-8);
}

.dispatch-reading p {
  color: var(--text-primary) !important;
  font-size: 18px !important;
  line-height: 1.9 !important;
  margin-bottom: var(--space-16) !important;
}

.dispatch-reading strong {
  color: var(--crimson-primary) !important;
  font-weight: 700 !important;
}

.dispatch-reading blockquote {
  background: var(--bg-surface-subtle);
  border-left: 4px solid var(--crimson-primary);
  border-radius: 3px;
  padding: var(--space-16) var(--space-24);
  margin: var(--space-20) 0;
  color: var(--text-body);
  font-style: italic;
  font-size: 17.5px;
  line-height: 1.8;
}

/* Institutional Attestation Sign-off */
.archivist-signoff {
  margin-top: var(--space-32);
  padding-top: var(--space-20);
  border-top: 1px solid var(--border-rule-subtle);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.signoff-attestation {
  flex: 1;
}

.attestation-text {
  font-family: var(--font-serif);
  font-size: 14.5px;
  font-style: italic;
  color: var(--text-muted);
  margin: 0 0 4px 0;
}

.attestation-meta {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text-faint);
  margin: 0;
}

.signoff-seal-and-sig {
  display: flex;
  align-items: center;
  gap: var(--space-20);
}

.hollis-institutional-seal {
  width: 58px;
  height: 58px;
  border-radius: 50%;
  border: 2px solid var(--crimson-primary);
  outline: 1px solid var(--brass-accent);
  outline-offset: -5px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--crimson-light);
  color: var(--crimson-primary);
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 10px;
  letter-spacing: 1px;
}

.seal-veritas {
  font-size: 9px;
  letter-spacing: 1.5px;
  color: var(--crimson-primary);
  font-weight: 900;
}

.seal-year {
  font-family: var(--font-mono);
  font-size: 8px;
  color: var(--brass-muted);
  font-weight: 700;
}

.signoff-signature-block {
  text-align: right;
}

.signoff-signature {
  font-family: var(--font-serif);
  font-size: 21px;
  font-weight: 700;
  color: var(--crimson-primary);
  line-height: 1.2;
  margin-bottom: 2px;
}

.signoff-title {
  font-family: var(--font-display);
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-muted);
  font-weight: 700;
}

/* ==========================================================================
   HOLLIS Card Catalog Slips (Evidentiary Extracts)
   ========================================================================== */
.catalog-card {
  background: var(--bg-sheet);
  border: 1px solid var(--border-rule-subtle);
  border-left: 4px solid var(--crimson-primary);
  border-radius: 4px;
  padding: var(--space-16) var(--space-20);
  margin-bottom: var(--space-16);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.catalog-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border-rule-subtle);
  padding-bottom: var(--space-8);
  margin-bottom: var(--space-12);
}

.catalog-card-callno {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
}

.catalog-call-label {
  color: var(--crimson-primary);
  font-weight: 700;
  margin-right: 4px;
}

.catalog-call-code {
  color: var(--text-primary);
  font-weight: 700;
}

.catalog-call-sep {
  margin: 0 var(--space-8);
  color: var(--brass-accent);
}

.catalog-folio-meta {
  font-family: var(--font-serif);
  font-style: italic;
  font-size: 13.5px;
}

.catalog-card-stamp {
  font-family: var(--font-display);
  font-size: 10px;
  font-weight: 800;
  color: var(--crimson-primary);
  border: 1px solid var(--crimson-border);
  background: var(--crimson-light);
  padding: 2px 8px;
  border-radius: 2px;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.catalog-card-body {
  font-family: var(--font-serif);
  font-size: 16.5px;
  font-style: italic;
  color: var(--text-primary);
  line-height: 1.8;
  margin-bottom: var(--space-12);
}

.catalog-quote {
  margin: 0;
  padding: 0;
}

.catalog-card-footer {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text-faint);
  border-top: 1px dashed var(--border-rule-subtle);
  padding-top: var(--space-8);
}

/* Expander Component */
[data-testid="stExpander"] {
  background-color: var(--bg-sheet) !important;
  border: 1px solid var(--border-rule-strong) !important;
  border-radius: 4px !important;
}

[data-testid="stExpander"] summary {
  color: var(--text-primary) !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 14.5px !important;
  letter-spacing: 0.5px !important;
}

/* ==========================================================================
   Accessibility: Reduced Motion Preference
   ========================================================================== */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
</style>
"""

st.html(HOLLIS_ARCHIVAL_CSS)

# ---------------------------------------------------------------------------
# Core Archival Database Logic
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate and cache the vector embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def check_catalogue_has_records() -> bool:
    """Check if the persistent Chroma catalogue contains indexed folios."""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return False
    try:
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
        return vs._collection.count() > 0
    except Exception:
        return False


def get_catalogue_record_count() -> int:
    """Return the total number of chunks currently bound in the catalogue."""
    try:
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
        return vs._collection.count()
    except Exception:
        return 0


def index_document_into_catalogue(file_path: str, is_pdf: bool = True) -> int:
    """Segment a manuscript and bind it into the persistent Chroma catalogue."""
    if is_pdf:
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    return len(chunks)


def load_retriever():
    """Load the catalogue retriever with Maximal Marginal Relevance."""
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def get_prompt() -> ChatPromptTemplate:
    """Return the scholarly prompt for the AI research assistant."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the Chief Research Archivist at the Harvard Library & HOLLIS Digital Archives.
You respond with scholarly authority, academic precision, and elevated diction, grounded strictly and exclusively upon the deposited manuscripts.

Rules of the Archives:
1. Ground your answer strictly and exclusively upon the provided context excerpts.
2. If the inquiry cannot be verified within the catalogued folios, state clearly:
   "According to the deposited records, no verifiable account of this matter is catalogued within this volume."
3. Cite specific textual points, methodologies, and principles directly from the passages provided.
""",
            ),
            (
                "human",
                """Catalogued Folio Excerpts:
{context}

Scholar's Inquiry:
{question}
""",
            ),
        ]
    )


def summarize_locally(text: str, question: str) -> str:
    """Scholarly direct transcription if no Mistral API key is configured."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return (
            "According to the deposited records, no verifiable account of this matter "
            "is catalogued within this volume."
        )

    matched = [line for line in lines if any(w.lower() in line.lower() for w in question.split() if len(w) > 3)]
    if matched:
        excerpt = " ".join(matched[:3])
    else:
        excerpt = " ".join(lines[:3])

    return (
        f"**From the Chief Archivist's Direct Transcription (Local Extraction):**\n\n"
        f"Upon examining the catalogued folios, the primary passage preserved within our records states:\n\n"
        f"> *\"{excerpt}\"*\n\n"
        f"*(Note: To enable generative scholarly syntheses and extended analytical prose, "
        f"ensure MISTRAL_API_KEY is configured in your environment.)*"
    )


# ---------------------------------------------------------------------------
# Sidebar: The Curator's Desk (Dark Oak #1E1E1E & Brass Accents)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.html(
        """
        <aside role="complementary" aria-label="Curator's Control & Registry">
          <div class="hollis-sidebar-header">
            <span class="hollis-shield-tag">VERITAS</span>
            <h2 class="sidebar-plaque-title">Curator's Desk</h2>
            <p class="sidebar-plaque-sub">HOLLIS Special Collections Registry</p>
          </div>
        </aside>
        """
    )

    # Live Catalogue Status Badge (Harvard Crimson Accented)
    has_records = check_catalogue_has_records()
    total_records = get_catalogue_record_count() if has_records else 0

    st.markdown("#### **Catalogue Status**")
    if has_records:
        st.html(
            f"""
            <div class="status-pill status-pill-ready">
              <span>●</span> <span>{total_records} Passages Catalogued</span>
            </div>
            """
        )
    else:
        st.html(
            """
            <div class="status-pill status-pill-empty">
              <span>○</span> <span>Repository Awaiting Folios</span>
            </div>
            """
        )

    st.html(
        """
        <div class="curator-ledger-card">
          <div class="ledger-header">System Specifications</div>
          <ul class="ledger-list">
            <li><strong>Index Store:</strong> Chroma Vector DB</li>
            <li><strong>Embeddings:</strong> MiniLM-L6-v2 Semantic</li>
            <li><strong>Retrieval:</strong> Maximal Marginal Relevance</li>
            <li><strong>Classification:</strong> HOLLIS Special Collections</li>
          </ul>
        </div>
        """
    )

    st.markdown("---")

    # Sample Treatise Loader Button (Harvard Crimson)
    if SAMPLE_TREATISE_PATH.exists():
        if st.button("Bind Sample Treatise into Codex", width="stretch", help="Load the 1888 Philosophical Treatise"):
            with st.spinner("Binding sample treatise into the catalogue..."):
                count = index_document_into_catalogue(str(SAMPLE_TREATISE_PATH), is_pdf=False)
            st.session_state.pop("retriever", None)
            st.toast(f"Sample Treatise bound ({count} passages indexed).", icon="📖")
            st.rerun()

    # Archival Reset Button (Harvard Crimson)
    if st.button("Purge Master Catalogue", width="stretch", help="Remove all current indexed volumes"):
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        st.session_state.pop("retriever", None)
        st.toast("Catalogue records cleared by the Curator.", icon="🧹")
        st.rerun()


# ---------------------------------------------------------------------------
# Main Stage: Master Frontispiece / Title Page
# ---------------------------------------------------------------------------
st.html(
    """
    <header role="banner" class="hollis-frontispiece">
      <div class="frontispiece-masthead">
        <span>HARVARD UNIVERSITY ARCHIVES & SPECIAL COLLECTIONS</span>
        <span class="masthead-divider">•</span>
        <span>HOLLIS DIGITAL CODEX MS-1888-ATH</span>
      </div>
      <h1 class="frontispiece-title">The Royal Athenæum</h1>
      <div class="double-line-rule"></div>
      <p class="frontispiece-motto">Veritas et Eruditio — Archival Folios & Retrieval-Augmented Inquest</p>
      <div class="double-line-rule"></div>
      <p class="frontispiece-preamble">
        An academic research environment engineered for scholars and archivists. Deposit authenticated manuscripts
        into the institutional vector repository to examine, cross-reference, and interrogate primary folios with grounded precision.
      </p>
    </header>
    """
)

# ---------------------------------------------------------------------------
# Chapter I: Manuscript Deposit (Intake)
# ---------------------------------------------------------------------------
st.html(
    """
    <section class="hollis-section-header" aria-labelledby="manuscript-deposit-title">
      <div class="section-header-content">
        <div class="section-badge">Section I</div>
        <div class="section-title-group">
          <h2 class="section-title" id="manuscript-deposit-title">Manuscript Ingestion & Folio Cataloguing</h2>
          <p class="section-subtitle">Deposit PDF treatises, scholarly monographs, or transcripts into the permanent semantic index.</p>
        </div>
      </div>
    </section>
    """
)

with st.container(border=True):
    col_upload, col_action = st.columns([3, 2], gap="large")

    with col_upload:
        st.markdown("**Select a manuscript for archival transcription:**")
        uploaded_file = st.file_uploader(
            "Select a document for archival transcription",
            type=["pdf", "txt", "md"],
            help="Accepted formats: PDF manuscripts, plain text, and markdown files.",
            label_visibility="collapsed",
        )

    with col_action:
        if uploaded_file:
            st.html(
                f"""
                <div class="accession-docket">
                  <div class="accession-docket-header">
                    <span class="docket-title">Archival Accession Voucher</span>
                    <span class="docket-stamp">DOC № 1888-IX</span>
                  </div>
                  <div class="accession-docket-body">
                    <div class="docket-row">
                      <span class="docket-label">Manuscript:</span>
                      <span class="docket-value" title="{uploaded_file.name}">{uploaded_file.name}</span>
                    </div>
                    <div class="docket-row">
                      <span class="docket-label">File Size:</span>
                      <span class="docket-value">{uploaded_file.size / 1024:.1f} KB</span>
                    </div>
                    <div class="docket-row">
                      <span class="docket-label">Classification:</span>
                      <span class="docket-value">HOLLIS Standard</span>
                    </div>
                  </div>
                </div>
                """
            )
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            if st.button("Inscribe Folio into Catalogue", type="primary", width="stretch"):
                suffix = Path(uploaded_file.name).suffix or ".pdf"
                is_pdf = suffix.lower() == ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                with st.spinner("The Chief Archivist is parsing, segmenting, and embedding the folio..."):
                    chunk_count = index_document_into_catalogue(temp_path, is_pdf=is_pdf)

                st.session_state.pop("retriever", None)
                st.toast(f"Manuscript bound successfully ({chunk_count} passages indexed).", icon="📚")
                st.rerun()
        else:
            st.html(
                """
                <div class="accession-docket">
                  <div class="accession-docket-header">
                    <span class="docket-title">Registry Docket</span>
                    <span class="docket-stamp">VAULT READY</span>
                  </div>
                  <div class="accession-docket-body">
                    <p class="docket-prompt">
                      <strong>Awaiting Document:</strong> Select a PDF document or scholarly text from your filesystem to begin indexing into the archives, or load the sample treatise via the Curator's Desk.
                    </p>
                  </div>
                </div>
                """
            )


# ---------------------------------------------------------------------------
# Chapter II: Reading Room & Scholarly Consultation
# ---------------------------------------------------------------------------
st.html(
    """
    <section class="hollis-section-header" aria-labelledby="reading-room-title">
      <div class="section-header-content">
        <div class="section-badge">Section II</div>
        <div class="section-title-group">
          <h2 class="section-title" id="reading-room-title">Archival Consultation & Scholarly Inquest</h2>
          <p class="section-subtitle">Submit inquiries to the archives. Retrieval algorithms evaluate source folios to synthesize grounded decrees.</p>
        </div>
      </div>
    </section>
    """
)

if not has_records:
    st.warning(
        "Notice from the Curator: No manuscripts currently reside within the Codex. "
        "Deposit a PDF volume in Section I above, or click 'Bind Sample Treatise' in the sidebar to awaken the archives.",
        icon="📖",
    )
else:
    # Pre-crafted scholarly suggestions
    st.markdown("**Suggested Archival Inquiries:**")
    suggestion_cols = st.columns([1, 1, 1], gap="medium")
    selected_suggestion = None
    with suggestion_cols[0]:
        if st.button("Expound upon core thesis", width="stretch"):
            selected_suggestion = "What is the primary thesis or core subject of this manuscript?"
    with suggestion_cols[1]:
        if st.button("Detail foundational tenets", width="stretch"):
            selected_suggestion = "What are the primary methods, components, or foundational principles detailed here?"
    with suggestion_cols[2]:
        if st.button("Summarise key conclusions", width="stretch"):
            selected_suggestion = "What notable conclusions or deductions are drawn in this work?"

    # Query Input & Consultation
    default_query = selected_suggestion if selected_suggestion else ""
    st.markdown("**Present your inquiry to the Research Scribe:**")
    
    col_input, col_btn = st.columns([5, 1], gap="medium")
    with col_input:
        query_input = st.text_input(
            "Enter your scholarly inquiry",
            value=default_query,
            placeholder="e.g. 'What observations are recorded concerning the mechanisms described?'",
            label_visibility="collapsed",
        )
    with col_btn:
        consult_clicked = st.button("Consult", type="primary", width="stretch")

    active_query = query_input.strip() if (consult_clicked or query_input) else ""

    if active_query:
        with st.spinner("The Research Scribe is evaluating the folios via Maximal Marginal Relevance..."):
            try:
                retriever = load_retriever()
                docs = retriever.invoke(active_query)
            except Exception:
                retriever = load_retriever()
                docs = retriever.invoke(active_query)

            context = "\n\n".join([doc.page_content for doc in docs])

            api_key = os.getenv("MISTRAL_API_KEY", "")

            if api_key:
                try:
                    final_prompt = get_prompt().invoke({
                        "context": context,
                        "question": active_query,
                    })
                    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)
                    result = llm.invoke(final_prompt)
                    answer_text = result.content if hasattr(result, "content") else str(result)
                except Exception as err:
                    answer_text = (
                        f"**Communication Disruption:** An issue occurred while contacting Mistral AI ({err}).\n\n"
                        + summarize_locally(context, active_query)
                    )
            else:
                answer_text = summarize_locally(context, active_query)

        # Master Reading Room Portfolio (Unified Container)
        with st.container(key="dispatch_portfolio", border=True):
            st.html(
                """
                <header class="dispatch-banner">
                  <div class="dispatch-title">
                    <span>Archival Decree & Research Synthesis</span>
                  </div>
                  <div class="dispatch-reference">
                    HOLLIS CODEX ARCHIVES • DISPATCH NO. CLXXXVIII
                  </div>
                </header>
                """
            )

            st.markdown(
                f'<div class="dispatch-reading">\n\n{answer_text}\n\n</div>',
                unsafe_allow_html=True,
            )

            st.html(
                """
                <footer class="archivist-signoff">
                  <div class="signoff-attestation">
                    <p class="attestation-text">Verified and transcribed in accordance with Harvard Library & HOLLIS Archival standards.</p>
                    <p class="attestation-meta">Concordance Engine: Maximal Marginal Relevance • HNSW Chroma Index</p>
                  </div>
                  <div class="signoff-seal-and-sig">
                    <div class="hollis-institutional-seal" aria-hidden="true">
                      <span class="seal-veritas">VERITAS</span>
                      <span class="seal-year">1888</span>
                    </div>
                    <div class="signoff-signature-block">
                      <div class="signoff-signature">Archibald Sterling, Ph.D.</div>
                      <div class="signoff-title">Curator of Special Collections & Manuscripts</div>
                    </div>
                  </div>
                </footer>
                """
            )

        # Supporting Card Catalog Evidence
        if docs:
            with st.expander(f"Inspect Supporting Folio Extracts & Provenance ({len(docs)} Passages Retrieved)", expanded=True):
                st.markdown(
                    "<p style='color:#565961; font-style:italic; margin-bottom:16px; font-size:15px;'>"
                    "The following verbatim excerpts were retrieved from the primary manuscripts to substantiate the decree above:"
                    "</p>",
                    unsafe_allow_html=True,
                )
                for idx, doc in enumerate(docs, start=1):
                    page_info = doc.metadata.get("page", 1)
                    source_info = doc.metadata.get("source", "Archival Volume")
                    st.html(
                        f"""
                        <article class="catalog-card" aria-label="Index Card No. {idx}">
                          <header class="catalog-card-header">
                            <div class="catalog-card-callno">
                              <span class="catalog-call-label">CALL NO:</span>
                              <span class="catalog-call-code">HOLLIS-ATH-FOL.{idx}</span>
                              <span class="catalog-call-sep">•</span>
                              <span class="catalog-folio-meta">Folio Page {page_info}</span>
                            </div>
                            <div class="catalog-card-stamp">
                              <span>HOLLIS VERIFIED</span>
                            </div>
                          </header>
                          <div class="catalog-card-body">
                            <blockquote class="catalog-quote">
                              "{doc.page_content.strip()}"
                            </blockquote>
                          </div>
                          <footer class="catalog-card-footer">
                            <span><strong>Manuscript:</strong> {Path(source_info).name}</span>
                            <span><strong>Concordance:</strong> Maximal Marginal Relevance</span>
                          </footer>
                        </article>
                        """
                    )

# ---------------------------------------------------------------------------
# Footer: Athenæum Gazette & Imprint
# ---------------------------------------------------------------------------
st.html(
    """
    <footer role="contentinfo" style="margin-top: 48px; padding-top: 24px; border-top: 3px double var(--brass-accent); text-align: center; color: var(--text-faint); font-size: 13.5px;">
      <p style="margin: 0 0 8px 0; font-family: var(--font-display); font-weight: 700; letter-spacing: 1.5px; color: var(--text-muted);">
        THE ROYAL ATHENÆUM • HARVARD LIBRARY & HOLLIS DIGITAL ARCHIVES
      </p>
      <p style="margin: 0; font-family: var(--font-serif); font-style: italic; color: var(--text-faint); font-size: 14px;">
        Chroma Vector Repository • MiniLM-L6-v2 Embeddings • Mistral AI Neural Inquest
      </p>
    </footer>
    """
)
