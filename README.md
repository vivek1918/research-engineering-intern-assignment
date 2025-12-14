# research-engineering-intern-assignment

# 🚀 SocialPulse – AI-Powered Social Media Analytics Platform

SocialPulse is a **full-stack AI-driven social media analytics platform** that analyzes Reddit content to extract **engagement trends, sentiment insights, influential users, trending topics**, and enables **natural language querying via an AI assistant**.

This project combines **FastAPI, Next.js, Retrieval-Augmented Generation (RAG), embeddings, and LLMs (Groq)** to deliver an interactive analytics dashboard and conversational AI experience.

---

## 🔗 Live Demo

🚀 **Try SocialPulse Live:**  
https://drive.google.com/file/d/1oS7tTE8HxTOBaMxCCnjbgv4onOI8FynV/view?usp=sharing

> ⚠️ Note: The demo may take a few seconds to load initially due to cold start on the backend.


## ✨ Key Features

### 🧠 AI Analytics Assistant
- Ask **natural language questions** about social media data
- Uses **RAG (Retrieval-Augmented Generation)** with embeddings
- Generates:
  - Structured insights
  - Statistical analysis
  - Actionable recommendations
  - Auto-generated charts (bar, line, pie)

---

### 📊 Analytics Dashboard
- **Total Posts & Engagement**
- **Weekly Post Distribution**
- **Engagement Trends (Monthly)**
- **Sentiment Trends (Positive / Neutral / Negative)**
- **Top Posts by Engagement**
- **Influential Users**
- **Trending Keywords**
- **Platform Distribution**

---

### 🔍 Advanced Data Processing
- Parses large **Reddit JSONL datasets**
- Separates **posts and comments**
- Computes engagement using:

### 🌐 News Integration
- Fetches live news via **NewsAPI**
- Supports:
- Search queries
- Categories
- Date range filtering
- In-memory caching for faster responses

## 🧠 AI & ML Stack

Embeddings - Sentence Transformers (all-MiniLM-L6-v2) 
Vector Search - KNN (cosine similarity) 
LLM - Groq (LLaMA 3.3 / 3.1 models) 
RAG - Context-aware retrieval 
Sentiment - Heuristic + engagement-based 
Keyword Extraction - LLM + NLP fallback 
