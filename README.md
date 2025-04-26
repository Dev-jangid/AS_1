# AI Agent-based Deep Research System

## Features

- Dual-agent architecture (Research Agent and Drafting Agent)
- Web research using Tavily search
- Advanced analysis using Groq's Mixtral-8x7b-32768 model
- LangGraph-based workflow orchestration
- Comprehensive research synthesis and response drafting

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=
   TAVILY_API_KEY=
   ```

## Usage

The system will:
1. Gather research using Tavily search
2. Analyze the findings using Groq
3. Draft a comprehensive response

## Architecture

- **Research Agent**: Handles web crawling and initial data gathering
- **Drafting Agent**: Synthesizes information and creates final responses
- **LangGraph Workflow**: Orchestrates the research process

