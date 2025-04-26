# AI Agent-based Deep Research System

A sophisticated research system that combines web crawling with advanced AI analysis using Groq's fast inference capabilities.

## Features

- Dual-agent architecture (Research Agent and Drafting Agent)
- Web research using Tavily search
- Advanced analysis using Groq's Mixtral-8x7b-32768 model
- LangGraph-based workflow orchestration
- Comprehensive research synthesis and response drafting

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. Get your API keys:
   - Groq API key: Sign up at [Groq](https://console.groq.com/)
   - Tavily API key: Sign up at [Tavily](https://tavily.com/)

## Usage

Run the research system:
```bash
python research_system.py
```

The system will:
1. Gather research using Tavily search
2. Analyze the findings using Groq
3. Draft a comprehensive response

## Architecture

- **Research Agent**: Handles web crawling and initial data gathering
- **Drafting Agent**: Synthesizes information and creates final responses
- **LangGraph Workflow**: Orchestrates the research process

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- API keys for Groq and Tavily

## License

MIT License 