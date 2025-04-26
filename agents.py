from typing import List, Dict, Any
from langchain.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os

class ResearchAgent:
    """Agent responsible for gathering and analyzing research data."""
    
    def __init__(self, tavily_api_key: str):
        self.tavily = TavilySearchResults(api_key=tavily_api_key)
        self.groq = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def gather_research(self, query: str) -> List[Dict[str, Any]]:
        """Gather research using Tavily search."""
        print("Research Agent: Gathering information from the web...")
        results = self.tavily.run(query)
        return results

    def analyze_research(self, query: str, research_results: List[Dict[str, Any]]) -> str:
        """Analyze and synthesize research findings."""
        print("Research Agent: Analyzing gathered information...")
        
        context = "\n".join([
            f"Source: {r['title']}\nContent: {r['snippet']}\nURL: {r['url']}"
            for r in research_results
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst. Your task is to:
            1. Analyze the provided research materials
            2. Identify key insights and patterns
            3. Synthesize the information into a coherent analysis
            4. Highlight any contradictions or gaps in the research
            5. Provide recommendations for further research if needed
            
            Be thorough, objective, and maintain academic rigor in your analysis."""),
            ("human", f"""Research Query: {query}
            
            Research Materials:
            {context}
            
            Please provide a detailed analysis of the research findings.""")
        ])

        response = self.groq.invoke(prompt.format_messages())
        return response.content

class DraftingAgent:
    """Agent responsible for creating well-structured responses."""
    
    def __init__(self):
        self.groq = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def draft_response(self, query: str, research_analysis: str) -> str:
        """Draft a comprehensive response based on research analysis."""
        print("Drafting Agent: Creating final response...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content drafter. Your task is to:
            1. Create a well-structured, comprehensive response
            2. Use clear and professional language
            3. Include relevant citations and sources
            4. Maintain objectivity and accuracy
            5. Format the response for easy reading
            
            The response should be informative, engaging, and authoritative."""),
            ("human", f"""Original Query: {query}
            
            Research Analysis:
            {research_analysis}
            
            Please draft a comprehensive response to the original query.""")
        ])

        response = self.groq.invoke(prompt.format_messages())
        return response.content 