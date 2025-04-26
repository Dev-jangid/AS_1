import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage
from agents import ResearchAgent, DraftingAgent

# Load environment variables
load_dotenv()

class ResearchAgent:
    def __init__(self, tavily_api_key: str):
        self.tavily = TavilySearchResults(api_key=tavily_api_key)
        self.groq = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def gather_research(self, query: str) -> List[Dict[str, Any]]:
        """Gather research using Tavily search."""
        results = self.tavily.run(query)
        return results

    def analyze_research(self, query: str, research_results: List[Dict[str, Any]]) -> str:
        """Analyze and synthesize research findings."""
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
    def __init__(self):
        self.groq = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def draft_response(self, query: str, research_analysis: str) -> str:
        """Draft a comprehensive response based on research analysis."""
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

def build_research_graph(research_agent: ResearchAgent, drafting_agent: DraftingAgent) -> Graph:
    """Build the LangGraph workflow for the research system."""
    
    # Define the state
    class ResearchState:
        def __init__(self):
            self.query = ""
            self.research_results = []
            self.research_analysis = ""
            self.final_response = ""

    # Create the graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("gather_research", research_agent.gather_research)
    workflow.add_node("analyze_research", research_agent.analyze_research)
    workflow.add_node("draft_response", drafting_agent.draft_response)

    # Define edges
    workflow.add_edge("gather_research", "analyze_research")
    workflow.add_edge("analyze_research", "draft_response")

    # Set entry point
    workflow.set_entry_point("gather_research")

    # Compile the graph
    return workflow.compile()

def main():
    # Initialize agents
    research_agent = ResearchAgent(os.getenv("TAVILY_API_KEY"))
    drafting_agent = DraftingAgent()

    # Build the research graph
    research_graph = build_research_graph(research_agent, drafting_agent)

    # Example query
    query = "What are the latest advancements in quantum computing?"

    print("Starting research process...")
    print(f"Query: {query}\n")

    # Run the research process
    result = research_graph.invoke({"query": query})
    
    print("\nFinal Response:")
    print(result["final_response"])

if __name__ == "__main__":
    main() 
