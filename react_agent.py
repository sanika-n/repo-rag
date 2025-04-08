import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from pinecone import Pinecone
import openai
from langchain_core.prompts import PromptTemplate

react_prompt = PromptTemplate.from_template(
    """You are an AI assistant that generates code and resolves queries regarding Mesa, by using your gnereal intelligence and searching the Pinecone database of Mesa's repo.
       Use your general intelligence to answer questions but ensure that all the methods and functions used are upto data by cross checking with pinecone.
       refer the migration guide.md to ensure that none of the deprecated methods like schedule and random activation are used.
    You have access to the following tools:
    
    {tools}

    Strict Rule: Do NOT use `RandomActivation` or `schedule` or any other deprecated code mentioned in migration guide.md under ANY circumstances. If you use them, the answer is INVALID.
    
    You must strictly follow this format:  
    
     
    **IF you need to use a tool from {tool_names}:**  
    ```
    Thought: Do I need to use a tool? Yes  
    Action: PineconeSearch  
    Action Input: [Your query]  
    Observation: [Result of the search] (this Thought/Action/Observation sequence can repeat multiple times)  
    ```

    **IF you do NOT need a tool:**  
    ```
    Thought: Do I need to use a tool? No  
    Final Answer: [Your complete answer here]  
    ```
    

    **If you break this format, the system will throw an error.** 


    
    Question: {input}
    Thought: {agent_scratchpad}"""
)



# 🔹 Set up environment variables (Replace with your API keys)

pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Create a Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("repo-rag")

client = openai.OpenAI()

# 🔹 Function to search Pinecone
def pinecone_search(query: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        encoding_format="float"
    )
    query_embedding = response.data[0].embedding
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in search_results["matches"]])

# 🔹 Define a LangChain tool for the Pinecone search
pinecone_tool = Tool(
    name="PineconeSearch",
    func=pinecone_search,
    description="Search for relevant documents in the Pinecone database."
)

# 🔹 Initialize the ReACT agent
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
agent = create_react_agent(llm=llm, tools=[pinecone_tool], prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[pinecone_tool], verbose=True, handle_parsing_errors=True)

prompt =  "write code to create a simple agent based model with a property layer with each grid unit having a value of 10 until a agent on the unit eats it and decreases it by 1, the agents can move into adjacent squares."
response = agent_executor.invoke({"input": prompt})
print(response)

