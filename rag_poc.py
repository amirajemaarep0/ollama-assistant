from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def test_rag():
    llm = Ollama(model="llama3")
    
    with open("rag_poc.py", "r") as f:
        code_content = f.read()

    template = """
    You are an AI coding assistant. Explain what the following Python code does:
    
    {code}
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    print("Sending query to Llama 3...")
    response = chain.invoke({"code": code_content})
    print("\n--- Response ---")
    print(response)

if __name__ == "__main__":
    test_rag()
