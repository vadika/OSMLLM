from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import torch

class OSMQueryInterface:
    def __init__(self):
        # Initialize a smaller model suitable for geographic queries
        self.pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Given the following OpenStreetMap features:
            {context}
            
            Answer the following query about these features:
            {query}
            
            Answer:"""
        )
        
        self.output_parser = StrOutputParser()
    
    def process_query(self, query: str, context_features: list) -> str:
        """Process a natural language query about OSM features"""
        context = "\n".join([str(feature) for feature in context_features])
        formatted_prompt = self.prompt.format(query=query, context=context)
        return self.llm.invoke(formatted_prompt)
