from typing import List
from langchain.embeddings.base import Embeddings
from openai import OpenAI
import os

class OpenAIEmbeddings(Embeddings):
    """
    Custom OpenAI Embeddings class that implements the LangChain Embeddings interface.
    
    This class provides methods to generate embeddings for documents and queries using OpenAI's API.
    
    Args:
        model_name (str): The name of the OpenAI embedding model to use. 
                         Defaults to "text-embedding-3-small".
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the OpenAIEmbeddings with the specified model."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
        """
        res = self.client.embeddings.create(
            model=self.model_name, 
            input=texts
        )
        return [d.embedding for d in res.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        This method is required by the LangChain Embeddings interface.
        """
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        This method is required by the LangChain Embeddings interface.
        """
        return self._encode([text])[0]
