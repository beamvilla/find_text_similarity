import os
import openai
import numpy as np
from dotenv import load_dotenv


class OpenAIEmbedding:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        load_dotenv()
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_embedding(
        self, 
        text: str
    ) -> np.array:
        text = text.replace("\n", " ")
        return np.array(openai.Embedding.create(input = [text], model=self.model).data[0].embedding) 