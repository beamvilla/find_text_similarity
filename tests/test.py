import sys
sys.path.append("./")

from src.text_embedding.openai_embedding import OpenAIEmbedding


openai_embedding_module = OpenAIEmbedding(model="text-embedding-3-large")
print(len(openai_embedding_module.get_embedding("เนื้อร้านนี้อร่อยมาก")))