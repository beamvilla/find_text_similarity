import sys
sys.path.append("./")

from text_similarity import calculate_cosine, OpenAIEmbedding


openai_embedding_module = OpenAIEmbedding(model="text-embedding-3-large")
a = openai_embedding_module.get_embedding("เนื้อร้านนี้อร่อยมาก")
b = openai_embedding_module.get_embedding("เนื้อร้านนี้เฉยๆ")
score = calculate_cosine(a, b)
print(score)