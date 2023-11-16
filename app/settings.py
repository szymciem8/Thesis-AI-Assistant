# OpenAI models
AI_MODEL_OPTIONS: list[str] = [
    'text-curie-001',
    'davinci-002',
]

AI_CHAT_MODELS: list[str] = [
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-32k-0613',
    'gpt-3.5-turbo-16k',
]

AI_LLM_MODELS = AI_MODEL_OPTIONS + AI_CHAT_MODELS

# Elasticsearch
ELASTICSEARCH_HOST = 'http://elasticsearch:9200'
ES_ARTICLES_INDEX = 'articles'