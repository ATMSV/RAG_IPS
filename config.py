"""
Конфигурация RAG системы для IPS PLM
"""
import os
from pathlib import Path

# Пути к файлам
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
TEMP_DIR = BASE_DIR / "temp"

# Создаем директории если их нет
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True) 
TEMP_DIR.mkdir(exist_ok=True)

# LMStudio API настройки
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # Стандартный порт LMStudio
LMSTUDIO_API_KEY = "lm-studio"  # LMStudio не требует реального ключа
LMSTUDIO_MODEL = "qwen2.5-coder-14b-instruct"  # Более точная модель для техдокументации

# Модель эмбеддингов (локальная)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Поддерживает русский

# ChromaDB настройки
CHROMA_COLLECTION_NAME = "ips_documentation"

# RAG настройки
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVED_DOCS = 5  # Увеличили обратно для более мощной модели

# API настройки
API_HOST = "localhost"
API_PORT = 8000

# Поддерживаемые форматы документов
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]

# Максимальная длина контекста для LLM
MAX_CONTEXT_LENGTH = 4000