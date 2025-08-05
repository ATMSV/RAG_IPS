"""
Менеджер векторной базы данных для RAG системы IPS PLM
Использует ChromaDB для хранения эмбеддингов документов
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
from pathlib import Path
from config import (
    VECTOR_DB_DIR, 
    CHROMA_COLLECTION_NAME, 
    EMBEDDING_MODEL,
    MAX_RETRIEVED_DOCS
)

class VectorDatabase:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Настройка ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Создаем или получаем коллекцию
        try:
            self.collection = self.client.get_collection(CHROMA_COLLECTION_NAME)
            print(f"Загружена существующая коллекция: {CHROMA_COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "IPS PLM документация"}
            )
            print(f"Создана новая коллекция: {CHROMA_COLLECTION_NAME}")
    
    def add_documents(self, documents: List[Dict]):
        """Добавляет документы в векторную базу"""
        if not documents:
            print("Нет документов для добавления")
            return
        
        # Подготавливаем данные для ChromaDB
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            content = doc['content']
            metadata = doc['metadata']
            
            texts.append(content)
            metadatas.append(metadata)
            ids.append(f"{metadata['filename']}_{metadata['chunk_id']}")
        
        print(f"Генерируем эмбеддинги для {len(texts)} документов...")
        
        # Генерируем эмбеддинги
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        embeddings = embeddings.tolist()
        
        # Добавляем в базу
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Добавлено {len(documents)} документов в векторную базу")
    
    def search(self, query: str, n_results: int = None) -> List[Dict]:
        """Поиск похожих документов"""
        if n_results is None:
            n_results = MAX_RETRIEVED_DOCS
        
        # Генерируем эмбеддинг для запроса
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding.tolist()
        
        # Поиск в базе
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Форматируем результаты
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            formatted_results.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': 1 - distance,  # Конвертируем расстояние в похожесть
                'rank': i + 1
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """Получает информацию о коллекции"""
        count = self.collection.count()
        return {
            'name': CHROMA_COLLECTION_NAME,
            'documents_count': count,
            'embedding_model': EMBEDDING_MODEL
        }
    
    def clear_collection(self):
        """Очищает коллекцию"""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "IPS PLM документация"}
        )
        print("Коллекция очищена")
    
    def get_document_sources(self) -> List[str]:
        """Получает список всех источников документов"""
        # Получаем все метаданные
        results = self.collection.get(include=['metadatas'])
        sources = set()
        
        for metadata in results['metadatas']:
            sources.add(metadata['filename'])
        
        return sorted(list(sources))

if __name__ == "__main__":
    # Тестирование векторной базы
    from document_processor import DocumentProcessor
    
    # Создаем базу и процессор
    db = VectorDatabase()
    processor = DocumentProcessor()
    
    print("Информация о базе данных:")
    info = db.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Если база пустая, обрабатываем документы
    if info['documents_count'] == 0:
        print("\nОбрабатываем документы...")
        documents = processor.process_directory()
        if documents:
            db.add_documents(documents)
        else:
            print("Нет документов для обработки")
    
    # Тестовый поиск
    if info['documents_count'] > 0 or len(processor.process_directory()) > 0:
        print("\nТестовый поиск:")
        test_queries = [
            "модули расширения IPS",
            "PDM интерфейс",
            "веб-портал API"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: '{query}'")
            results = db.search(query, n_results=3)
            for result in results:
                print(f"  Файл: {result['metadata']['filename']}")
                print(f"  Похожесть: {result['similarity_score']:.3f}")
                print(f"  Фрагмент: {result['content'][:100]}...")
                print()