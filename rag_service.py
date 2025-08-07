"""
Основной RAG сервис для IPS PLM
Интегрирует поиск в векторной базе с генерацией ответов через LMStudio
"""
import httpx
import json
from typing import List, Dict, Optional
from vector_database import VectorDatabase
from config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_API_KEY,
    LMSTUDIO_MODEL,
    MAX_CONTEXT_LENGTH,
    MAX_RETRIEVED_DOCS
)

class RAGService:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.lmstudio_client = httpx.Client(
            base_url=LMSTUDIO_BASE_URL,
            timeout=180.0  # Увеличили timeout до 3 минут для медленных моделей
        )
        self.available_models = []  # Инициализируем список доступных моделей
        self.current_model = None  # Инициализируем текущую модель
    
    def _refresh_available_models(self):
        """Получает список доступных моделей из LMStudio"""
        try:
            response = self.lmstudio_client.get("/models")
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['id'] for model in models_data['data']]
                if not self.current_model and self.available_models:
                    self.current_model = self.available_models[0]
                return self.available_models
            else:
                print(f"Ошибка при получении списка моделей: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Ошибка при подключении к LMStudio: {str(e)}")
            return []
    
    def _format_context(self, documents: List[Dict]) -> str:
        """Форматирует найденные документы в контекст для LLM"""
        context_parts = []
        
        for doc in documents:
            source = doc['metadata']['filename']
            content = doc['content']
            similarity = doc['similarity_score']
            
            context_parts.append(
                f"[Источник: {source}, релевантность: {similarity:.2f}]\n{content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Обрезаем контекст если он слишком длинный
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH] + "...\n[Контекст обрезан]"
        
        return context
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Создает промпт для LLM"""
        prompt = f"""Ты эксперт по системе IPS PLM и модулям расширения. 

На основе предоставленной документации ответь на вопрос пользователя максимально точно и подробно. Используй только информацию из предоставленных источников.

ДОКУМЕНТАЦИЯ:
{context}

ВОПРОС: {query}

ОТВЕТ: Отвечай на русском языке, структурируй ответ, указывай источники информации когда это уместно."""

        return prompt
    
    def _call_lmstudio(self, prompt: str, model: str = None) -> str:
        """Отправляет запрос к LMStudio API"""
        try:
            # Проверяем доступность LMStudio
            health_response = self.lmstudio_client.get("/health")
            if health_response.status_code != 200:
                return "Ошибка: LMStudio недоступен. Убедитесь, что сервер запущен на localhost:1234"
            
            # Получаем список моделей если модель не указана
            if not model:
                model = LMSTUDIO_MODEL  # Используем модель из конфига
                print(f"Используем модель: {model}")
            
            # Отправляем запрос на генерацию
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,  # Баланс между точностью и креативностью
                "max_tokens": 1200,  # Достаточно для подробного ответа
                "stream": False
            }
            
            response = self.lmstudio_client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {LMSTUDIO_API_KEY}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Ошибка LMStudio API: {response.status_code} - {response.text}"
                
        except httpx.ConnectError:
            return "Ошибка подключения к LMStudio. Убедитесь, что LMStudio запущен на localhost:1234"
        except Exception as e:
            return f"Ошибка при обращении к LMStudio: {str(e)}"
    
    def query(self, question: str, n_docs: int = None) -> Dict:
        """Основной метод для запросов к RAG системе"""
        if n_docs is None:
            n_docs = MAX_RETRIEVED_DOCS
        
        # Поиск релевантных документов
        print(f"Поиск документов по запросу: '{question}'")
        relevant_docs = self.vector_db.search(question, n_results=n_docs)
        
        if not relevant_docs:
            return {
                "answer": "Извините, я не нашел релевантной информации в документации по вашему запросу.",
                "sources": [],
                "query": question
            }
        
        # Формируем контекст и промпт
        context = self._format_context(relevant_docs)
        prompt = self._build_prompt(question, context)
        
        print(f"Найдено документов: {len(relevant_docs)}")
        print("Отправка запроса к LMStudio...")
        
        # Получаем ответ от LLM
        answer = self._call_lmstudio(prompt)
        
        # Извлекаем источники
        sources = list(set([doc['metadata']['filename'] for doc in relevant_docs]))
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": len(relevant_docs),
            "query": question,
            "context_length": len(context)
        }
    
    def get_status(self) -> Dict:
        """Получает статус RAG системы"""
        # Проверяем векторную базу
        db_info = self.vector_db.get_collection_info()
        
        # Проверяем LMStudio
        lmstudio_status = "offline"
        lmstudio_model = None
        
        try:
            health_response = self.lmstudio_client.get("/health")
            if health_response.status_code == 200:
                lmstudio_status = "online"
                self._refresh_available_models()
                lmstudio_model = self.current_model if self.current_model else None
        except:
            pass
        
        return {
            "vector_database": {
                "status": "online" if db_info['documents_count'] > 0 else "empty",
                "documents_count": db_info['documents_count'],
                "embedding_model": db_info['embedding_model']
            },
            "lmstudio": {
                "status": lmstudio_status,
                "base_url": LMSTUDIO_BASE_URL,
                "model": lmstudio_model,
                "available_models": self.available_models
            },
            "document_sources": self.vector_db.get_document_sources()
        }

if __name__ == "__main__":
    # Тестирование RAG сервиса
    rag = RAGService()
    
    print("=== Статус RAG системы ===")
    status = rag.get_status()
    
    print("\nВекторная база:")
    db_status = status['vector_database']
    print(f"  Статус: {db_status['status']}")
    print(f"  Документов: {db_status['documents_count']}")
    print(f"  Модель эмбеддингов: {db_status['embedding_model']}")
    
    print("\nLMStudio:")
    lms_status = status['lmstudio']
    print(f"  Статус: {lms_status['status']}")
    print(f"  URL: {lms_status['base_url']}")
    print(f"  Модель: {lms_status['model'] or 'не определена'}")
    
    if status['document_sources']:
        print(f"\nИсточники документов:")
        for source in status['document_sources']:
            print(f"  - {source}")
    
    # Тестовые запросы (только если есть документы)
    if db_status['documents_count'] > 0:
        print("\n=== Тестовые запросы ===")
        test_queries = [
            "Что такое модули расширения IPS PLM?",
            "Как работает PDM интерфейс?",
            "Какие API доступны для веб-портала?"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            result = rag.query(query)
            print(f"Найдено документов: {result['retrieved_docs']}")
            print(f"Источники: {', '.join(result['sources'])}")
            print(f"Ответ: {result['answer'][:200]}...")
    else:
        print("\nНет документов для тестирования. Сначала обработайте документы.")
