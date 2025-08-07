"""
FastAPI веб-сервис для RAG системы IPS PLM
Предоставляет HTTP API для интеграции с Claude Code и другими клиентами
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from pathlib import Path

from rag_service import RAGService
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from config import API_HOST, API_PORT

# Модели данных для API
class QueryRequest(BaseModel):
    question: str
    max_docs: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_docs: int
    query: str
    context_length: int

class StatusResponse(BaseModel):
    vector_database: Dict
    lmstudio: Dict
    document_sources: List[str]

class ProcessDocumentsRequest(BaseModel):
    directory: Optional[str] = None
    clear_existing: Optional[bool] = False

# Создаем FastAPI приложение
app = FastAPI(
    title="RAG IPS PLM API",
    description="REST API для системы поиска в документации IPS PLM",
    version="1.0.0"
)

# Настройка CORS для интеграции с различными клиентами
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервисов
rag_service = RAGService()
document_processor = DocumentProcessor()
vector_database = VectorDatabase()

@app.get("/", summary="Корневая страница")
async def root():
    """Информация об API"""
    return {
        "service": "RAG IPS PLM API",
        "version": "1.0.0",
        "description": "REST API для поиска информации в документации IPS PLM",
        "endpoints": {
            "query": "/query - Поиск и генерация ответа",
            "status": "/status - Статус системы",
            "process": "/process-documents - Обработка документов",
            "docs": "/docs - Документация API"
        }
    }

@app.get("/status", response_model=StatusResponse, summary="Статус системы")
async def get_status():
    """Получить статус RAG системы"""
    try:
        status = rag_service.get_status()
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")

@app.get("/models", response_model=Dict, summary="Получить список доступных моделей")
async def get_available_models():
    """Получить список всех доступных моделей в LMStudio"""
    try:
        available_models = rag_service.get_available_models()
        return {"models": available_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка моделей: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Поиск и генерация ответа")
async def query_rag(request: QueryRequest):
    """
    Основной эндпоинт для запросов к RAG системе
    
    - **question**: Вопрос пользователя
    - **max_docs**: Максимальное количество документов для поиска (по умолчанию 5)
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
        
        result = rag_service.query(request.question, n_docs=request.max_docs)
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

@app.post("/process-documents", summary="Обработка документов")
async def process_documents(request: ProcessDocumentsRequest, background_tasks: BackgroundTasks):
    """
    Обработка документов и добавление их в векторную базу
    
    - **directory**: Путь к директории с документами (опционально)  
    - **clear_existing**: Очистить существующие документы перед добавлением новых
    """
    try:
        def process_docs_task():
            # Очищаем базу если запрошено
            if request.clear_existing:
                vector_database.clear_collection()
                print("Векторная база очищена")
            
            # Обрабатываем документы
            documents = document_processor.process_directory(request.directory)
            
            if documents:
                vector_database.add_documents(documents)
                print(f"Обработано и добавлено {len(documents)} документов")
            else:
                print("Нет документов для обработки")
        
        # Запускаем обработку в фоне
        background_tasks.add_task(process_docs_task)
        
        return {
            "message": "Обработка документов запущена в фоновом режиме",
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка запуска обработки документов: {str(e)}")

@app.get("/documents", summary="Список источников документов")
async def get_document_sources():
    """Получить список всех источников документов в базе"""
    try:
        sources = vector_database.get_document_sources()
        return {
            "sources": sources,
            "count": len(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка документов: {str(e)}")

@app.get("/search/{query}", summary="Только поиск документов")
async def search_documents(query: str, max_docs: int = 5):
    """
    Поиск документов без генерации ответа (только векторный поиск)
    
    - **query**: Поисковый запрос
    - **max_docs**: Максимальное количество результатов
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
        
        results = vector_database.search(query, n_results=max_docs)
        
        # Форматируем результаты для API
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result['content'][:300] + "..." if len(result['content']) > 300 else result['content'],
                "source": result['metadata']['filename'],
                "chunk_id": result['metadata']['chunk_id'],
                "similarity_score": result['similarity_score']
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")

@app.get("/health", summary="Проверка работоспособности")
async def health_check():
    """Простая проверка работоспособности сервиса"""
    return {"status": "healthy", "service": "RAG IPS PLM API"}

if __name__ == "__main__":
    print(f"Запуск RAG API сервера на {API_HOST}:{API_PORT}")
    print("Доступные эндпоинты:")
    print(f"  - Документация: http://{API_HOST}:{API_PORT}/docs")
    print(f"  - Статус: http://{API_HOST}:{API_PORT}/status")
    print(f"  - Запросы: http://{API_HOST}:{API_PORT}/query")
    
    uvicorn.run(
        app, 
        host=API_HOST, 
        port=API_PORT,
        reload=True,
        log_level="info"
    )

@app.get("/endpoints")
async def get_endpoints():
    """Возвращает список доступных эндпоинтов для Cline"""
    return {
        "service": "RAG_IPS",
        "version": "1.0",
        "description": "RAG система для документации IPS PLM",
        "endpoints": [
            {
                "path": "/query",
                "method": "POST",
                "description": "Основной эндпоинт для запросов к RAG системе",
                "parameters": {
                    "question": {
                        "type": "string",
                        "required": True,
                        "description": "Вопрос для поиска в документации"
                    },
                    "max_docs": {
                        "type": "integer", 
                        "required": False,
                        "default": 5,
                        "description": "Максимальное количество документов для поиска"
                    }
                },
                "example": {
                    "question": "Что такое модули расширения IPS PLM?",
                    "max_docs": 5
                }
            },
            {
                "path": "/status",
                "method": "GET", 
                "description": "Получение статуса RAG системы",
                "parameters": {},
                "example": {}
            },
            {
                "path": "/search",
                "method": "POST",
                "description": "Поиск документов без генерации ответа",
                "parameters": {
                    "query": {
                        "type": "string",
                        "required": True,
                        "description": "Поисковый запрос"
                    },
                    "max_docs": {
                        "type": "integer",
                        "required": False, 
                        "default": 5,
                        "description": "Максимальное количество документов"
                    }
                },
                "example": {
                    "query": "модули расширения",
                    "max_docs": 3
                }
            },
            {
                "path": "/process-documents",
                "method": "POST",
                "description": "Обработка новых документов",
                "parameters": {
                    "directory": {
                        "type": "string",
                        "required": True,
                        "description": "Путь к директории с документами"
                    },
                    "clear_existing": {
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Очистить существующие документы перед добавлением"
                    }
                },
                "example": {
                    "directory": "./docs",
                    "clear_existing": False
                }
            }
        ]
    }

@app.get("/favicon.ico")
async def get_favicon():
    return {"message": "No favicon available"}