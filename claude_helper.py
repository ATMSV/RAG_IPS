#!/usr/bin/env python3
"""
Вспомогательный скрипт для интеграции RAG с Claude Code
"""
import httpx
import sys
import json

def query_rag(question: str) -> dict:
    """Отправляет запрос к RAG системе"""
    try:
        response = httpx.post(
            "http://localhost:8000/query",
            json={"question": question, "max_docs": 5},
            timeout=180.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": f"Ошибка подключения: {str(e)}"}

def main():
    if len(sys.argv) < 2:
        print("Использование: python claude_helper.py 'ваш вопрос'")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    print(f"Отправка запроса: {question}")
    
    result = query_rag(question)
    
    if "error" in result:
        print(f"Ошибка: {result['error']}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("ОТВЕТ RAG СИСТЕМЫ:")
    print("="*50)
    print(result["answer"])
    
    print("\n" + "="*50)
    print("ИСТОЧНИКИ:")
    print("="*50)
    for source in result["sources"]:
        print(f"- {source}")
    
    print(f"\nИспользовано документов: {result['retrieved_docs']}")

if __name__ == "__main__":
    main()