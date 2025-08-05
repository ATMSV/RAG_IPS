"""
Главный скрипт для запуска RAG системы IPS PLM
Предоставляет интерфейс командной строки для управления системой
"""
import sys
import argparse
from pathlib import Path

def setup_system():
    """Инициализация и настройка системы"""
    print("=== Настройка RAG системы IPS PLM ===")
    
    from document_processor import DocumentProcessor
    from vector_database import VectorDatabase
    
    # Создаем процессор и базу данных
    processor = DocumentProcessor()
    db = VectorDatabase()
    
    print("Проверка наличия документов...")
    
    # Проверяем, есть ли PDF файлы в корневой директории
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if pdf_files:
        print(f"Найдено PDF файлов: {len(pdf_files)}")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        # Обрабатываем документы
        print("\nОбработка документов...")
        documents = processor.process_directory(".")
        
        if documents:
            print(f"Добавление {len(documents)} чанков в векторную базу...")
            db.add_documents(documents)
            print("✓ Документы успешно обработаны и добавлены в базу")
        else:
            print("⚠ Не удалось обработать документы")
    else:
        print("⚠ PDF файлы не найдены в текущей директории")
    
    # Показываем статус
    info = db.get_collection_info()
    print(f"\nСтатус векторной базы:")
    print(f"  Документов в базе: {info['documents_count']}")
    print(f"  Модель эмбеддингов: {info['embedding_model']}")
    
    return info['documents_count'] > 0

def test_rag():
    """Тестирование RAG системы"""
    print("=== Тестирование RAG системы ===")
    
    from rag_service import RAGService
    
    rag = RAGService()
    
    # Проверяем статус
    status = rag.get_status()
    
    print("\nСтатус компонентов:")
    print(f"  Векторная база: {status['vector_database']['status']} ({status['vector_database']['documents_count']} документов)")
    print(f"  LMStudio: {status['lmstudio']['status']}")
    if status['lmstudio']['model']:
        print(f"  Модель LLM: {status['lmstudio']['model']}")
    
    if status['vector_database']['documents_count'] == 0:
        print("\n⚠ Нет документов в базе. Запустите сначала setup.")
        return False
    
    # Тестовые запросы
    if status['lmstudio']['status'] == 'online':
        print("\n=== Тестовые запросы ===")
        test_queries = [
            "Что такое модули расширения IPS PLM?"
        ]
        
        for query in test_queries:
            print(f"\nВопрос: {query}")
            result = rag.query(query)
            print(f"Источники: {', '.join(result['sources'])}")
            print(f"Ответ: {result['answer'][:300]}...")
    else:
        print("\n⚠ LMStudio не запущен. Для полного тестирования запустите LMStudio на localhost:1234")
    
    return True

def start_api():
    """Запуск API сервера"""
    print("=== Запуск API сервера ===")
    
    from api import app
    from config import API_HOST, API_PORT
    import uvicorn
    
    print(f"Запуск сервера на http://{API_HOST}:{API_PORT}")
    print("Доступные эндпоинты:")
    print(f"  - Документация: http://{API_HOST}:{API_PORT}/docs")
    print(f"  - Статус: http://{API_HOST}:{API_PORT}/status")
    print(f"  - Запросы: http://{API_HOST}:{API_PORT}/query")
    print("\nДля остановки нажмите Ctrl+C")
    
    try:
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
    except KeyboardInterrupt:
        print("\n✓ Сервер остановлен")

def interactive_mode():
    """Интерактивный режим для тестирования"""
    print("=== Интерактивный режим RAG системы ===")
    
    from rag_service import RAGService
    
    rag = RAGService()
    
    # Проверяем готовность системы
    status = rag.get_status()
    
    if status['vector_database']['documents_count'] == 0:
        print("⚠ Нет документов в базе. Запустите сначала: python main.py setup")
        return
    
    if status['lmstudio']['status'] != 'online':
        print("⚠ LMStudio не запущен. Запустите LMStudio на localhost:1234")
        print("Без LMStudio доступен только поиск документов.\n")
    
    print(f"Загружено документов: {status['vector_database']['documents_count']}")
    print("Введите ваши вопросы (для выхода введите 'quit'):\n")
    
    while True:
        try:
            question = input("Вопрос: ").strip()
            
            if question.lower() in ['quit', 'exit', 'выход']:
                print("До свидания!")
                break
            
            if not question:
                continue
            
            print("Поиск...")
            result = rag.query(question)
            
            print(f"\nИсточники: {', '.join(result['sources'])}")
            print(f"Ответ:\n{result['answer']}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="RAG система для документации IPS PLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py setup          # Настройка и обработка документов
  python main.py test           # Тестирование системы  
  python main.py api            # Запуск API сервера
  python main.py interactive    # Интерактивный режим
        """
    )
    
    parser.add_argument(
        'command',
        choices=['setup', 'test', 'api', 'interactive'],
        help='Команда для выполнения'
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    try:
        if args.command == 'setup':
            setup_system()
        elif args.command == 'test':
            test_rag()
        elif args.command == 'api':
            start_api()
        elif args.command == 'interactive':
            interactive_mode()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())