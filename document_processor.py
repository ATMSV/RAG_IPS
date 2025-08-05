"""
Обработчик документов для RAG системы IPS PLM
Извлекает текст из PDF и других документов
"""
try:
    import fitz  # PyMuPDF
    PDF_LIBRARY = "pymupdf"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber" 
    except ImportError:
        import PyPDF2
        PDF_LIBRARY = "pypdf2"

from pathlib import Path
from typing import List, Dict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Извлекает текст из PDF файла"""
        try:
            if PDF_LIBRARY == "pymupdf":
                doc = fitz.open(pdf_path)
                text = ""
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page_text = page.get_text()
                    text += f"\n[Страница {page_num + 1}]\n{page_text}"
                doc.close()
                return self.clean_text(text)
                
            elif PDF_LIBRARY == "pdfplumber":
                import pdfplumber
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or ""
                        text += f"\n[Страница {page_num + 1}]\n{page_text}"
                return self.clean_text(text)
                
            elif PDF_LIBRARY == "pypdf2":
                import PyPDF2
                text = ""
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text += f"\n[Страница {page_num + 1}]\n{page_text}"
                return self.clean_text(text)
            
        except Exception as e:
            print(f"Ошибка обработки PDF {pdf_path} с библиотекой {PDF_LIBRARY}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Очищает текст от лишних символов"""
        # Убираем лишние пробелы и переносы
        text = re.sub(r'\s+', ' ', text)
        # Убираем только опасные контрольные символы, сохраняя кириллицу
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()
    
    def process_document(self, file_path: str) -> List[Dict]:
        """Обрабатывает документ и возвращает чанки с метаданными"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Извлекаем текст в зависимости от типа файла
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(str(file_path))
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        if not text.strip():
            print(f"Предупреждение: пустой документ {file_path}")
            return []
        
        # Разбиваем на чанки
        chunks = self.text_splitter.split_text(text)
        
        # Добавляем метаданные
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'content': chunk,
                'metadata': {
                    'source': str(file_path),
                    'filename': file_path.name,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return processed_chunks
    
    def process_directory(self, directory: str = None) -> List[Dict]:
        """Обрабатывает все документы в директории"""
        if directory is None:
            directory = DOCS_DIR
        
        directory = Path(directory)
        all_chunks = []
        
        # Ищем все поддерживаемые файлы
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                print(f"Обрабатываем: {file_path.name}")
                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                    print(f"  Создано чанков: {len(chunks)}")
                except Exception as e:
                    print(f"  Ошибка: {e}")
        
        print(f"Всего обработано чанков: {len(all_chunks)}")
        return all_chunks

if __name__ == "__main__":
    # Тестирование обработчика
    processor = DocumentProcessor()
    
    # Копируем PDF файлы в папку docs для обработки
    import shutil
    source_pdf1 = Path("IPS. Руководство программиста. Модули расширения.pdf")
    source_pdf2 = Path("IPS. Руководство программиста.pdf")
    
    if source_pdf1.exists():
        shutil.copy(source_pdf1, DOCS_DIR / source_pdf1.name)
        print(f"Скопирован: {source_pdf1.name}")
    
    if source_pdf2.exists():
        shutil.copy(source_pdf2, DOCS_DIR / source_pdf2.name)
        print(f"Скопирован: {source_pdf2.name}")
    
    # Обрабатываем документы
    chunks = processor.process_directory()
    
    if chunks:
        print(f"\nПример чанка:")
        print(f"Источник: {chunks[0]['metadata']['filename']}")
        print(f"Содержимое (первые 200 символов):")
        print(chunks[0]['content'][:200] + "...")