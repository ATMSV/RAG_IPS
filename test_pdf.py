#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки обработки PDF
"""
from document_processor import DocumentProcessor
import traceback

def test_pdf_processing():
    print("=== Тест обработки PDF ===")
    proc = DocumentProcessor()
    
    pdf_files = [
        "IPS. Руководство программиста. Модули расширения.pdf",
        "IPS. Руководство программиста.pdf"
    ]
    
    for pdf_file in pdf_files:
        print(f"\nТестируем файл: {pdf_file}")
        try:
            # Сначала тестируем извлечение текста
            print("1. Извлечение текста...")
            text = proc.extract_text_from_pdf(pdf_file)
            print(f"   Длина текста: {len(text)} символов")
            
            if len(text) > 0:
                print(f"   Первые 100 символов: {repr(text[:100])}")
                
                # Теперь тестируем обработку документа
                print("2. Обработка документа...")
                result = proc.process_document(pdf_file)
                print(f"   Создано чанков: {len(result)}")
                
                if result:
                    print(f"   Первый чанк: {repr(result[0]['content'][:100])}")
                    print(f"   Метаданные: {result[0]['metadata']}")
                    print("   ✓ Успешно!")
                else:
                    print("   ⚠ Пустой результат")
            else:
                print("   ⚠ Пустой текст")
                
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_pdf_processing()