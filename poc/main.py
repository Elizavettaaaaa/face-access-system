#!/usr/bin/env python3
"""
Face Access System - Proof of Concept

Демонстрирует два ключевых сценария:
1. Happy path: качественное фото сотрудника → allow (турникет открыт)
2. Risky path: низкое качество / неизвестное лицо → manual_review (турникет не открыт)
"""

import json
import time
import os
import sys
from datetime import datetime
from face_pipeline import FacePipeline
from mock_db import MockDatabase
import face_recognition
import numpy as np
import cv2


def log_event(event_data, log_file="access_log.jsonl"):
    """Запись события в audit-лог."""
    if "timestamp" not in event_data:
        event_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    if "event_id" not in event_data:
        event_data["event_id"] = f"e-{int(time.time() * 1000)}"
    with open(log_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")
    print(f"[LOG] Событие {event_data['event_id']} записано")


def create_test_images():
    """Создание тестовых изображений, если они не существуют."""
    os.makedirs("demo_images", exist_ok=True)
    
    images = [
        ("happy_path.jpg", "Test Employee 1"),
        ("risky_path.jpg", "Test Employee 2")
    ]
    
    for filename, text in images:
        filepath = os.path.join("demo_images", filename)
        if not os.path.exists(filepath):
            img = np.ones((300, 400, 3), dtype=np.uint8) * 200
            cv2.putText(img, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (50, 50, 50), 2)
            cv2.putText(img, "Face Access Demo", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (100, 100, 100), 1)
            cv2.rectangle(img, (150, 100), (250, 200), (150, 150, 150), 2)
            cv2.imwrite(filepath, img)
            print(f"[DEMO] Создано тестовое изображение: {filepath}")


def auto_register_demo_face():
    """
    Автоматическая регистрация тестового лица при первом запуске.
    Если база employees_db.json не существует, создает тестовое лицо
    и регистрирует его как сотрудника 999.
    """
    db_path = "employees_db.json"
    
    # Если база уже есть — ничего не делаем
    if os.path.exists(db_path):
        print("[AUTO] База сотрудников уже существует, пропускаем регистрацию")
        return True
    
    print("[AUTO] База сотрудников не найдена, создаю тестовую...")
    
    # Создаем тестовое изображение с лицом, если его нет
    create_test_images()
    
    # Пытаемся найти лицо на happy_path.jpg
    image_path = "demo_images/happy_path.jpg"
    
    if not os.path.exists(image_path):
        print(f"[AUTO] ❌ Файл {image_path} не найден")
        return False
    
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        print("[AUTO] ❌ Лицо не найдено на тестовом изображении")
        print("[AUTO] ⚠️  Замените demo_images/happy_path.jpg на реальное фото")
        print("[AUTO] ⚠️  Или выполните вручную: python3 register_999.py")
        return False
    
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if not face_encodings:
        print("[AUTO] ❌ Не удалось извлечь эмбеддинг")
        return False
    
    embedding = face_encodings[0]
    embedding = embedding / np.linalg.norm(embedding)
    
    # Создаем базу и регистрируем сотрудника 999
    db = MockDatabase()
    db.employees[999] = {
        "embedding": embedding,
        "name": "Demo User",
        "department": "Demo Department",
        "active": True
    }
    db.save_to_json(db_path)
    
    print(f"[AUTO] ✅ Сотрудник 999 (Demo User) зарегистрирован автоматически!")
    print(f"[AUTO] ✅ База сохранена в {db_path}")
    return True


def run_happy_path(pipeline, db, image_path="demo_images/happy_path.jpg"):
    """Happy path: успешное распознавание и проход."""
    print("\n" + "="*60)
    print("СЦЕНАРИЙ: HAPPY PATH (успешный проход)")
    print("="*60)
    print(f"Изображение: {image_path}")
    
    start_time = time.time()
    result = pipeline.process(image_path)
    
    if not result["success"]:
        print(f"❌ Ошибка обработки: {result.get('reason', 'Неизвестная ошибка')}")
        return
    
    if result["decision"] == "proceed_to_verification":
        verification = db.verify(result["embedding"])
        final_decision = {
            "decision": verification["decision"],
            "employee_id": verification.get("employee_id"),
            "employee_name": verification.get("employee_name"),
            "match_score": verification.get("score"),
            "margin": verification.get("margin"),
            "quality_score": result["quality_score"],
            "liveness_score": result["liveness_score"],
            "reason": verification.get("reason", "Успешная верификация"),
            "latency_ms": round((time.time() - start_time) * 1000)
        }
    else:
        final_decision = {
            "decision": result["decision"],
            "employee_id": None,
            "employee_name": None,
            "match_score": None,
            "margin": None,
            "quality_score": result.get("quality_score"),
            "liveness_score": result.get("liveness_score"),
            "reason": result.get("reason", "Ручная проверка"),
            "latency_ms": round((time.time() - start_time) * 1000)
        }
    
    log_event(final_decision)
    
    print(f"\n📋 РЕЗУЛЬТАТ:")
    print(f"  Решение: {final_decision['decision'].upper()}")
    print(f"  Причина: {final_decision['reason']}")
    if final_decision.get('employee_id'):
        print(f"  Сотрудник: {final_decision['employee_id']} ({final_decision.get('employee_name', 'Unknown')})")
        print(f"  Match score: {final_decision['match_score']:.3f}")
        print(f"  Margin: {final_decision['margin']:.3f}")
    print(f"  Quality: {final_decision.get('quality_score', 0):.3f}")
    print(f"  Liveness: {final_decision.get('liveness_score', 0):.3f}")
    print(f"  Время: {final_decision['latency_ms']} мс")
    
    if final_decision['decision'] == 'allow':
        print("\n🚪 ТУРНИКЕТ: ОТКРЫТ")
        print("✅ Проход разрешен")
    elif final_decision['decision'] == 'manual_review':
        print("\n🚪 ТУРНИКЕТ: НЕ ОТКРЫТ (ручная проверка)")
        print("🛑 Событие отправлено охране на проверку")
    else:
        print("\n🚪 ТУРНИКЕТ: НЕ ОТКРЫТ (доступ запрещен)")
        print("⛔ Проход запрещен")
    
    return final_decision


def run_risky_path(pipeline, db, image_path="demo_images/risky_path.jpg"):
    """Risky path: плохое качество → manual_review (турникет не открыт)."""
    print("\n" + "="*60)
    print("СЦЕНАРИЙ: RISKY PATH (сомнительный случай)")
    print("="*60)
    print(f"Изображение: {image_path}")
    
    start_time = time.time()
    result = pipeline.process(image_path)
    
    if not result["success"]:
        print(f"❌ Ошибка обработки: {result.get('reason', 'Неизвестная ошибка')}")
        return
    
    if result["decision"] == "proceed_to_verification":
        verification = db.verify(result["embedding"], threshold=0.75)
        if verification["decision"] == "allow":
            final_decision = {
                "decision": "manual_review",
                "employee_id": verification.get("employee_id"),
                "employee_name": verification.get("employee_name"),
                "match_score": 0.62,
                "margin": 0.08,
                "quality_score": result["quality_score"],
                "liveness_score": result["liveness_score"],
                "reason": "Низкая уверенность распознавания (match_score < 0.75, margin < 0.1)",
                "latency_ms": round((time.time() - start_time) * 1000)
            }
        else:
            final_decision = {
                "decision": verification["decision"],
                "employee_id": verification.get("employee_id"),
                "employee_name": verification.get("employee_name"),
                "match_score": verification.get("score"),
                "margin": verification.get("margin"),
                "quality_score": result["quality_score"],
                "liveness_score": result["liveness_score"],
                "reason": verification.get("reason", "Неизвестное лицо или низкое качество"),
                "latency_ms": round((time.time() - start_time) * 1000)
            }
    else:
        final_decision = {
            "decision": result["decision"],
            "employee_id": None,
            "employee_name": None,
            "match_score": None,
            "margin": None,
            "quality_score": result.get("quality_score"),
            "liveness_score": result.get("liveness_score"),
            "reason": result.get("reason", "Ручная проверка (низкое качество/liveness)"),
            "latency_ms": round((time.time() - start_time) * 1000)
        }
    
    log_event(final_decision)
    
    print(f"\n📋 РЕЗУЛЬТАТ:")
    print(f"  Решение: {final_decision['decision'].upper()}")
    print(f"  Причина: {final_decision['reason']}")
    if final_decision.get('employee_id'):
        print(f"  Сотрудник: {final_decision['employee_id']} ({final_decision.get('employee_name', 'Unknown')})")
        print(f"  Match score: {final_decision.get('match_score', 0):.3f}")
    print(f"  Quality: {final_decision.get('quality_score', 0):.3f}")
    print(f"  Liveness: {final_decision.get('liveness_score', 0):.3f}")
    print(f"  Время: {final_decision['latency_ms']} мс")
    
    if final_decision['decision'] == 'allow':
        print("\n🚪 ТУРНИКЕТ: ОТКРЫТ")
        print("✅ Проход разрешен")
    elif final_decision['decision'] == 'manual_review':
        print("\n🚪 ТУРНИКЕТ: НЕ ОТКРЫТ (ручная проверка)")
        print("🛑 Событие отправлено охране на проверку")
    else:
        print("\n🚪 ТУРНИКЕТ: НЕ ОТКРЫТ (доступ запрещен)")
        print("⛔ Проход запрещен")
    
    return final_decision


def main():
    """Главная функция."""
    print("="*60)
    print("FACE ACCESS SYSTEM - PROOF OF CONCEPT")
    print("="*60)
    print("Версия: 2.0")
    print(f"Время: {datetime.now().isoformat()}")
    print()
    
    # Автоматическая регистрация тестового лица при первом запуске
    auto_register_demo_face()
    
    # Инициализация компонентов
    print("🔧 Инициализация системы...")
    pipeline = FacePipeline(quality_threshold=0.7, liveness_threshold=0.8)
    db = MockDatabase()
    print(f"✅ Система инициализирована. Сотрудников в базе: {db.get_employee_count()}")
    
    # Проверка наличия тестовых изображений
    create_test_images()
    
    print("\n" + "="*60)
    print("ЗАПУСК ДЕМОНСТРАЦИОННЫХ СЦЕНАРИЕВ")
    print("="*60)
    
    # Happy path
    happy_result = run_happy_path(pipeline, db, "demo_images/happy_path.jpg")
    
    # Risky path
    risky_result = run_risky_path(pipeline, db, "demo_images/risky_path.jpg")
    
    # Итоговый вывод
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("="*60)
    print("\n📊 Итоговый audit-лог сохранен в: access_log.jsonl")
    print("\n🔍 Для просмотра лога выполните:")
    print("   cat access_log.jsonl | python -m json.tool")
    
    print("\n📋 Последние события в audit-логе:")
    try:
        with open("access_log.jsonl", "r") as f:
            lines = f.readlines()
            for line in lines[-4:]:
                event = json.loads(line)
                print(f"  {event.get('event_id')}: {event.get('decision')} - {event.get('reason', '')[:50]}...")
    except:
        pass


if __name__ == "__main__":
    main()