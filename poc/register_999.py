import face_recognition
import numpy as np
import os
from mock_db import MockDatabase

def register_face_as_employee_999(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Файл {image_path} не найден")
        return False
    
    print(f"📸 Обработка фото: {image_path}")
    
    # Загружаем изображение
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        print(f"❌ Лицо не найдено на {image_path}")
        print("   Попробуй другое фото (анфас, хорошее освещение)")
        return False
    
    print(f"✅ Найдено {len(face_locations)} лиц(о)")
    
    # Извлекаем эмбеддинг
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if not face_encodings:
        print(f"❌ Не удалось извлечь эмбеддинг")
        return False
    
    embedding = face_encodings[0]
    embedding = embedding / np.linalg.norm(embedding)
    
    # Обновляем сотрудника 999
    db = MockDatabase()
    db.employees[999] = {
        "embedding": embedding,
        "name": "Registered User",
        "department": "Test Department",
        "active": True
    }
    
    print(f"✅ Сотрудник 999 успешно зарегистрирован!")
    print(f"   Имя: Registered User")
    print(f"   Размер эмбеддинга: {len(embedding)}")
    
    # Сохраняем базу в файл
    db.save_to_json("employees_db.json")
    print(f"   База сохранена в employees_db.json")
    
    return True

if __name__ == "__main__":
    image_path = "demo_images/happy_path.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Файл {image_path} не найден!")
        print("   Сначала положи фото в папку demo_images/happy_path.jpg")
    else:
        register_face_as_employee_999(image_path)