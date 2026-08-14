import numpy as np
import json
import os

class MockDatabase:
    def __init__(self, db_path=None):
        """
        Инициализация mock-базы данных сотрудников.
        Если db_path указан, загружает сотрудников из JSON-файла.
        Иначе создает 10 тестовых сотрудников со случайными эмбеддингами.
        """
        self.employees = {}
        self.db_path = db_path
        
        # Проверяем, есть ли сохраненная база
        if db_path and os.path.exists(db_path):
            self._load_from_json(db_path)
        elif os.path.exists("employees_db.json"):
            # Если файл есть в текущей папке, загружаем его
            self._load_from_json("employees_db.json")
            print(f"[MockDB] Загружена база из employees_db.json")
        else:
            # Иначе генерируем тестовых сотрудников
            self._generate_test_employees()
    
    def _generate_test_employees(self):
        """Генерация 10 тестовых сотрудников со случайными эмбеддингами."""
        for i in range(1, 11):
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.employees[i] = {
                "embedding": embedding,
                "name": f"Test Employee {i}",
                "department": f"Department {i % 3 + 1}",
                "active": True
            }
        
        # Специальный сотрудник с известным эмбеддингом для тестирования
        known_embedding = np.random.randn(128).astype(np.float32)
        known_embedding = known_embedding / np.linalg.norm(known_embedding)
        self.employees[999] = {
            "embedding": known_embedding,
            "name": "Known Test Employee",
            "department": "Test Department",
            "active": True
        }
        
        print(f"[MockDB] Сгенерировано {len(self.employees)} тестовых сотрудников")
    
    def _load_from_json(self, db_path):
        """Загрузка сотрудников из JSON-файла."""
        with open(db_path, 'r') as f:
            data = json.load(f)
        
        for emp_id, emp_data in data.items():
            emp_id = int(emp_id)
            embedding = np.array(emp_data["embedding"], dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.employees[emp_id] = {
                "embedding": embedding,
                "name": emp_data.get("name", f"Employee {emp_id}"),
                "department": emp_data.get("department", "Unknown"),
                "active": emp_data.get("active", True)
            }
        
        print(f"[MockDB] Загружено {len(self.employees)} сотрудников из {db_path}")
    
    def verify(self, embedding, threshold=0.75):
        """
        Поиск сотрудника по эмбеддингу.
        Возвращает решение: allow / manual_review с информацией о найденном сотруднике.
        """
        if embedding is None:
            return {
                "decision": "manual_review",
                "employee_id": None,
                "score": 0.0,
                "reason": "Эмбеддинг отсутствует"
            }
        
        # Нормализуем входной эмбеддинг
        embedding = embedding / np.linalg.norm(embedding)
        
        best_match = None
        best_score = -1
        second_score = -1
        
        for emp_id, emp_data in self.employees.items():
            if not emp_data["active"]:
                continue
            
            db_embedding = emp_data["embedding"]
            # Косинусное сходство (векторы уже нормализованы)
            score = np.dot(embedding, db_embedding)
            
            if score > best_score:
                second_score = best_score
                best_score = score
                best_match = emp_id
            elif score > second_score:
                second_score = score
        
        margin = best_score - second_score if second_score > -1 else 1.0
        
        # Принятие решения
        if best_match is None:
            return {
                "decision": "deny",
                "employee_id": None,
                "score": 0.0,
                "reason": "Сотрудник не найден в базе"
            }
        
        if best_score >= threshold and margin >= 0.1:
            return {
                "decision": "allow",
                "employee_id": best_match,
                "score": float(best_score),
                "margin": float(margin),
                "second_best_score": float(second_score) if second_score > -1 else None,
                "employee_name": self.employees[best_match]["name"],
                "reason": f"match_score={best_score:.3f}, margin={margin:.3f}"
            }
        elif best_score >= 0.55:
            return {
                "decision": "manual_review",
                "employee_id": best_match,
                "score": float(best_score),
                "margin": float(margin),
                "second_best_score": float(second_score) if second_score > -1 else None,
                "employee_name": self.employees[best_match]["name"],
                "reason": f"Низкий match_score ({best_score:.3f}) или малый margin ({margin:.3f})"
            }
        else:
            return {
                "decision": "deny",
                "employee_id": None,
                "score": float(best_score),
                "reason": f"match_score={best_score:.3f} ниже порога deny (0.55)"
            }
    
    def add_employee(self, emp_id, embedding, name="Unknown", department="Unknown"):
        """Добавление нового сотрудника в базу."""
        embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        self.employees[emp_id] = {
            "embedding": embedding,
            "name": name,
            "department": department,
            "active": True
        }
        print(f"[MockDB] Добавлен сотрудник {emp_id}: {name}")
    
    def remove_employee(self, emp_id):
        """Удаление сотрудника из базы (деактивация)."""
        if emp_id in self.employees:
            self.employees[emp_id]["active"] = False
            print(f"[MockDB] Сотрудник {emp_id} деактивирован")
            return True
        return False
    
    def get_employee_count(self):
        """Возвращает количество активных сотрудников в базе."""
        return sum(1 for emp in self.employees.values() if emp["active"])
    
    def save_to_json(self, db_path):
        """Сохранение базы сотрудников в JSON-файл."""
        data = {}
        for emp_id, emp_data in self.employees.items():
            data[str(emp_id)] = {
                "embedding": emp_data["embedding"].tolist(),
                "name": emp_data["name"],
                "department": emp_data["department"],
                "active": emp_data["active"]
            }
        
        with open(db_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[MockDB] База сохранена в {db_path}")