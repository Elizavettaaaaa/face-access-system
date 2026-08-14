import face_recognition
import numpy as np
import cv2
import os

class FacePipeline:
    def __init__(self, quality_threshold=0.7, liveness_threshold=0.8):
        """
        Инициализация пайплайна обработки лица.
        
        Args:
            quality_threshold: порог качества кадра (0-1)
            liveness_threshold: порог liveness (0-1)
        """
        self.quality_threshold = quality_threshold
        self.liveness_threshold = liveness_threshold
    
    def process(self, image_path):
        """
        Полный пайплайн обработки кадра.
        
        Args:
            image_path: путь к изображению
        
        Returns:
            dict: результат обработки с полями:
                - success: bool
                - decision: str (manual_review / proceed_to_verification)
                - embedding: np.array (если успешно)
                - quality_score: float
                - liveness_score: float
                - reason: str
        """
        # Проверка существования файла
        if not os.path.exists(image_path):
            return {
                "success": False,
                "decision": "manual_review",
                "reason": f"Файл не найден: {image_path}"
            }
        
        # Загрузка изображения
        try:
            image = face_recognition.load_image_file(image_path)
        except Exception as e:
            return {
                "success": False,
                "decision": "manual_review",
                "reason": f"Ошибка загрузки изображения: {str(e)}"
            }
        
        # Детекция лиц
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return {
                "success": False,
                "decision": "manual_review",
                "reason": "Лицо не обнаружено на кадре"
            }
        
        # Берем первое обнаруженное лицо (в production — выбор по размеру или уверенности)
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        
        # Оценка качества (эвристика по размеру и резкости)
        quality_score = self._estimate_quality(face_image)
        
        # Проверка качества
        if quality_score < self.quality_threshold:
            return {
                "success": True,
                "decision": "manual_review",
                "embedding": None,
                "quality_score": quality_score,
                "liveness_score": 0.0,
                "reason": f"Низкое качество кадра: {quality_score:.2f} < {self.quality_threshold}"
            }
        
        # Извлечение эмбеддинга
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            return {
                "success": False,
                "decision": "manual_review",
                "reason": "Не удалось извлечь эмбеддинг"
            }
        
        embedding = face_encodings[0]
        embedding = embedding / np.linalg.norm(embedding)  # Нормализация
        
        # Liveness проверка (в PoC — mock-реализация)
        liveness_score = self._check_liveness_mock(image, face_locations[0])
        
        if liveness_score < self.liveness_threshold:
            return {
                "success": True,
                "decision": "manual_review",
                "embedding": embedding,
                "quality_score": quality_score,
                "liveness_score": liveness_score,
                "reason": f"Liveness не подтвержден: {liveness_score:.2f} < {self.liveness_threshold}"
            }
        
        # Успешный проход — передаем на верификацию
        return {
            "success": True,
            "decision": "proceed_to_verification",
            "embedding": embedding,
            "quality_score": quality_score,
            "liveness_score": liveness_score,
            "reason": f"Качество: {quality_score:.2f}, Liveness: {liveness_score:.2f}"
        }
    
    def _estimate_quality(self, face_image):
        """
        Оценка качества кадра на основе размера и резкости.
        
        Returns:
            float: оценка качества от 0 до 1
        """
        height, width, _ = face_image.shape
        
        # Размер лица (чем больше, тем лучше)
        size_score = min(height / 100, 1.0)  # 100px = максимальный размер
        
        # Резкость (вариация Лапласа)
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 50, 1.0)  # 50 = хорошая резкость
        
        # Яркость
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128  # 128 = идеальная яркость
        brightness_score = max(0, min(1, brightness_score))
        
        # Итоговая оценка
        quality = 0.4 * size_score + 0.3 * sharpness_score + 0.3 * brightness_score
        return round(quality, 3)
    
    def _check_liveness_mock(self, image, face_location):
        """
        Mock-проверка liveness.
        
        В production здесь должен быть реальный anti-spoofing:
        - Texture analysis (LBP, BRISK)
        - Motion detection (optical flow)
        - Depth estimation (MediaPipe)
        
        Returns:
            float: liveness_score от 0 до 1
        """
        # В PoC возвращаем случайное значение в диапазоне 0.7-0.95
        # с небольшой детерминированной составляющей от размера лица
        top, right, bottom, left = face_location
        face_size = (bottom - top) * (right - left)
        
        # Более крупные лица чаще проходят liveness (эвристика)
        size_factor = min(face_size / 10000, 1.0)
        
        # Случайная составляющая
        import random
        random.seed(hash((top, right, bottom, left)))
        random_factor = 0.7 + 0.25 * random.random()
        
        score = 0.3 * size_factor + 0.7 * random_factor
        return round(score, 3)
    
    def process_batch(self, image_paths):
        """
        Обработка нескольких кадров (для multi-shot).
        
        Args:
            image_paths: список путей к изображениям
        
        Returns:
            list: результаты обработки каждого кадра
        """
        results = []
        for path in image_paths:
            results.append(self.process(path))
        return results