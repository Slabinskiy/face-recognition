import cv2
import face_recognition
import os
from datetime import datetime
import asyncio
from telegram import Bot
from telegram.error import TelegramError

# Установка токена бота Telegram
TOKEN = ""
# ID пользователя, которому будет отправлено сообщение
USER_ID =   # замените на ваш настоящий Telegram User ID

# Создание объекта бота
bot = Bot(token=TOKEN)

# Загрузка фотографии, с которой будем сравнивать лица
imgmain = face_recognition.load_image_file('photo')
imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)  # Конвертация изображения из BGR (формат OpenCV) в RGB (формат face_recognition)

# Нахождение лиц и кодирование
encodeElon = face_recognition.face_encodings(imgmain)[0]  # Кодирование лица на изображении

# Инициализация видеозахвата
cap = cv2.VideoCapture(0)  # Запуск видеозахвата с веб-камеры

if not cap.isOpened():
    print("Ошибка при открытии веб-камеры.")
    exit(1)

# Определение пути для сохранения скриншотов
current_directory = os.getcwd()  # Получение текущего рабочего каталога
screenshots_directory = os.path.join(current_directory, "screenshots")  # Определение пути к каталогу для скриншотов
if not os.path.exists(screenshots_directory):
    os.makedirs(screenshots_directory)  # Создание каталога, если он не существует

face_present_start_time = None  # Время начала обнаружения лица
screenshot_taken = False  # Флаг, указывающий, был ли сделан скриншот

async def send_photo_to_telegram(photo_path):
    # Отправка изображения в Telegram
    with open(photo_path, 'rb') as photo_file:
        try:
            await bot.send_photo(chat_id=USER_ID, photo=photo_file)
            print(f"Labeled screenshot sent to Telegram user {USER_ID}")
        except TelegramError as e:
            print(f"Failed to send photo: {e}")

async def process_video():
    global face_present_start_time, screenshot_taken
    while True:
        ret, frame = cap.read()  # Чтение кадра с веб-камеры
        if not ret:
            print("Ошибка при захвате кадра.")
            break

        # Конвертация изображения из BGR в RGB
        rgb_frame = frame[:, :, ::-1]  # Конвертация кадра из BGR (формат OpenCV) в RGB (формат face_recognition)

        # Поиск всех лиц в текущем кадре видео
        face_locations = face_recognition.face_locations(rgb_frame)  # Нахождение всех лиц в кадре

        if face_locations:  # Если лица найдены
            if face_present_start_time is None:
                face_present_start_time = datetime.now()  # Установка времени начала обнаружения лица
            else:
                elapsed_time = (datetime.now() - face_present_start_time).total_seconds()  # Вычисление времени, прошедшего с начала обнаружения
                if elapsed_time >= 3 and not screenshot_taken:  # Если прошло 3 секунды и скриншот еще не был сделан
                    screenshot_path = os.path.join(screenshots_directory, f"face_{len(os.listdir(screenshots_directory)) + 1}.png")
                    cv2.imwrite(screenshot_path, frame)  # Сохранение скриншота
                    saved_screenshot = cv2.imread(screenshot_path)  # Загрузка сохраненного скриншота

                    # Поиск всех лиц на скриншоте
                    face_locations_test = face_recognition.face_locations(saved_screenshot)  # Нахождение всех лиц на скриншоте
                    encodings_test = face_recognition.face_encodings(saved_screenshot)  # Кодирование всех лиц на скриншоте

                    for (top, right, bottom, left), encodeTest in zip(face_locations_test, encodings_test):
                        results = face_recognition.compare_faces([encodeElon], encodeTest)  # Сравнение каждого лица на скриншоте с эталонным лицом
                        label = "Slava" if True in results else "Unknown"  # Определение метки для лица

                        cv2.rectangle(saved_screenshot, (left, top), (right, bottom), (255, 0, 255), 2)  # Отрисовка прямоугольника вокруг лица
                        cv2.putText(saved_screenshot, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)  # Добавление метки

                    print(f"Screenshot saved as {screenshot_path}")
                    screenshot_taken = True  # Установка флага, что скриншот был сделан
                    cv2.imshow('Test Image', saved_screenshot)  # Отображение скриншота с помеченными лицами

                    # Сохранение изображения с помеченными лицами
                    labeled_screenshot_path = os.path.join(screenshots_directory, f"labeled_face_{len(os.listdir(screenshots_directory)) + 1}.png")
                    cv2.imwrite(labeled_screenshot_path, saved_screenshot)  # Сохранение скриншота с метками
                    print(f"Labeled screenshot saved as {labeled_screenshot_path}")

                    # Отправка изображения в Telegram
                    await send_photo_to_telegram(labeled_screenshot_path)

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Отрисовка прямоугольника вокруг лиц в текущем кадре
        else:
            face_present_start_time = None  # Сброс времени начала обнаружения лица
            screenshot_taken = False  # Сброс флага скриншота

        cv2.imshow('Video', frame)  # Отображение текущего кадра

        if cv2.waitKey(25) == 13:  # Нажатие Enter для выхода из цикла
            break

    cap.release()  # Освобождение видеозахвата
    cv2.destroyAllWindows()  # Закрытие всех окон OpenCV

asyncio.run(process_video())  # Запуск асинхронной функции обработки видео