# Первый Робот Политолог
Прототип реализации модели типа Retrieval Augmented Generation для автоматизации информационной поддержки политических экспертов. Идея бота - использование LLM поверх готовой базы знаний. Этапы функционирования: 

1) Семантический поиск – модель-трансформер кодирует вопрос и находит для него вектор с наибольшим скалярным произведением среди предварительно закодированных фрагментов текста
2) Формулирование ответа с помощью Text-Davinci-003 на основе фрагментов текста, отобранных на первом этапе

В качестве данных использовалось предобработанное текстовое содержание статей журнала "Гражданин. Выборы. Власть", опубликованных после 2018 года (https://www.rcoit.ru/lib/gvv/)
