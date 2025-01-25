# Historical Text Classifier (by HistoryMoment)

## О проекте
HTCHM создан для обработки текстов и нахождения нужных тематических отрывков. По умолчанию проект ориентируется на область истории и увлекательные рассказы, предлагая пользователям удобные инструменты для поиска, анализа и извлечения ключевых тем и информации из текстов. Этот проект будет полезен как исследователям, так и любителям литературы, стремящимся глубже понять прочитанное.

## Установка

```bash
pip install git+https://github.com/yourusername/historical_text_classifier.git

## Использование

```python
from historical_text_classifier import RandomForestTextClassifier

# Инициализация модели
model = RandomForestTextClassifier()

# Обучение модели
model.train()

# Оценка модели
model.evaluate()

# Прогнозирование
prediction = model.predict("В 1757 г. для размещения Университета, основанного М.В. Ломоносовым в 1755 г., была приобретена усадьба князя Репнина на Моховой.")
print("Prediction:", prediction)

# Извлечение значимых отрывков
with open('books_1.txt', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

with open('books_1.txt', 'r', encoding=encoding, errors='replace') as file:
    input_text = file.read()

valuable_passages = model.extract_valuable_passages(input_text)
for passage, score in valuable_passages:
    print(f"{passage} (Score: {score:.2f})")

# Автоматическое расширение датасета
new_data = model.fetch_data_from_wikipedia("Hermitage museum")
for snippet in new_data:
    print(snippet)
```

## Документация
### Методы
1. train(): Обучает модель на данных из dataset.json.
2. predict(text): Предсказывает метку для заданного текста.
3. predict_proba(text): Возвращает вероятности классов для заданного текста.
4. evaluate(): Оценивает модель на тестовых данных.
5. extract_valuable_passages(input_text, threshold=0.5, min_length=10): Извлекает значимые отрывки из текста.
6. fetch_data_from_wikipedia(query, num_results=5): Получает сниппеты из Wikipedia по заданному запросу.
