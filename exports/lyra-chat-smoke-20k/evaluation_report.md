# Отчёт об оценке модели

Чекпойнт: `C:\Users\erikh\OneDrive\Документы\Lyra Chat 0.1\artifacts\runs\smoke-sft\checkpoints\best.pt`
Шаг обучения: 4; обработано токенов: 652
Параметров: 104,274,432
Устройство: `cpu`

## Loss / perplexity

| Сплит | Loss | Perplexity | Target-токены | Бачей |
| --- | ---: | ---: | ---: | ---: |
| val | 8.9813 | 7952.81 | 356 | 5 |
| test | 8.8706 | 7119.70 | 266 | 5 |

Train loss (последний из training_report): 8.822046279907227
Validation loss (последний из training_report): 9.028329902971295

## Метрики генерации

| Метрика | Значение |
| --- | ---: |
| responses | 16 |
| nonempty | 16 |
| empty_rate | 0.0 |
| stopped_rate | 1.0 |
| max_tokens_rate | 0.0 |
| mean_chars | 5.25 |
| min_chars | 1 |
| max_chars | 13 |
| distinct_1 | 0.5 |
| distinct_2 | 1.0 |
| mean_repetition_4gram | 0.0 |
| adjacent_duplicate_rate | 0.0 |
| role_bleed_rate | 0.0 |

## Примеры диалогов

### greeting (greeting)

**Вы:** Привет! Как дела?

**Модель:** на Осталось я

<sub>stop=<|turn_end|>, tokens=3, prompt_tokens=9</sub>

### short_question (short)

**Вы:** Что делаешь?

**Модель:** з

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=7</sub>

### slang (slang, short)

**Вы:** Здарова, чё как?

**Модель:** шоп

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=10</sub>

### clarification (clarification, context)

**Вы:** Я вчера начал смотреть новый сериал.

**Модель:** ,

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=11</sub>

**Вы:** Как думаешь, стоит его досматривать?

**Модель:** к

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=25</sub>

### topic_change (context, topic_change)

**Вы:** Привет, как прошли выходные?

**Модель:** на

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=10</sub>

**Вы:** Кстати, ты любишь кофе?

**Модель:** . малыш

<sub>stop=<|turn_end|>, tokens=2, prompt_tokens=21</sub>

### shorter (instruction, length)

**Вы:** Расскажи, чем ты обычно занимаешься вечером.

**Модель:** . на

<sub>stop=<|turn_end|>, tokens=2, prompt_tokens=12</sub>

**Вы:** Покороче, пожалуйста.

**Модель:** на призываю я

<sub>stop=<|turn_end|>, tokens=3, prompt_tokens=24</sub>

### emotional (emotion)

**Вы:** У меня сегодня был ужасный день, всё валится из рук.

**Модель:** остановке

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=17</sub>

### memory (context, memory)

**Вы:** Меня зовут Аня.

**Модель:** малыш

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=9</sub>

**Вы:** Как меня зовут?

**Модель:** которую на

<sub>stop=<|turn_end|>, tokens=2, prompt_tokens=19</sub>

### smalltalk (context, multi_turn)

**Вы:** Привет.

**Модель:** к остановке

<sub>stop=<|turn_end|>, tokens=2, prompt_tokens=6</sub>

**Вы:** Как настроение?

**Модель:** на

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=15</sub>

**Вы:** Что нового?

**Модель:** ,

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=23</sub>

### open (long)

**Вы:** Расскажи что-нибудь интересное.

**Модель:** я

<sub>stop=<|turn_end|>, tokens=1, prompt_tokens=10</sub>

## Ограничения

Это небольшая модель, обученная на разговорном корпусе с восстановленными эвристикой ролями. Она не обязана отвечать фактически верно и не является аналогом GPT-2 по интеллектуальным способностям. Примеры выше не отфильтрованы.
