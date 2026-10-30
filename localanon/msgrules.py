"""Rules and dictionaries of the message filter: dev | commercial | personal | other.

The rules are a fallback filter for when there is no filter model: they remove only on a strong
and unambiguous signal; losing a message about development is worse than missing a commercial one
(the main pipeline and the model clean up the rest).

How it is scored:
  * the text is normalized (lowercase, ё→е), homonym phrases are dropped ("насчёт задачи",
    "за счёт кэша" are not about invoices, "кнопка «Отпуск»" is not about a vacation);
  * dev context: an interface/code word (экран, кнопка, страница, форма, поле, модуль, виджет,
    метод, API, эндпоинт, таблица, корзина, карточка …) within five words of a commercial one
    (оплата, цена, счёт, стоимость …) — the commercial word is dropped, development gets a plus;
  * dictionaries — word stems with an ending allowance (up to three letters for short stems, up to
    six for stems of five letters or more: ``тестир`` → тестирование, ``авториз`` → авторизация) or
    explicit regexes; each stem has a weight 1–3;
  * code and technical markers (snake_case outside a @handle, camelCase, file extensions, HTTP
    codes, git/jira links, backquotes) — development weight;
  * amounts with a currency — commercial ("500к" — only next to a money word); a phone, a card
    number (13–19 digits, Luhn check; solid 17–19 digits — only with the word "карта",
    "оплата", "перевод", "счёт", visa/mastercard/мир, otherwise it is an id) — personal;
  * result: development wins on a tie ("оплата" as a payment module, "счёт" in a dispute about a
    task — dev); with weak signals (< 3) 1–3 previous messages of the same chat help (with decay):
    development from context always counts, commercial and personal — only if the message itself
    hints at them; no signals — other, except a short deadline reply after a strong commercial
    message ("Давайте в среду" after "Когда оплатите счёт?" — commercial).
"""
from __future__ import annotations

import re

DEV, COMMERCIAL, PERSONAL, OTHER = "dev", "commercial", "personal", "other"

_SUF_SHORT = "[а-яa-z]{0,3}"           # a stem shorter than 5 letters: "баг" → баги, but not "багаж"
_SUF_LONG = "[а-яa-z]{0,6}"            # a stem of 5 letters or more: "тестир" → тестирование


def _rx(stem: str) -> re.Pattern:
    if stem.startswith("re:"):
        body = stem[3:]
    else:
        body = re.escape(stem) + (_SUF_LONG if len(stem) >= 5 else _SUF_SHORT)
    return re.compile(r"(?<![а-яa-z0-9])(?:" + body + r")(?![а-яa-z0-9])")


# Full words (after normalization) the dictionary must not catch: homonyms and neighbours.
STOP_PREFIXES = (
    "актив", "актер", "актри", "актуал", "акци", "цент", "ценю", "ценн", "договорил", "договорит",
    "подписк", "подписчик", "счетчик", "счетн", "пенис", "штрафбат", "сдачк", "срочн", "платежк",
    "детал", "детект", "жених", "женск", "мамонт", "папк", "сынов", "долгот", "долго",
    "картин", "карточ", "картог", "врачеб", "семьдес", "мужск", "мужч", "ценз", "релизн", "финт",
    "сроч", "здоровен", "кодек", "коди", "кодир", "картридж", "картоф", "налогооб", "премиал",
    "премиум", "платежн", "тестер", "погоди", "футболк", "отпускае", "отпускат",
    "классн", "полезн", "базар", "багаж", "гитар", "мокр", "тесть", "тесто", "поправляйс",
    "поправляйт",
)


def _stopped(word: str) -> bool:
    return any(word.startswith(p) for p in STOP_PREFIXES)


# ---------------------------------------------------------------------------------------------
# Dictionaries: (stem or re:regex, weight)
# ---------------------------------------------------------------------------------------------
DEV_TERMS = [
    # tasks and process
    ("задач", 2), ("тикет", 3), ("таск", 3), ("спринт", 3), ("джир", 3), ("jira", 3),
    ("бэклог", 3), ("беклог", 3), ("backlog", 3), ("ревью", 3), ("review", 3), ("рефактор", 3),
    ("тред", 1), ("thread", 1),
    ("refactor", 3), ("оптимиз", 2), ("интеграци", 2), ("re:баг(?:а|и|у|ов|ом|е|ам|ами|ах)?", 3),
    ("bug", 3), ("багфикс", 3),
    ("фич", 3), ("feature", 3), ("хотфикс", 3), ("hotfix", 3), ("фикс", 3), ("re:fix(?:ed|es|ing)?", 3),
    ("патч", 3), ("patch", 3), ("эпик", 2), ("story", 1), ("issue", 3), ("qa", 3), ("тестировщ", 3),
    ("тестир", 2), ("тест", 2), ("автотест", 3), ("юнит", 3), ("регресс", 2), ("ретест", 3),
    ("воспроизв", 3), ("воспроизвод", 3), ("дебаж", 3), ("дебаг", 3), ("debug", 3), ("логи", 2),
    ("re:лог(?:а|ов|ах|и)?", 2), ("re:logs?", 2), ("трейс", 3), ("traceback", 3), ("stacktrace", 3),
    ("exception", 3), ("исключени", 2), ("ошибк", 2), ("error", 2), ("краш", 3), ("crash", 3),
    ("падает", 2), ("упал", 2), ("упала", 2), ("упало", 2), ("не работает", 2), ("не проходит", 2),
    ("не открывается", 2), ("не грузится", 2), ("тормозит", 2), ("виснет", 2), ("зависает", 2),
    ("null", 3), ("undefined", 3), ("nil", 2), ("таймаут", 3), ("timeout", 3), ("флак", 2),
    ("дубл", 2), ("выборк", 2), ("утечк", 2), ("ретра", 2), ("retry", 2), ("rollback", 3),
    ("re:rolled back", 3),
    # code and infrastructure
    ("re:код(?:а|у|ом|е)?", 2), ("re:code", 2), ("коммит", 3), ("commit", 3), ("закоммит", 3),
    ("запуш", 3), ("push", 2), ("мерж", 3), ("мердж", 3), ("merge", 3), ("re:пр(?:ы|а|ов)?", 1),
    ("re:pr", 2), ("pull request", 3), ("пулл", 3), ("реквест", 2), ("ветк", 3), ("branch", 3),
    ("репозитор", 3), ("репо", 3), ("repo", 3), ("re:гит(?:а|е|у|ом|хаб[а-я]*|лаб[а-я]*)?", 3),
    ("re:git(?:hub|lab)?", 3), ("деплой", 3),
    ("deploy", 3), ("deployment", 3), ("выкат", 3), ("выкладк", 3), ("накат", 3), ("раскат", 3),
    ("релиз", 3), ("release", 3), ("сборк", 3), ("билд", 3), ("build", 3), ("стенд", 3), ("re:прод(?:а|е|у)?", 3),
    ("продакшн", 3), ("production", 3), ("re:dev", 2), ("стейдж", 3), ("staging", 3), ("re:test", 2),
    ("api", 3), ("апи", 3), ("эндпоинт", 3), ("endpoint", 3), ("ручк", 2), ("бэкенд", 3), ("бекенд", 3),
    ("backend", 3), ("бэк", 2), ("фронт", 3), ("frontend", 3), ("верстк", 3), ("сверст", 3),
    ("макет", 3), ("фигм", 3), ("figma", 3), ("дизайн", 2), ("re:ui|ux", 3), ("re:тз", 3), ("техзадан", 3),
    ("спецификац", 2), ("требовани", 2), ("swagger", 3), ("postman", 3), ("docker", 3), ("кубер", 3),
    ("kubernetes", 3), ("k8s", 3), ("ci/cd", 3), ("пайплайн", 3), ("pipeline", 3), ("миграци", 3),
    ("migration", 3), ("re:бд", 3), ("re:база данных|базы данных|базе данных", 3), ("database", 3),
    ("sql", 3), ("postgres", 3), ("orm", 3),
    ("mysql", 3), ("redis", 3), ("редис", 3), ("mongo", 3), ("кэш", 2), ("кеш", 2), ("cache", 2),
    ("сервер", 2), ("server", 2), ("хостинг", 2), ("nginx", 3), ("апстрим", 3), ("ssl", 2),
    ("сертификат", 1), ("докер", 3), ("контейнер", 2), ("микросервис", 3), ("сервис", 1),
    ("очеред", 1), ("кафк", 2), ("kafka", 3), ("воркер", 3), ("worker", 2), ("крон", 2), ("cron", 3),
    ("json", 2), ("xml", 2), ("yaml", 2), ("токен", 2), ("token", 2), ("авториз", 2), ("auth", 2),
    ("логин", 1), ("пароль", 1), ("кнопк", 2), ("экран", 2), ("скрин", 1), ("виджет", 3), ("компонент", 2),
    ("модул", 2), ("функци", 2), ("метод", 2), ("класс", 1), ("скрипт", 2), ("библиотек", 2),
    ("пакет", 1), ("зависимост", 2), ("верси", 1), ("апдейт", 1), ("обновлени", 1), ("мобилк", 3),
    ("ios", 3), ("android", 3), ("андроид", 3), ("айос", 3), ("айфон", 2), ("apk", 3), ("testflight", 3),
    ("ipa", 2), ("планшет", 2), ("сафари", 3), ("safari", 3), ("re:хром[а-я]{0,2}", 2), ("браузер", 2),
    ("приложени", 2), ("сайт", 1), ("страниц", 1), ("форм", 1), ("запрос", 1), ("респонс", 3),
    ("response", 2), ("request", 2), ("рендер", 3), ("лейаут", 3), ("layout", 2), ("адаптив", 2),
    ("пользовател", 2), ("юзер", 2), ("user", 1), ("админк", 3), ("дашборд", 2), ("парсер", 3),
    ("парсин", 3), ("валид", 3), ("регулярк", 3), ("regex", 3), ("алгоритм", 2), ("структур", 1),
    ("архитектур", 2), ("рефакторинг", 3), ("нейминг", 3), ("линтер", 3), ("линт", 3), ("форматир", 1),
    ("python", 3), ("django", 3), ("flask", 3), ("fastapi", 3), ("react", 3), ("vue", 3), ("angular", 3),
    ("javascript", 3), ("typescript", 3), ("kotlin", 3), ("swift", 3), ("java", 2), ("php", 3), ("golang", 3),
    ("node", 2), ("npm", 3), ("pip", 2), ("venv", 3), ("sdk", 3), ("webhook", 3), ("вебхук", 3),
    ("колбэк", 3), ("колбек", 3), ("callback", 3), ("шлюз", 2), ("эквайринг", 2), ("песочниц", 2),
    ("сокет", 3), ("websocket", 3), ("http", 2), ("https", 1), ("urls", 2), ("roadmap", 1),
    ("пагинац", 3), ("сериализ", 3), ("сортир", 2), ("re:id", 2), ("re:клик[а-я]{0,6}", 2),
    ("бэкфилл", 3), ("backfill", 3), ("ротаци", 2), ("импорт", 2), ("экспорт", 1), ("csv", 3),
    ("webp", 3), ("re:lazy", 2), ("шаринг", 2), ("index", 2), ("legacy", 2), ("handler", 3),
    ("хендлер", 3), ("обработчик", 2), ("контроллер", 3), ("округлени", 2), ("статус", 1),
    ("re:safe area", 3), ("xiaomi", 2), ("apns", 3), ("пуш", 2), ("нотификаци", 2), ("уведомлени", 1),
    # work on a task, deadlines inside the work
    ("додел", 2), ("поправ", 2), ("почин", 2), ("исправ", 2), ("допил", 3), ("запил", 3), ("пилю", 3),
    ("прикрут", 3), ("подключ", 2),
    ("re:настро(?:ить|ил[аи]?|им|ю|ишь|ит|ят|йк[а-я]*|ен[аоы]?|йщик[а-я]*|иваем|ивал[аи]?|иваю|ивать)", 2),
    ("отрису", 2), ("накид", 1), ("проверю", 1),
    ("протест", 3), ("перезал", 3), ("залил", 2), ("залью", 2), ("выложил", 2), ("закрыл задач", 3),
    ("сделаю", 1), ("доделаю", 2), ("добавлю", 1), ("добавил", 1), ("реализ", 2), ("внедр", 1),
    ("демо", 1), ("прототип", 2), ("mvp", 3), ("стори", 1), ("фикстур", 3),
    ("re:мок(?:и|а|ов|ами|ах|нуть|ать)?", 2), ("mock", 3),
    ("конфиг", 3), ("config", 3), ("env", 2), ("переменн", 2), ("параметр", 1), ("аргумент", 1),
    ("баз", 1), ("таблиц", 1), ("индекс", 1), ("колонк", 1), ("поле", 1), ("поля", 1), ("схем", 1),
    ("ссылк", 1), ("ссылку", 1), ("доступ", 1), ("права", 1), ("роли", 1), ("запис", 1),
    ("неправильн", 1), ("некорректн", 2), ("сделаем", 1), ("сделал", 1), ("готово", 1),
    ("re:добав[а-я]{1,4} (?:поле|поля|кнопк[а-я]+|экран[а-я]*|валидаци[а-я]+|проверк[а-я]+|метод[а-я]*|тест[а-я]*)", 2),
    ("беру", 1), ("возьму", 1), ("приоритет", 1), ("аналитик", 1),
]

COMMERCIAL_TERMS = [
    ("договор", 3), ("контракт", 3), ("re:доп\\.?\\s?соглашени[а-я]{0,2}", 3), ("допник", 3),
    ("допсоглашени", 3),
    ("соглашени", 2), ("re:акт(?:а|ы|у|ом|е|ов|ам|ами|ах)?", 3), ("re:счет(?:а|у|ом|е|ов|ам|ами|ах)?", 3),
    ("инвойс", 3), ("invoice", 3), ("оплат", 3), ("оплач", 3), ("платеж", 3), ("предоплат", 3),
    ("постоплат", 3),
    ("re:(?:до|за|пере)?плат(?:ил[аи]?|им|ят|ить|ите|ишь|ит)", 3), ("доплат", 3), ("доплач", 3),
    ("re:платить не будем|не будем платить|не заплатим|не оплатим", 3),
    ("аванс", 3), ("re:ндс", 3), ("vat", 3), ("re:смет(?:а|ы|у|ой|е)", 3), ("стоимост", 3), ("re:цен(?:а|ы|у|ой|е)", 2),
    ("ценник", 2), ("прайс", 3), ("price", 2), ("cost", 2), ("бюджет", 2), ("budget", 2), ("реквизит", 3),
    ("re:юр\\.?\\s?лиц[а-я]{0,2}", 3), ("re:ооо", 3), ("re:ип", 1), ("re:инн|кпп|огрн|бик", 3),
    ("re:р/с|к/с|расчетн[а-я]{0,2} счет[а-я]{0,2}", 3), ("неустойк", 3), ("re:пен(?:и|я|ей|ям|ями|ях)", 3),
    ("штраф", 3), ("penalty", 3), ("претензи", 3), ("просроч", 3), ("задолженност", 3),
    ("re:долг(?:а|у|ом|е|и|ов|ам|ами|ах)?", 2), ("расторг", 3), ("расторж", 3), ("юрист", 3), ("lawyer", 3),
    ("бухгалтер", 3), ("бухгалтери", 3), ("accountant", 3), ("закрывающ", 3), ("закрывашк", 3),
    ("re:упд", 3), ("накладн", 2), ("оригинал", 1), ("подпис", 2), ("подпиш", 2), ("re:эцп|эдо|диадок", 3),
    ("re:сда(?:ть|м|ем|ете|ют|л|ла|ли|ча|чи|че|чу|чей|дим|вать|ем)", 3), ("приемк", 2), ("этап", 2),
    ("milestone", 2),
    ("майлстоун", 2), ("дедлайн", 1), ("deadline", 1), ("re:срок(?:а|у|ом|е|и|ов|ам|ами|ах)?", 1),
    ("re:перенос[а-я]{0,2} срок", 3), ("re:сдвиг[а-я]{0,2} срок", 3), ("re:сдвинуть срок", 3),
    ("re:перенести срок", 3), ("опоздани", 1), ("ставк", 2), ("re:рейт|rate", 2), ("hourly", 3),
    ("тариф", 3), ("скидк", 3), ("наценк", 3), ("марж", 3), ("прибыл", 2), ("выручк", 3), ("рентабельн", 3),
    ("коммерческ", 3), ("re:кп", 2), ("предложени", 1), ("тендер", 3), ("заказчик", 1), ("исполнител", 1),
    ("подрядчик", 2), ("субподряд", 3), ("обязательств", 3), ("гарантийн", 2), ("re:условия договора", 3),
    ("re:nda", 3), ("re:sow", 3), ("agreement", 2), ("contract", 3), ("payment", 1), ("paid", 2),
    ("overdue", 3), ("settle", 2),
    ("refund", 3), ("quote", 2), ("estimate", 1), ("оценк", 1), ("часов", 1), ("re:часы", 1), ("таймшит", 3),
    ("timesheet", 3), ("отчет по часам", 3), ("к оплате", 3), ("деньг", 2), ("денег", 2), ("финанс", 2),
    ("re:доход(?:а|ы|у|ов|ом|е|ами|ах|ность|ности)?", 2), ("налог", 3), ("самозанят", 3), ("выплат", 2),
    ("перечисл", 2), ("транш", 3),
    ("компенсац", 2), ("возмещ", 2), ("убытк", 3), ("ущерб", 3),
    ("re:исков[а-я]{0,3}|судебн[а-я]{0,3}|в суде?|до суда|иском|подать иск|подали иск|подадим иск|иск в", 2),
    ("арбитраж", 3),
    ("нотариус", 3), ("доверенност", 3), ("печат", 1), ("скан", 1),
    ("re:\\d[\\d\\s.,]*\\s?(?:руб[а-я]*|р\\.|₽|\\$|€|usd|eur|евро|доллар[а-я]*)", 2),
    ("re:(?:руб|₽|\\$|€|usd|eur)\\s?\\d", 2), ("стоит", 1), ("дорого", 1), ("бесплатно", 1), ("сумм", 2), ("итого", 1),
    ("выстав", 2), ("re:сколько (?:\\S+ ){0,2}?должн[а-я]+", 3), ("re:сколько стоит", 2),
    ("санкци", 3), ("финдир", 2),
]

PERSONAL_TERMS = [
    ("отпуск", 3), ("vacation", 3), ("больничн", 3), ("re:sick", 3), ("заболел", 3), ("болею", 3), ("болеет", 3),
    ("приболел", 3), ("заболева", 3), ("простыл", 3), ("температур", 2), ("врач", 3), ("doctor", 3),
    ("больниц", 3),
    ("hospital", 3), ("таблетк", 3), ("аптек", 3), ("здоровь", 3), ("самочувств", 3), ("простуд", 3),
    ("ковид", 3), ("covid", 3), ("грипп", 3), ("re:семь(?:я|и|е|ю|ей)", 3), ("family", 3),
    ("re:жен(?:а|ы|е|у|ой|юсь|ился|имся|ится|иться)", 3), ("wife", 3), ("re:муж(?:а|у|ем|е)?", 3),
    ("husband", 3),
    ("re:дет(?:и|ей|ям|ями|ях)", 3), ("kids", 3), ("ребен", 3), ("ребят", 1), ("re:сын(?:а|у|ом|е)?", 3),
    ("re:доч(?:ь|ка|ке|ку|ки|кой|ери|ерью|ек|кам|ками|ках)", 3), ("re:мам(?:а|ы|е|у|ой)", 3),
    ("re:пап(?:а|ы|е|у|ой)", 3), ("бабушк", 3),
    ("дедушк", 3), ("родител", 3), ("re:день рождени[а-я]", 3), ("re:днем рождени[а-я]", 3), ("re:др", 2),
    ("днюх", 3), ("birthday", 3), ("свадьб", 3), ("wedding", 3), ("юбиле", 3), ("поздравл", 2),
    ("поздрав", 2), ("праздни", 2), ("re:с новым годом|с наступающим", 3), ("re:8 марта|23 февраля", 3),
    ("выходн", 1), ("отдых", 2), ("отдохн", 2), ("путешеств", 2), ("поездк", 2), ("отел", 2), ("самолет", 2),
    ("re:мор(?:е|я|ю|ем)", 2), ("re:дач(?:а|и|е|у|ей)", 2), ("ресторан", 2), ("re:бар(?:а|е|у)?", 2),
    ("пиццери", 2), ("кафе", 1),
    ("re:пив(?:о|а|у|ом|е)", 2), ("re:вин(?:о|а|у|ом|е)", 2), ("футбол", 2), ("хокке", 2), ("спортзал", 2),
    ("тренировк", 2), ("re:кин(?:о|ошк)", 1), ("re:сериал(?:а|ы|ов|е|ом|ами|ах|чик[а-я]*)?", 2),
    ("сезон", 1), ("спойлер", 2), ("фильм", 1),
    ("собак", 2), ("re:кот(?:а|у|ом|е|ы|ик)?", 2),
    ("кошк", 2), ("квартир", 2), ("ипотек", 3), ("кредит", 2), ("переезд", 2), ("переезжа", 2), ("прописк", 3),
    ("паспорт", 3), ("passport", 3), ("снилс", 3), ("загран", 3), ("re:домашн[а-я]{0,2} адрес", 3),
    ("re:адрес (?:прописки|доставки|проживания)", 3), ("re:адрес(?:а|у|ом|е|ов|ам|ами|ах)?", 1), ("личк", 2),
    ("re:скинь(?:те)?|закинь(?:те)?|сбрось(?:те)?", 1),
    ("re:сбер|тинькофф|тиньков|альфа-банк|втб|райффайзен", 2), ("re:номер телефона", 2),
    ("re:мой номер", 2), ("re:(?:\\+7|8)[\\s(-]*\\d{3}[\\s)-]*\\d{3}[\\s-]*\\d{2}[\\s-]*\\d{2}", 3),
    ("зарплат", 3), ("re:зп", 3), ("salary", 3),
    ("оклад", 3), ("преми", 3), ("re:кадр(?:ы|ов|ам|ами|ах)", 2), ("re:hr", 3), ("трудов", 3), ("увольн", 3),
    ("увольня", 3), ("re:ухожу (?:из|с) (?:компании|работы|проекта)", 3), ("resign", 3), ("оффер", 3),
    ("offer", 3), ("собесед", 3), ("собес", 2), ("резюме", 3), ("испытательн", 3),
    ("отгул", 3), ("устал", 2), ("выспал", 2), ("не выспал", 2), ("настроени", 2), ("погод", 2), ("пробк", 2),
    ("re:обед(?:а|у|ом|е|ать|аю)?", 1), ("ужин", 2), ("завтрак", 2), ("беремен", 3), ("роды", 3), ("декрет", 3),
    ("похорон", 3), ("умер", 3), ("соболезн", 3), ("школ", 2), ("садик", 3), ("экзамен", 2), ("диплом", 2),
    ("церк", 2), ("политик", 1), ("re:выбор(?:ы|ов|ам|ах)", 1),
    ("re:личн(?:о|ый|ая|ое|ые|ого|ому|ым|ых|ую)", 2), ("люблю", 2),
    ("целую", 3), ("обнимаю", 3), ("скучаю", 3), ("голов", 1), ("болит", 3), ("стоматолог", 3), ("зуб", 2),
    ("спин", 1), ("травм", 3), ("авари", 1), ("лекарств", 3), ("re:опозда(?:ю|л|ла|ем)", 1), ("проспал", 2),
    ("новорожд", 3), ("племян", 3), ("брат", 2), ("сестр", 2), ("подруг", 2), ("утренник", 2),
    ("подар", 2), ("re:скин[уе]мся|скинусь|сброс[иь]мся|сброшусь|скидываемся|скинемся", 2),
    ("девушк", 2), ("парень", 2), ("пожени", 3), ("развод", 3), ("алимент", 3), ("sick leave", 3),
    ("day off", 3), ("holiday", 2), ("weekend", 1),
]

# homonym phrases: dropped before scoring
_NEUTRALIZE = [
    re.compile(r"\bна\s?счет\b"), re.compile(r"\bнасчет\b"), re.compile(r"\bза\s?счет\b"),
    re.compile(r"\bсчетчик[а-я]*"), re.compile(r"\bв конечном счете\b"),
    re.compile(r"\bучетн[а-я]+ запис[а-я]*"), re.compile(r"\bip[\s-]?адрес[а-я]*"),
    re.compile(r"\bадрес[а-я]* (?:сервера|почты|сайта|страницы|api|бд|базы)"),
    re.compile(r"\bemail[\s-]?адрес[а-я]*"), re.compile(r"\bкарт[а-я]* (?:сайта|проекта|процесса|пользовател[а-я]*)"),
    re.compile(r"\bтестов[а-я]* (?:оплат[а-я]*|платеж[а-я]*|счет[а-я]*|карт[а-я]*)"),
    re.compile(r"\bденежн[а-я]* перевод[а-я]*"),
    re.compile(r"\bличн[а-я]* кабинет[а-я]*"),                       # personal account page is interface, not personal
    re.compile(r"\bэтап[а-я]*(?= (?:тестир[а-я]*|разработк[а-я]*|интеграци[а-я]*|qa|ревью|отладк[а-я]*))"),
    re.compile(r"\bсда[а-я]{0,4}(?= (?:в|на) (?:тест[а-я]*|ревью|qa|проверк[а-я]*|приемочн[а-я]*))"),
    re.compile(r"\bприложени[а-я]*(?= (?:к договору|к контракту|№|n\d))"),  # contract appendix
    re.compile(r"\bсрок[а-я]* действия\b"),                            # card/token expiry
]
# a button «Отпуск», a section «Больничные» — an interface element label, not the topic
_UI_LABEL = re.compile(
    r"\b((?:кнопк|экран|раздел|вкладк|пункт|пол[еяй]|таб|меню|чекбокс|тогл|переключател|фильтр|"
    r"опци|статус|колонк|заголов|поле)[а-я]*) [«\"“][^»\"”]{1,40}[»\"”]")

# dev context: an interface/code word near (± 5 words) a commercial word
_UI_WORD = re.compile(
    r"^(?:(?:экран|кнопк|страниц|форм|пол[еяй]|модул|виджет|метод|api|эндпоинт|ручк|таблиц|корзин|"
    r"карточк|кабинет|раздел|вкладк|список|списк|фильтр|сортир|колонк|столбц|строк|чекбокс|селект|"
    r"дропдаун|инпут|попап|модалк|баннер|блок|компонент|сервис|воркер|крон|джоб|бд|баз[аеы]|"
    r"сущност|модел|класс|функци|хендлер|обработчик|контроллер|шлюз|эквайринг|песочниц|сендбокс|"
    r"подсчет|пересчет|округлени|валидаци|валидатор|маск|генератор|pdf|csv|импорт|экспорт|"
    r"синхрониз|интеграци|вебхук|webhook|колбэк|колбек|callback|флаг|событи|ивент|админк|дашборд|"
    r"лейбл|заголов|тултип|иконк|скрин|макет|дизайн|верстк|фронт|бэк|бек|"
    r"мобилк|сайт|бот|парсер|скрипт|миграци|индекс|кэш|кеш|респонс|json|хранит|int|float|число|"
    r"enum|енам|пагинац|фокус|каталог|товар|профил|промокод)[а-яa-z0-9]*"
    r"|расчет(?:а|у|ом|е|ы|ов)?|заказ(?:а|у|ов|ы|е|ом|ами|ах)?)$")
_MONEY_WORD = re.compile(
    r"^(?:оплат[а-я]*|платеж[а-я]*|платёж|цен(?:а|ы|у|ой|е|ами|ам|ах)|стоимост[а-я]*|"
    r"счет(?:а|у|ом|е|ов|ам|ами|ах)?|инвойс[а-я]*|invoice|тариф[а-я]*|скидк[а-я]*|сумм[а-я]*|итого|"
    r"аванс[а-я]*|акт(?:а|ы|у|ом|е|ов|ам|ами|ах)?|бюджет[а-я]*|смет[аыуео]й?|комисси[а-я]*|"
    r"бонус[а-я]*|баланс[а-я]*|транзакци[а-я]*|price|payment|cost)$")
_DEVCTX_WINDOW = 5
_DEVCTX_WEIGHT = 2.0

# "500к", "1,2 млн" — an amount only next to a money word (otherwise "500к записей")
_AMOUNT_K = re.compile(r"(?<![а-яa-z0-9])\d[\d\s.,]*\s?(?:тыс[а-я]*|млн|млрд|k|к)(?![а-яa-z0-9])")
_MONEY_NEAR = re.compile(
    r"руб|₽|\$|€|usd|eur|евро|доллар|оплат|плат[аеи]|стоим|стоит|\bцен[аыуе]\b|\bсчет|бюджет|смет|"
    r"аванс|сумм|деньг|денег|ставк|заплат|платеж|прайс|скидк|тариф|зарплат|\bзп\b|оклад|преми|кредит|"
    r"ипотек|транш|выручк|доход")

# Card number: 13–19 digits (groups separated by a space or a hyphen) that pass the Luhn check.
# Solid 17–19 digits (Discord snowflake, Telegram/Slack id, unix time in ms) are a card only
# with a money or card word in the message.
_CARD_RX = re.compile(r"(?<![а-яa-z0-9])\d(?:[ -]?\d){12,18}(?![а-яa-z0-9])")
_CARD_WORD = re.compile(
    r"(?<![а-яa-z])(?:карт(?:а|ы|у|ой|е|ах|ам|ами|очк[а-я]*)?|card|оплат[а-я]*|"
    r"перевод[а-я]*|переве[лдс][а-я]*|счет(?:а|у|ом|е|ов)?|visa|mastercard|maestro|мир)(?![а-яa-z])")
_CARD_WEIGHT = 3.0


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_number(norm: str) -> bool:
    for m in _CARD_RX.finditer(norm):
        digits = re.sub(r"[ -]", "", m.group(0))
        if not _luhn(digits):
            continue
        if len(digits) >= 17 and digits == m.group(0) and not _CARD_WORD.search(norm):
            continue                     # a solid long id without a money word
        return True
    return False


# technical markers on the source (not normalized) text
_CODE_MARKERS = [
    (re.compile(r"`[^`\n]+`|```"), 3),
    (re.compile(r"(?<![@\w.-])[a-z]+_[a-z][a-z0-9_]*\b(?![\w.-]*@)"), 3),   # snake_case (not a @handle, not an email)
    (re.compile(r"(?<![@\w.-])[a-z]+[A-Z][A-Za-z0-9]+\b"), 2),              # camelCase
    (re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"), 1),                 # PascalCase
    (re.compile(r"\.(?:py|js|ts|tsx|jsx|java|kt|swift|go|rs|php|rb|cs|cpp|h|sql|json|ya?ml|html|css|scss|xml|md|sh|toml|ini|log)\b"), 3),
    (re.compile(r"\b(?:4\d\d|5\d\d)\b(?!\s?(?:руб|р\.|₽|\$|к\b|k\b|тыс|млн|евро|usd))", re.I), 2),
    (re.compile(r"https?://(?:[\w.-]*(?:github|gitlab|bitbucket|jira|atlassian|localhost|127\.0\.0\.1|figma|swagger|sentry|grafana|notion)[\w.-]*)", re.I), 3),
    (re.compile(r"\b(?:[a-z]+/)?[a-z]+-[a-z]+(?:-[a-z]+)*\b(?=.*\b(?:ветк|branch|pr|мерж|merge)\w*)", re.I), 1),
    (re.compile(r"\bv?\d+\.\d+\.\d+\b"), 1),                             # version 1.2.3
    (re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/"), 3),
    (re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\s+(?:\*|FROM|INTO|\w+\s+SET)\b"), 3),
    (re.compile(r"[{}\[\]]\s*$|^\s*[{}\[\]]|=>|->|::|==|!=|\(\)|\$\{"), 1),
]

_CATS = (DEV, COMMERCIAL, PERSONAL)
_COMPILED = {
    DEV: [(_rx(s), w) for s, w in DEV_TERMS],
    COMMERCIAL: [(_rx(s), w) for s, w in COMMERCIAL_TERMS],
    PERSONAL: [(_rx(s), w) for s, w in PERSONAL_TERMS if w > 0],
}
_WORD = re.compile(r"[а-яa-z0-9]+")


def normalize(text: str) -> str:
    t = text.casefold().replace("ё", "е")
    t = _UI_LABEL.sub(r"\1 ", t)
    for rx in _NEUTRALIZE:
        t = rx.sub(" ", t)
    return t


def dev_context(norm: str) -> tuple[str, int]:
    """Commercial words next to interface/code words are dropped. → (text, how many dropped)."""
    words = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(norm)]
    ui = [i for i, (_, _, w) in enumerate(words) if _UI_WORD.match(w)]
    if not ui:
        return norm, 0
    money = [i for i, (_, _, w) in enumerate(words) if _MONEY_WORD.match(w)]
    cut = [j for j in money if any(abs(i - j) <= _DEVCTX_WINDOW for i in ui)]
    if not cut:
        return norm, 0
    out = list(norm)
    for j in cut:
        s, e, _ = words[j]
        out[s:e] = " " * (e - s)
    return "".join(out), len(cut)


def scores(text: str) -> dict[str, float]:
    """Raw scores by category for one message (no context)."""
    out = {c: 0.0 for c in _CATS}
    if not text or not text.strip():
        return out
    norm, devctx = dev_context(normalize(text))
    for cat, terms in _COMPILED.items():
        total = 0.0
        for rx, w in terms:
            for m in rx.finditer(norm):
                word = m.group(0)
                if _stopped(word):
                    continue
                total += w
                break                    # each stem once
        out[cat] = total
    if _AMOUNT_K.search(norm) and _MONEY_NEAR.search(norm):
        out[COMMERCIAL] += 2
    if _card_number(norm):
        out[PERSONAL] += _CARD_WEIGHT
    code = 0.0
    for rx, w in _CODE_MARKERS:
        if rx.search(text):
            code += w
    out[DEV] += code + (_DEVCTX_WEIGHT if devctx else 0.0)
    return out


def _words(text: str) -> int:
    return len(_WORD.findall(text.casefold()))


_DECAY = (0.5, 0.3, 0.2)
STRONG = 3.0
DECIDE = 2.0

# Topic inheritance: only commercial and only by a short reply without content of its own —
# "Давайте в среду", "до пятницы", "да, 150 тысяч" after a strong commercial message.
# A reply of ≤ 4 words made of filler words and time/quantity words (at least one of those),
# without mentions (<@U…>, @handle, #123, №14) and identifier digits (4+ digits in a row).
# "ок", "спасибо" — other; personal is never inherited (a short reply carries no personal data,
# and losing a neutral message breaks the conversation).
_FILLER = frozenset("""
ок окей ok да нет ага угу хорошо ладно давай давайте тогда лучше можно думаю наверное наверно
конечно точно пусть предлагаю скорее примерно ориентировочно мб ну вроде где-то пока же
""".split())
_TIME_WORDS = frozenset("""
в во до к на после через с по ближе конце концу конца начале начала неделе недели неделю
следующей следующую следующий этой эту числа число числу месяца месяц утра утром вечера вечером
обеда дня днем часов час часа минут минуты полчаса завтра послезавтра сегодня сейчас
понедельник понедельника вторник вторника среда среду среды четверг четверга пятница пятницу
пятницы суббота субботу субботы воскресенье воскресенья выходных выходные праздников квартала
квартал года год января февраля марта апреля мая июня июля августа сентября октября ноября
декабря тысяч тыс тысячи млн к k штук половину половина треть
monday tuesday wednesday thursday friday saturday sunday tomorrow today next week by on in at
end of month this
""".split())
_MENTION_OR_ID = re.compile(r"<@|(?<![\w.])@\w|#\d|№\s?\d|\d{4,}")


def _short_time_reply(text: str) -> bool:
    low = text.casefold().replace("ё", "е")
    if _MENTION_OR_ID.search(low):
        return False
    words = _WORD.findall(low)
    if not words or len(words) > 4:
        return False
    if not all(w in _FILLER or w in _TIME_WORDS or w.isdigit() for w in words):
        return False
    return any(w in _TIME_WORDS or w.isdigit() for w in words)


def classify_text(text: str, context: list[str]) -> str:
    """Rules and dictionaries: dev | commercial | personal | other."""
    own = scores(text)
    prev = [c for c in context if isinstance(c, str) and c.strip()][-3:]
    prev.reverse()                       # the nearest first
    if not any(own.values()):
        if prev and _short_time_reply(text):
            near = scores(prev[0])
            if near[COMMERCIAL] >= STRONG and near[COMMERCIAL] >= max(near[DEV], near[PERSONAL]):
                return COMMERCIAL
        return OTHER
    final = dict(own)
    if max(own.values()) < STRONG and prev:
        for c, k in zip(prev, _DECAY):
            cs = scores(c)
            for cat in _CATS:
                if cat == DEV or own[cat] > 0:   # development from context — always
                    final[cat] += k * cs[cat]
    dev, com, per = final[DEV], final[COMMERCIAL], final[PERSONAL]
    if dev >= DECIDE and dev >= com and dev >= per:
        return DEV
    if max(com, per) >= DECIDE:
        return COMMERCIAL if com >= per else PERSONAL
    if dev >= DECIDE:
        return DEV
    return OTHER
