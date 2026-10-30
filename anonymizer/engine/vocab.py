"""Shared vocabularies used across detectors (general, not corpus-specific).

Corpus-specific literal lists (specific client names, specific cities, brand
variants) live in ``anonymizer/known_values/`` and are optional.
"""
from __future__ import annotations

# Legal forms — kept verbatim (never masked), used to anchor org names.
LEGAL_FORMS = [
    "ООО", "ЗАО", "ОАО", "ПАО", "АО", "НАО", "ИП", "НКО", "АНО", "ГК", "ФГУП", "МУП",
    "LLC", "Ltd", "Inc", "Corp", "Co", "GmbH", "PLC", "S.A", "SA", "AG", "BV", "B.V",
    "Oy", "AB", "SRL", "SAS", "Pte", "Pvt", "LLP", "Limited", "Holdings", "Group",
]
# Housing/garden association forms (housing registries): «ТСЖ «Наш дом»», «СНТ
# «Берёзка»» are a counterparty name, like a legal form; they are not in LEGAL_FORMS (the column
# profile, packgen and csv_profile count legal forms by LEGAL_FORMS, their output stays the same); the
# opf_name rule and the tabular registry profile take LEGAL_FORMS + HOUSING_FORMS.
HOUSING_FORMS = ["ТСЖ", "ЖСК", "СНТ", "ГСК", "ДНТ", "ДНП", "ТСН", "ОСМД", "ЖК"]
LEGAL_SUFFIX_LATIN = [
    "Inc", "Ltd", "LLC", "LLP", "GmbH", "Corp", "Limited", "Holdings", "Pte",
    "B.V", "S.A", "N.V", "Oy", "AB", "Co", "Group", "Studios", "Studio", "Labs",
    "Lab", "Tech", "Digital", "Agency", "Software", "Systems",
]

# Generic descriptor words — kept verbatim inside company columns.
GENERIC_DESCRIPTORS = {
    "ФИРМА", "КОМПАНИЯ", "ГРУППА", "ГРУПП", "ХОЛДИНГ", "КОРПОРАЦИЯ", "ГК",
    "GROUP", "HOLDING", "COMPANY", "CORPORATION", "THE", "OF", "AND", "И",
}

# Left-window anchors that signal an org is nearby (for quoted / ambiguous names).
ORG_CONTEXT = (
    r"(?i)(ооо|зао|оао|пао|ао|компани|клиент|заказчик|поставщик|бренд|магазин|"
    r"холдинг|подрядчик|партн[её]р|конкурент|фирм|вендор|контрагент|студи|агентств)"
)

# UI labels / common words that look like «quoted names» but aren't.
UI_LABEL_STOP = {
    "назад", "далее", "заявка", "регион", "согласование", "отправить", "сохранить",
    "город", "дата", "да", "нет", "стороны", "договор", "заказчик", "компания",
    "наименование", "статус", "готово", "отмена", "выбрать", "поиск", "войти",
    "купить", "продолжить", "подтвердить", "удалить", "изменить", "добавить",
}

# Tech terms / frameworks / tools that must never become ORG/PERSON entities.
TECH_TERMS = {
    "kotlin", "java", "python", "react", "vue", "angular", "docker", "kubernetes",
    "figma", "api", "sdk", "json", "csv", "html", "css", "sql", "git", "github",
    "gitlab", "jira", "slack", "redis", "nginx", "postgres", "mysql", "mongodb",
    "flutter", "swift", "typescript", "javascript", "golang", "rust", "linux",
    "aws", "gcp", "azure", "npm", "webpack", "babel", "eslint", "grpc", "graphql",
}

# English common words that look name-ish but aren't personal names.
COMMON_EN = {
    "email", "name", "user", "login", "report", "project", "status", "admin",
    "system", "default", "test", "prod", "main", "master", "build", "commit",
    "error", "message", "content", "title", "description", "value", "data",
    "double", "single", "team", "group", "manager", "owner", "member", "guest",
}

# ---- Russian name morphology -------------------------------------------------
# Productive surname endings (nominative). Used to detect surname candidacy.
SURNAME_ENDINGS = (
    r"(?:ов|ев|[её]в|ин|ын|ский|цкий|ской|цкой|енко|ук|юк|чук|ян|янц|янк|дзе|"
    r"швили|адзе|берг|ман|штейн|off|ко|ец|ых|их)"
)

# Diminutive / hypocoristic given-name stems (require a vocative anchor).
DIMINUTIVES = [
    "Сереж", "Серёж", "Серёга", "Саш", "Сань", "Вань", "Паш", "Кать", "Оль",
    "Свет", "Тань", "Дим", "Димон", "Вов", "Кол", "Миш", "Наташ", "Юр", "Лен",
    "Ань", "Настя", "Кирюх", "Лёш", "Лех", "Ром",
]

# Transliteration table (Cyrillic -> Latin) for name matching.
TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}
# Cyrillic->Latin homoglyph map (for mixed-script stylization detection).
HOMOGLYPH = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "к": "k",
    "м": "m", "т": "t", "в": "b", "н": "h",
}
# Leet substitutions.
DELEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a",
          "$": "s", "8": "b", "9": "g", "2": "z", "6": "b"}

# Non-Latin script ranges (names hide from Latin/Cyrillic regexes here).
SCRIPT_RANGES = {
    "georgian": r"Ⴀ-ჿ",
    "armenian": r"԰-֏",
    "hebrew": r"֐-׿",
    "arabic": r"؀-ۿ",
    "cjk": r"一-鿿",
}
