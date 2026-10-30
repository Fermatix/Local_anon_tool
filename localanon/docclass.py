"""Classifier of "papers": contract, power of attorney, supplementary agreement, act, invoice,
UPD, waybill, claim, letter of guarantee, disagreement protocol — the ``config.DOC_KINDS`` kinds
(kinds without rules here are skipped).

Rules only: the file name and the start of the text (the first ~5 pages). Points add up from three
parts and are mapped to 0..1:

* **kind signs** — a title in the first ``TITLE_LINES`` meaningful lines ("ДОГОВОР №…",
  "Счёт на оплату", "АКТ", "ДОВЕРЕННОСТЬ", "Invoice", …; a list item, a long line and a sentence
  line do not count as a title; requisite lines and digit lines do not use up the window; a line
  with a correspondence word — "подписан", "получен" — is not a title unless it has "№" or a date,
  and with a correspondence verb — "выставлен", "подписали" — not even with them), the file name
  (whole words: "расчёт" is not "счёт", "договорённости" is not "договор", bare ``inv``/``ds``/``dop``
  are no hint), typical phrases ("именуемое в дальнейшем", "уполномочивает", "работы выполнены в
  полном объёме", "Итого к оплате", "грузополучатель", "требуем уплатить неустойку", "гарантирует
  оплату", "редакция Заказчика");
* **general paper signs** — INN/KPP/BIK/bank account, VAT, signatures and seal, document number
  and date, party roles nearby (Заказчик/Исполнитель), amounts;
* **development signs** (subtracted) — a title of a spec/specification/README/meeting notes/
  user agreement, development terms (API, JSON, sprint, repository…), code.

The removal threshold is ``DELETE_THRESHOLD``, below it down to ``BORDERLINE_THRESHOLD`` —
"borderline" (removed with borderline=True, marked "check" in the report). Without a title or a
file name hint the score does not rise above borderline: phrases in the body alone are no reason
for a confident removal (a letter "на счёт задачи", meeting notes discussing an act); the exception
is a 1C export without a title line: two kind phrases with a bank account and INN/BIK with values
and not a single development sign (DB schema, API fixture — no). An ASCII/Markdown table line
("… | int | 15") and a line where "№ / no." is followed by a word, not a number, are not titles. A
hint by the file name alone with development terms or code — not above borderline
("acceptance.txt" with acceptance criteria), except text with ≥ 4 kind points and a strong paper
sign (act act_15.docx with "доработка API" — confident). A file name alone with text without any
paper sign (no kind phrase, no general sign) — below the borderline band: "Расчёт нагрузки.xlsx",
"contract_tests.md" stay. Empty text (a scan in docx) with a name hint — borderline, as before.

Development terms (bug, test case, README, "ожидаемый результат", an API response example, a
tracker export header, code) with a hint other than the file name drop the kind only if there is
no strong paper sign: amounts with a currency, VAT, requisite values, party signatures and seal, a
role with a full name, "работы выполнены", "претензий не имеет", an offer, SOW, "именуемое в
дальнейшем" / "в лице", two kind phrases outside the title. A role with an organization
("Исполнитель: ООО «…»"), a "Сумма, руб." column with numbers with a thousands separator and
"Итого / Всего / Сумма / К оплате" from 1 000 also set the kind, but not above borderline (an
invoice example in a spec or README). "API", "JSON", "request/response", "ошибк", "воспроизв" are
the subject of an IT studio's work: with a title "Счёт № N от ДАТА" the kind is not dropped (not
below borderline), without number and date — dropped; a ticket word with such a title and a work
item with an amount at the line end ("Bug fixing: 1,200.00") without code — also not below
borderline. A print form template (≥ 2 markup placeholders ``{{…}}``, ``{%…%}``, ``${…}``,
``[[…]]``, ``<%…%>``, ``<ФИО>`` and not a single real value: requisites, amount, passport, full
name) is not a paper, even with a file name hint. Underscores and date blanks
(«___» ______ 20__ г.) are not placeholders: an empty form of underscores only is classified as a
regular paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import DOC_KINDS

HEAD_CHARS = 20000            # ≈ 5 pages
TITLE_LINES = 5               # how many first meaningful lines are searched for a kind title
DEV_TITLE_LINES = 10          # … and for a development document title (softer — in favour of "keep")
TITLE_MAX_CHARS = 80          # longer — not a title
TITLE_SENTENCE_CHARS = 40     # longer and ending with a period — a sentence, not a title
TITLE_MIN_LETTERS = 3         # shorter lines (dates, "г.", numbers) do not use up the title window
MIN_TEXT_LETTERS = 20         # fewer letters/digits — no text (a scan without OCR, an empty document)
DELETE_THRESHOLD = 0.6
BORDERLINE_THRESHOLD = 0.35
_SCALE = 8.0                  # points per unit: 8 points or more → 1.0
_BODY_CAP = 6.0               # points cap for phrases of one kind
_GENERIC_CAP = 8.0
_DEV_TERMS_CAP = 6.0
_NO_TITLE_MAX = DELETE_THRESHOLD - 0.01
_FNAME_ONLY_MAX = BORDERLINE_THRESHOLD - 0.01
_TITLE_PTS = 4.0
_FNAME_PTS = 3.0
_FNAME_PTS_DEV = 1.0          # file name with a development document title — a weak hint


@dataclass
class Detail:
    kind: str | None
    score: float
    borderline: bool
    features: list[str] = field(default_factory=list)   # what triggered (for the data owner's report)
    points: float = 0.0
    kind_points: dict = field(default_factory=dict)
    cues: list[str] = field(default_factory=list)       # "title", "filename" — strong kind hints


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts), re.I | re.M)


# --- titles (line start) -------------------------------------------------------------------
# after the kind word — a number, a date, a subject ("на оказание", "поставки") or the line end:
# "Договор продлён до марта", "Акт за сентябрь подписан", "Contract testing", "Service Level
# Agreement (SLA) портала" do not count as titles
_TITLE = {
    "contract": _rx(r"^(договор|контракт)(а|у|ом|е)?\s*(№|n\b|#|\d|на |об? |оказания|поставки|(суб)?подряда|"
                    r"купли|аренды|займа|(без)?возмездного|присоединения|-оферта|$)",
                    r"^соглашение\s*(№|n\b|#|\d|о |об |$)",
                    r"^(?:[\w-]+ ){0,3}(agreement|contract)\s*(no\b|#|№|\d|for\b|of\b|on\b|between\b|dated\b|$)",
                    r"^statement of work\b", r"^\(?sow\b", r"^(non-disclosure|nda)\b"),
    "supplementary_agreement": _rx(r"^(дополнительное|доп\.?)\s*соглашение", r"^допсоглашение",
                                   r"^(addendum|amendment|supplementa(ry|l) agreement)\b"),
    "power_of_attorney": _rx(r"^доверенность\b", r"^power of attorney\b"),
    "act": _rx(r"^акт(а|ы)?\s*(№|n\b|#|\d|сдачи|при[её]м|об? |выполн|оказан|сверки|$)",
               r"^(acceptance (certificate|act|report)|certificate of (completion|acceptance)|"
               r"act of acceptance|completion certificate)\b"),
    "invoice": _rx(r"^сч[её]т(-фактура)?\s*(№|n\b|#|\d|на оплату|от\s+\d|$)",
                   r"^(pro\s?-?forma )?(tax |commercial )?invoice\b", r"^bill\s*(no\b|#|number|$)"),
    "upd": _rx(r"^универсальный передаточный документ", r"^упд\b"),
    "waybill": _rx(r"^(товарн(ая|о)[- ]?(транспортная )?|транспортная )?накладная\b", r"^торг[- ]?12",
                   r"^(waybill|delivery note|bill of lading|packing list|consignment note)\b"),
    "claim": _rx(r"^(досудебная |претензионное )?претензи[яи]\b", r"^требование(?!\s+к\s)\b",
                 r"^(claim|letter of claim|notice of (breach|default|claim)|demand (letter|for payment))\b"),
    "guarantee_letter": _rx(r"^(гарантийное письмо|письмо[- ]гарантия|письмо гарантийное)\b",
                            r"^(letter of guarantee|guarantee letter)\b"),
    "disagreement_protocol": _rx(r"^протокол (урегулирования |согласования )?разногласий",
                                 r"^(protocol|record|statement) of disagreements?\b"),
}
# a line with such words is correspondence about the document, not its title ("готовой продукции",
# "(приложение № 2)", "согласования разногласий" — a title); a line with "№" or a date — a title
_TITLE_CHATTER = re.compile(r"\b(подписан|подпишем|подписать|подписали|выставлен|выставим|выставили|отправлен|"
                            r"отправим|отправили|получен|получили|продл[её]н|обсужда|ждем|прислать|"
                            r"согласован(?!\w*\s+разноглас))\w*", re.I)
# a correspondence verb (short participle, personal form) — not a title even with "№" and a date:
# "Счет № 15 от 01.10.2025 выставлен, но не отображается"; "полученного оборудования" is not a verb
_TITLE_CHATTER_VERB = re.compile(r"\b(?:(?:подписан|выставлен|отправлен|получен|продлен|продлён|согласован)[аоы]?\b"
                                 r"(?!\s+разноглас)|(?:подпишем|подписали|выставим|выставили|отправим|отправили|"
                                 r"получили|ждем|ждём|прислать)\b|обсужда\w*)", re.I)
_TITLE_NUMDATE = re.compile(r"№|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|"
                            r"июля|августа|сентября|октября|ноября|декабря)\b", re.I)
# "<kind> № N [от D] — не формируется / пример ответа API / проверить итог": after the number and/or
# date, through a dash, comma or colon — a task or problem description, not a title; "за период:
# 01.01.2025 - 30.09.2025" — a title
_MONTHS = r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
_DATE = r"(?:\d{1,2}[./]\d{1,2}[./]\d{2,4}|«?\d{1,2}»?\s+" + _MONTHS + r"(?:\s+\d{4})?)"
_TITLE_TAIL = re.compile(r"(?:№\s*[\w/{}.\-]+|" + _DATE + r")(?:\s+от\s+" + _DATE + r")?(?:\s*г\.?)?"
                         r"\s*[-,:]\s*(?P<tail>\S.*)$", re.I)
# a description is a negation, an error, a task ("нужно", "пример", TODO); "API", "тест…", "проверить",
# "исправить" without them are the work subject: "Счёт № 45 — разработка API", ": тестирование приложения" — a title
# after a number label ("№", "no.", "N", "#") — a value: digits or a code ("15", "ЦБ-15", "INV-2024-17"), not
# a word: "Invoice no. генерируется автоматически.", "Счет № формируется" — not a title; "Счет № от …",
# "Договор № б/н", "№ ДП/15", a cell label "Счет-фактура №⇥Номер ЦБ-12⇥от …" — a title. Only the first label
# in a line is checked; "#" after a letter (C#, F#) is not a label, "no" is a label only with ".", ":", "№" or a
# digit ("no VAT" — not); "бн", "б.н.", a single letter ("к договору № А") — a value
_TITLE_NO_LABEL_RX = r"№|\bno\b(?=\s*[.:№]|\s*\d)|\bn\b|(?<![a-zа-яё])#"
_TITLE_NO_LABEL = re.compile(_TITLE_NO_LABEL_RX, re.I)
_TITLE_NO_WORD = re.compile(r"(?:№|no|n|#)\s*[:.]?\s*(?!(?:от|of|dated|date|дата|номер|number|бн|б\.н)\b|[a-zа-яё]\b)"
                            r"[a-zа-яё]+(?![\w/-])", re.I)
_TAIL_DESC = re.compile(r"\bне\s|\bнет\b|ошибк|\bбаг|\bbug|пада|слома|пример|нужн|надо|todo|fixme", re.I)
# development terms with which a kind title alone is not a reason: ticket, bug report, README, test cases,
# API documentation ("пример ответа", endpoint) — they "hold as development"
_DEV_MARKERS = re.compile(r"эндпоинт|endpoint|\bбаг|\bbugs?\b|тест-кейс|test[- ]?cases?\b|не формируется|"
                          r"шаг\w* воспроизведения|ответ\w* (?:сервиса|сервера|api)|пример\w* (?:ответа|запроса)|"
                          r"readme|steps to reproduce|"
                          r"(?:ожидаем|фактическ)\w* результат|(?:expected|actual) result|"
                          r"\b(?:ключ|key)\b[^\n]{0,30}\b(?:тема|статус|summary|status)\b", re.I)   # tracker export header
# … while "API", "JSON", "request/response", "ошибк", "воспроизв" are an IT studio's work subject ("Доработка API и
# backend: 60 000", "Интеграция JSON-обмена с 1С — 80 000"): with a title "Счёт № N от ДАТА" the kind is not dropped
# (not below borderline); backend, development, integration were never markers
_DEV_SUBJECT = re.compile(r"\bapi\b|\bjson\b|\brequest\b|\bresponse\b|ошибк|воспроизв", re.I)
# a title with a number value and a date: "Счет № 21 от 01.05.2025", "Invoice No. 18 dated 01.04.2025"
_TITLE_NUMBERED = re.compile(r"(?:" + _TITLE_NO_LABEL_RX + r")\s*[:.]?\s*(?!(?:от|of|dated)\b)[\w/]", re.I)
_TITLE_DATED = re.compile(_DATE, re.I)
# a work item with an amount from 1 000 at the line end: "Исправление ошибки в API: 60 000", "Bug fixing — 1,200.00"
_ITEM_AMOUNT = re.compile(r"\S\s*[:—–-]\s*\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d{2})?\s*$", re.M)
# strong paper signs — with any of them development terms do not drop the kind: "именуемое", "в лице",
# an amount with a currency, VAT, requisite values, party signatures and seal, a role with a full name or a
# signature line, "работы выполнены", "претензий не имеет", an offer, SOW (and two kind phrases outside the title — in code)
_PAPER_STRONG = re.compile(r"именуем\w+ в дальнейшем|hereinafter (referred to as|called)", re.I)
# a label and a value — one or two words apart ("ИНН/КПП продавца: 7700000000/770001001", "ИНН поставщика 7700000000");
# the pair "ИНН/КПП … 7700000000/770001001" — anywhere in the line
_REQ_VALUE = _rx(r"\b(?:инн|кпп|бик|огрн(?:ип)?)(?:\s*/\s*(?:инн|кпп))?(?:\s+[а-яa-z]+){0,2}\s*[:№]?\s*\d{9,15}\b",
                 r"\bинн\s*/\s*кпп\b[^\n\d]{0,60}\b\d{10}(?:\d{2})?\s*/\s*\d{9}\b",
                 r"(?:\bр/сч?|\bк/сч?|расч[её]тн\w* сч[её]т|корр\w*\.? ?сч[её]т|сч\.\s*№)\s*[:№]?\s*\d{20}\b",
                 r"\b(?:407|408|301)\d{17}\b", r"\biban\b\s*:?\s*[a-z]{2}\d{2}(?:\s?[a-z0-9]){8,}")
# a table column "ИНН" / "ИНН/КПП" with a 10/12-digit number under it (INN as a column in a 1C export)
_INN_HEAD_CELL = re.compile(r"инн(?:\s*/\s*кпп)?")
_INN_CELL = re.compile(r"\d{10}(?:\d{2})?(?:\s*/\s*\d{9})?")
# a bank account and INN/BIK with values — a payment details header (a 1C invoice without a title line)
_BANK_ACCOUNT = _rx(r"(?:\bр/сч?|расч[её]тн\w* сч[её]т|сч\.\s*№)\s*[:№]?\s*\d{20}\b", r"\b40[78]\d{17}\b")
_TAX_VALUE = _rx(r"\b(?:инн|бик)(?:\s*/\s*кпп)?(?:\s+[а-яa-z]+){0,2}\s*[:№]?\s*\d{9,12}\b")
_SWIFT_VALUE = re.compile(r"\b(?:SWIFT|Swift|swift)(?:\s*code|/BIC)?\s*:?\s*[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
_STRONG_SIGNS = _rx(r"\bм\.\s?п\.", r"подпис[ьи] сторон", r"signatures?( of the parties)?\s*:", r"authori[sz]ed signatory",
                    r"(работы|услуги) (выполнены|оказаны)", r"претензий[^.\n]{0,60}не имеет", r"оферт",
                    r"statement of work", r"сальдо (начальное|конечное|на начало|на конец)|обороты за период",
                    r"\b(исполнител|заказчик|подрядчик|поставщик|покупател|продав[ец]|руководител|бухгалтер)\w*\s*_{3,}")
_ROLE_FIO = re.compile(r"(?i:\b(?:исполнитель|заказчик|подрядчик|поставщик|покупатель|продавец))\b[^\n]{0,60}?"
                       r"(?:[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.|[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+)")
# a party role with an organization ("Исполнитель: ООО «Код Студия»", "Покупатель: ИП Иванов Иван") — never in tickets
_ROLE_ORG = re.compile(r"\b(?:исполнител|заказчик|поставщик|покупател|подрядчик)\w*\s*:\s*"
                       r"(?:(?:ооо|ао|пао|зао|оао|ип)\s*[«\"“']|ип\s+[а-яё]+\s+[а-яё])", re.I)
# a column with a currency in the table header ("Сумма, руб.", "Стоимость, ₽") — with a number with a thousands separator
_CURRENCY_COLUMN = re.compile(r"\b(?:сумм|цен|стоимост|всего|итого)\w*\s*,\s*(?:руб|₽|rub\b|usd\b|eur\b|\$|€)", re.I)
_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?: \d{3})+(?:[.,]\d{1,2})?(?![\d])")
# "Итого", "Всего", "Сумма", "К оплате" with an amount from 1 000 (thousands separator): "Итого: 50 ч × 2 400 = 120 000",
# "Сумма 80 000"; "Итого к оплате: 100", "Всего 3 000 записей" — not
_TOTAL_AMOUNT = re.compile(r"(?:\bитого|\bвсего|\bсумм[аы]|\bк оплате)\b[^\n]{0,40}?(?<![\d.,])\d{1,3}(?: \d{3})+"
                           r"(?:[.,]\d{1,2})?(?![\d])(?![ \t]*(?!руб|р\.)[a-zа-яё])", re.I)
# "… от 01 октября 2025 г." — an abbreviation period, not a sentence end
_YEAR_ABBR = re.compile(r"\d{4}\s*г\.$")
# bank details lines (a 1C invoice header) do not use up the title window
_REQ_LINE = re.compile(r"^(бик|инн|кпп|сч\.\s*№|банк получателя|банк плательщика|получатель|плательщик)\b", re.I)
# --- file name hints (word boundaries) -------------------------------------------------------
_RU_END = r"([^а-я]|$)"
_EN_END = r"([^a-z]|$)"
_FNAME = {
    "contract": _rx(r"(^|[^а-я])договор(а|ы|ов|у|е|ом|ами|ах)?" + _RU_END,
                    r"(^|[^а-я])контракт(а|ы|ов|у|е|ом)?" + _RU_END,
                    r"(^|[^a-z])dogovor", r"(^|[^a-z])kontrakt", r"(^|[^a-z])contracts?" + _EN_END,
                    r"(^|[^a-z])agreements?" + _EN_END, r"(^|[^a-z])sow" + _EN_END, r"(^|[^a-z])nda" + _EN_END),
    "supplementary_agreement": _rx(r"доп\.?\s*согл", r"допсогл", r"(^|[^а-я])дс" + _RU_END,
                                   r"addendum", r"amendment", r"supplement"),
    "power_of_attorney": _rx(r"доверенност", r"doverennost", r"(^|[^a-z])poa" + _EN_END,
                             r"power[\s_-]*of[\s_-]*attorney", r"attorney"),
    "act": _rx(r"(^|[^а-я])акт(а|ы|ов|у|е|ом)?" + _RU_END, r"(^|[^a-z])akt" + _EN_END,
               r"(^|[^a-z])act" + _EN_END, r"acceptance"),
    "invoice": _rx(r"(^|[^а-я])сч[её]т(а|у|ом|е|ов|ы)?" + _RU_END, r"(^|[^а-я])сч[её]т-фактур",
                   r"(^|[^a-z])schet", r"invoice", r"(^|[^a-z])bill" + _EN_END, r"на[\s_-]*оплату"),
    "upd": _rx(r"(^|[^а-я])упд" + _RU_END, r"(^|[^a-z])upd" + _EN_END),
    "waybill": _rx(r"накладн", r"nakladn", r"торг[\s_-]*12", r"torg[\s_-]*12", r"waybill",
                   r"delivery[\s_-]*note", r"packing[\s_-]*list"),
    "claim": _rx(r"претензи", r"pretenzi", r"неустойк", r"(^|[^а-я])требование" + _RU_END,
                 r"(^|[^a-z])claims?" + _EN_END, r"demand[\s_-]*letter", r"notice[\s_-]*of[\s_-]*(breach|default)"),
    "guarantee_letter": _rx(r"гарантийн", r"garantijn", r"guarantee"),
    "disagreement_protocol": _rx(r"разноглас", r"raznoglas", r"disagreement"),
}
# --- phrases in the text: (label for the report, regex) ---------------------------------------
_BODY = {
    "contract": [
        ("именуемое в дальнейшем", r"именуем\w+ в дальнейшем"),
        ("предмет договора", r"предмет (договора|контракта|соглашения)"),
        ("настоящий договор", r"настоящ\w+ (договор|контракт)\w*|по настоящему договору"),
        ("срок действия договора", r"срок действия (договора|контракта)"),
        ("стороны договорились", r"стороны (договорились|согласовали)"),
        ("ответственность сторон", r"ответственность сторон|права и обязанности сторон"),
        # a short work contract: "Исполнитель обязуется разработать …", "Стоимость работ: 400 000 (…) рублей"
        ("исполнитель обязуется", r"(исполнитель|подрядчик|заказчик|поставщик|покупатель|арендатор|арендодатель)\w* обязу\w+"),
        ("цена договора / стоимость работ", r"(цена|стоимость) (договора|контракта)|"
                                            r"стоимость (работ|услуг)( по (настоящему )?договору)?\s*(составляет|:)"),
        ("hereinafter referred to as", r"hereinafter (referred to as|called)"),
        ("this agreement", r"this (agreement|contract)\b|under this sow\b"),
        ("the parties agree", r"the parties (hereby )?agree|whereas\b"),
        ("scope of work", r"scope of (work|services)|deliverables\b"),
        ("terms and conditions", r"terms and conditions|governing law|entire agreement"),
    ],
    "supplementary_agreement": [
        ("к договору №", r"к (договору|контракту|соглашению)\s*(№|n|от)"),
        ("стороны пришли к соглашению", r"стороны пришли к соглашению|стороны договорились о нижеследующем"),
        ("внести изменения в договор", r"внести (следующие )?изменения в (договор|контракт)|изложить в следующей редакции"),
        ("неотъемлемая часть договора", r"неотъемлемой частью (договора|контракта)"),
        ("продлить срок действия", r"продлить срок (действия )?(договора|контракта)"),
        ("amendment to the agreement", r"(amendment|addendum) (no\.? ?\d+ )?to the (agreement|contract)"),
        ("the parties agree to amend", r"agree to amend|is hereby amended|shall be amended"),
    ],
    "power_of_attorney": [
        ("уполномочивает", r"уполномочива\w+|уполномочен\w*"),
        ("доверяет", r"\bдоверяет\b|настоящей доверенностью|доверенность выдана"),
        ("представлять интересы", r"представлять (его |ее )?интересы|совершать (от имени|действия)"),
        ("право передоверия", r"(без|с) права? передоверия"),
        ("подпись доверенного лица", r"подпись (доверенного лица|представителя)|образец подписи"),
        ("hereby authorizes", r"hereby (authorizes?|appoints?|constitutes?)|attorney-in-fact|on behalf of"),
    ],
    "act": [
        ("акт выполненных работ", r"акт\w* (выполненных работ|оказанных услуг|сдачи[- ]при[её]мки|при[её]ма[- ]передачи|сверки)"),
        # and the canonical phrases of a 1C act: "Вышеперечисленные услуги выполнены полностью и в срок", "Заказчик
        # претензий по объему, качеству и срокам оказания услуг не имеет", "Всего оказано услуг 1, на сумму …"
        ("работы выполнены в полном объёме", r"(работы|услуги) (выполнены|оказаны)( полностью| в полном объ[её]ме| в срок)|"
                                             r"всего оказано услуг|вышеперечисленные (работы|услуги)"),
        ("претензий не имеет", r"претензий (по объ[её]му, качеству и срокам( оказания услуг| выполнения работ)? )?не имеет|"
                               r"без замечаний|без претензий"),
        ("исполнитель сдал, заказчик принял", r"(исполнитель|подрядчик) сдал|заказчик принял|принял\w* на себя"),
        # a reconciliation act without a title line: "Сальдо начальное", "Обороты за период", "По данным ООО «Ромашка», руб."
        ("сальдо / сверка", r"сальдо|взаимных расч[её]тов"),
        ("сальдо начальное / обороты за период", r"сальдо (начальное|конечное|на начало|на конец)|обороты за период"),
        ("по данным стороны", r"по данным (заказчика|исполнителя|поставщика|покупателя|ооо|ао|пао|зао|оао|ип)\b"),
        ("services have been rendered", r"(services|work) (have|has) been (rendered|completed|performed|accepted)"),
        ("accepted without objections", r"accepted (without|with no) (objections|reservations)|hereby accepts"),
    ],
    "invoice": [
        ("счёт на оплату", r"сч[её]т на оплату|сч[её]т-фактура|сч[её]т\s*№"),
        ("оплатить до", r"оплатить (до|в течение|не позднее)|срок оплаты"),
        ("назначение платежа", r"назначение платежа|основание платежа"),
        # "Поставщик ООО …" — and the cell "Поставщик:" in an xlsx view (a colon at the line edge is cut)
        ("поставщик / покупатель", r"поставщик\s*[:(]|покупатель\s*[:(]|плательщик\s*[:(]|получатель\s*[:(]|"
                                   r"\b(поставщик|покупатель|плательщик|получатель)\s+(ооо|ао|пао|зао|оао|ип)\b"),
        ("всего наименований", r"всего наименований|всего к оплате|итого к оплате"),
        ("invoice no / date", r"invoice (no\.?|number|#|date)|invoice is (due|payable)"),
        ("bill to / total due", r"bill to\b|billed to\b|(total|amount|balance) due\b|payment (terms|due)|due date"),
    ],
    "upd": [
        ("универсальный передаточный документ", r"универсальн\w+ передаточн\w+ документ|передаточн\w+ документ"),
        ("статус 1/2", r"статус[:\s]*[12]\b"),
        ("товар (груз) передал / получил", r"товар \(груз\) (передал|получил)|(передал|получил)\s*\(сдал\)"),
        ("основание передачи", r"основание передачи|данные о транспортировке|иные сведения об отгрузке"),
        ("ответственный за правильность оформления", r"ответственн\w+ за правильность оформления"),
    ],
    "waybill": [
        ("товарная накладная", r"товарн\w+ накладн\w+|транспортн\w+ накладн\w+|торг[- ]?12|накладная\s*№"),
        ("грузоотправитель / грузополучатель", r"грузоотправител|грузополучател"),
        ("отпуск груза", r"отпуск груза (разрешил|произв[её]л)|груз (принял|получил)"),
        ("масса груза / мест", r"масса (груза|брутто|нетто)|количество мест|всего мест"),
        ("consignee / shipper", r"consignee\b|shipper\b|consignor\b|carrier\b"),
        ("delivery note", r"delivery note|bill of lading|packing list|goods (received|delivered)"),
    ],
    "claim": [
        ("неустойка / штраф", r"неустойк\w+|\bпен[яи]\b|штрафн\w+ санкци|liquidated damages|penalty|late fee"),
        ("требуем уплатить", r"(требуем|требует|просим|прошу)[^.\n]{0,60}?(уплатить|оплатить|возместить|погасить|перечислить)|"
                             r"demands? (payment|that|the)|hereby demand"),
        ("просрочка / нарушение срока", r"просрочк\w+|нарушени\w+ срок\w*|delivered[^.\n]{0,30} late|late delivery|breach of"),
        ("иначе — суд", r"арбитражн\w+ суд|в судебном порядке|обратиться в суд|legal action|\bcourt\b|arbitration"),
        ("досудебный порядок", r"претензионн\w+ порядк|досудебн\w+|в течение \d+ (календарных |рабочих |банковских )?дней с (момента|даты) получения|"
                               r"within \d+ (business |calendar )?days"),
    ],
    "guarantee_letter": [
        ("гарантирует оплату", r"гарантиру\w+ (оплат\w+|выполнени\w+|погашени\w+|исполнени\w+)|guarantees? (payment|to pay|the payment)"),
        ("настоящим гарантирует", r"настоящим (письмом )?гарантиру\w+|hereby guarantees?"),
        ("исх. №", r"исх\.?\s*№"),
        ("по счёту №", r"по сч[её]ту\s*№|по договору\s*№|under invoice|under the agreement"),
        ("в срок до", r"в срок до \d|не позднее \d|в течение \d+ (банковских |рабочих |календарных )?дней"),
    ],
    "disagreement_protocol": [
        ("редакция заказчика / исполнителя", r"редакци\w+ (заказчика|исполнителя|поставщика|покупателя|подрядчика|сторон\w*|"
                                             r"арендатора|арендодателя|лицензиата|лицензиара)"),
        ("согласованная редакция", r"согласованн\w+ редакци\w+|принять в редакции|изложить в редакции|в следующей редакции"),
        ("разногласия", r"разноглас\w+|disagreements?|objections? to clause"),
        ("к договору №", r"к (договору|контракту|соглашению)\s*(№|n|от|оказания|поставки|подряда)"),
        ("пункт — редакция", r"пункт\w* \d+(\.\d+)*\s*[-:—–]|clause \d+(\.\d+)*\s*[-:—–]"),
    ],
}
# --- general paper signs -------------------------------------------------------------------
_REQUISITES = [
    ("инн", r"\bинн\b"), ("кпп", r"\bкпп\b"), ("бик", r"\bбик\b"), ("огрн", r"\bогрн(ип)?\b"),
    ("р/с", r"\bр/с\b|расч[её]тн\w+ сч[её]т|р/сч"), ("к/с", r"\bк/с\b|корр\w*\.? ?сч[её]т|к/сч"),
    ("tax id / iban", r"\b(tax id|vat (number|id|no\.?|reg)|iban|swift|bic code|ein)\b"),
]
_VAT = _rx(r"в т\.? ?ч\.? ндс", r"в том числе ндс", r"ндс не облагается", r"без ндс", r"ндс\s*\d{1,2}\s*%",
           r"including vat", r"vat\s*\(?\d{1,2}\s*%", r"excl(uding|\.) vat", r"sales tax")
_TOTAL = _rx(r"итого к оплате", r"всего к оплате", r"сумма прописью", r"^\s*(итого|всего)\s*[:：]",
             r"\b(grand )?total\s*[:：]", r"amount due")
_SIGN = _rx(r"\bм\.\s?п\.", r"подпис[ьи] сторон", r"подпись\b", r"\bsignatures?\b", r"authori[sz]ed signatory",
            r"\bstamp\b", r"расшифровка подписи")
_ADDR = _rx(r"реквизиты сторон", r"юридическ\w+ адрес", r"адреса и реквизиты", r"банковские реквизиты",
            r"bank details", r"registered (office|address)", r"legal address")
_NUMDATE = _rx(r"№\s*[\w/\-]+\s+от\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}",
               r"от\s+«?\d{1,2}»?\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}",
               r"\bdated\s+(as of\s+)?\w+ \d", r"\b(no\.?|№)\s*[\w/\-]+\s+(dated|of)\s+\d")
_INLICE = _rx(r"в лице\b", r"действующ\w+ на основании", r"represented by", r"acting (on the basis|under)")
_ROLES = re.compile(r"\b(заказчик|исполнител|подрядчик|поставщик|покупател|продав[ец]|арендатор|арендодател|"
                    r"лицензиар|лицензиат|доверител|customer|contractor|client|vendor|supplier|buyer|seller|"
                    r"licensor|licensee|principal)\w*", re.I)
# a currency sign or code before a number — "$4,000.00", "€ 1 200", "USD 4,000.00"; "${amount}", "$1" (an argument) — not amounts
_MONEY = _rx(r"\d[\d\s]*(?:[.,]\d{1,2})?\s*(руб|₽|rub\b|usd\b|eur\b|gbp\b|\$|€|£)", r"рубл\w+ \d{2} коп", r"\d\s*коп\.",
             r"[$€£₽]\s?\d{1,3}(?:[,\s]\d{3})+(?:[.,]\d{2})?", r"[$€£₽]\s?\d+[.,]\d{2}\b", r"\b(?:usd|eur|gbp|rub)\s?\d",
             r"\d[\d\s]*(?:[.,]\d{1,2})?\s*\([а-я\s-]{3,80}\)\s*(?:руб|рубл)")    # "400 000 (четыреста тысяч) рублей"

# --- development signs -----------------------------------------------------------------------
_DEV_TITLE = _rx(r"^(техническое|частное техническое|тех\.?) задание", r"^тз\b", r"^technical (specification|design|requirements)",
                 r"^(функциональн\w+ |нефункциональн\w+ )?(спецификация|требования)\b", r"^(software )?requirements?( specification)?\b",
                 r"^specification\b", r"^readme\b", r"^протокол (встречи|совещания|созвона|синка)", r"^(meeting|call) (notes|minutes)",
                 r"^(описание|документация|спецификация) api",
                 r"^api\b(?!.*(?:[$€£₽]\s?\d|\d\s?(?:руб|₽|usd|eur|\$|€|£)))",   # "API integration — $4,000.00" — an invoice item
                 r"^архитектур", r"^user stor(y|ies)",
                 r"^(при[её]мочн\w+ критери|критерии при[её]мки|acceptance criteria)",     # acceptance criteria spec, not an act
                 r"^(changelog|release notes|roadmap|test plan|test cases|checklist|чек-лист|тест-план|план тестирования)\b",
                 r"^инструкция", r"^руководство", r"^регламент", r"^план (работ|проекта|спринта|разработки)",
                 r"^отч[её]т о (работе|тестировании|проделанной работе)", r"^бриф\b", r"^brief\b",
                 r"^постановка задачи", r"^описание (задачи|проекта|функционала)", r"^(design|product) (doc|document|spec)",
                 r"^(onboarding|getting started|installation|установка|usage|использование)\b",
                 # product texts: user agreement, policy — not deal papers
                 r"^(пользовательское|лицензионное) соглашение", r"^политика конфиденциальности",
                 r"^(terms of (service|use)|terms and conditions of use|privacy policy|end[- ]user licen[cs]e agreement|eula)\b")
_DEV_TERMS = [
    ("api", r"\bapi\b|эндпоинт|endpoint"), ("json", r"\bjson\b|\byaml\b|\bxml\b"),
    ("http/rest", r"\bhttps?\b|\brest(ful)?\b|graphql|websocket|\bget\s*/|\bpost\s*/|\bput\s*/|\bdelete\s*/|\bpatch\s*/"),
    ("backend/frontend", r"backend|бэкенд|frontend|фронтенд|full ?stack"),
    ("деплой", r"деплой|deploy|release\b|релиз\b|staging|prod(uction)?\b|продакшен"),
    ("репозиторий", r"репозитор|repositor|\bgit(hub|lab)?\b|pull request|merge request|\bcommit|коммит|\bbranch|ветк[аеи]\b"),
    ("docker/ci", r"docker|kubernetes|k8s|ci/cd|pipeline|пайплайн"),
    ("спринт/бэклог", r"\bsprint|спринт|backlog|бэклог|user stor|\bjira\b|\btrello|\bstory points?"),
    ("баги/тесты", r"\bbug|баг[аиов]?\b|тест-кейс|test case|unit[- ]test|юнит-тест|автотест|регресс|qa\b"),
    ("микросервисы", r"микросервис|microservice|монолит|архитектур"),
    ("база данных", r"база данных|database|\bsql\b|postgres|mysql|mongo|redis|clickhouse|миграци"),
    ("авторизация", r"авторизац|аутентификац|authenticat|authoriz|oauth|\bjwt\b|\btoken\b|токен"),
    ("ui/ux/макеты", r"\bui\b|\bux\b|макет|figma|wireframe|экран\w*\b|screen\b|кнопк|button|виджет|widget|вёрстк|верстк"),
    ("push/уведомления", r"push-?уведомлен|push notification|deep ?link|диплинк"),
    ("платформы", r"\bios\b|android|\bswift\b|kotlin|flutter|react( native)?|\bvue\b|angular|\bnode(js)?\b|python|\bjava\b|typescript|javascript|golang|\bphp\b|\bc#|\.net\b"),
    ("требования", r"функциональн\w+ требован|нефункциональн\w+ требован|сценари\w+ использования|use case|пользовательск\w+ сценари|acceptance criteria|критерии при[её]мки"),
    ("модули/компоненты", r"\bмодул[ьяеи]\b|module\b|компонент|component|библиотек|library|framework|фреймворк|\bsdk\b|плагин|plugin"),
    ("сервер/логи", r"\bсервер|server\b|логирован|\blogg?ing\b|\blogs?\b|мониторинг|monitoring|кэш|cache|очередь|queue|\bcron\b"),
    ("интеграция", r"интеграци|integration|webhook|вебхук|импорт|экспорт|синхронизац"),
    ("задача/фича", r"\bфич[аиу]\b|feature|\bтаск|\btask\b|\bepic\b|\bissue\b|тикет|ticket"),
]
_DEV_TERMS = [(label, re.compile(rx, re.I)) for label, rx in _DEV_TERMS]
_CODE = _rx(r"```", r"^\s*[{\[]\s*$", r"^\s*\"[\w-]+\"\s*:", r"\bcurl\b", r"localhost", r"\bdef \w+\(", r"\bfunction\s+\w+\(",
            r"\bclass \w+[:(]", r"^\s*(import|from) \w+", r"select .+ from |insert into|create table", r"\$\{", r"=>", r"\w+\.\w+\(\)", r"</?\w+>")
# a list item line: a marker or a number at the start
_LIST_ITEM = re.compile(r"^\s*(?:[-•·*–—>]+|\d{1,2}[.)])\s+\S")
# a table header: a cell ends with "№", the next is a column word ("Счёт на оплату №⇥Дата⇥Сумма"), the line has no
# "от" and no date; "Счет на оплату №⇥15⇥от⇥01.10.2025", "Счет №⇥А-45", "Счет на оплату №⇥ЦБ-15⇥от⇥…" — a title
_HEADER_ROW = re.compile(r"№[ ]*\t[ \t]*(?:дат|сумм|наименован|товар|работ|услуг|кол-?во|кол\.|количеств|цен[аы]|ед\.|"
                         r"единиц|ставк|ндс|всего|итого|стоимост|описан|артикул|код\b|период|статус|контрагент|"
                         r"тест|ожидаем|документ|номер|date|amount|description|item|qty|quantity|price|total)", re.I)
_HEADER_NOT = re.compile(r"\bот\b|\b\d{2}\.\d{2}\.\d{4}\b", re.I)
# an ASCII or Markdown table line is not a title, like a table header: ≥ 3 "|" ("| № | Работа | Сумма |"),
# 2 "|" not from the line start ("Счёт на оплату | int | 15") or a separator ("+---", "|---", "|___", "____|____");
# a one-cell frame "| СЧЕТ № 15 от 01.04.2025 |" is not a table line
_TABLE_ROW = re.compile(r"\|[^|\n]*\|[^|\n]*\||^(?![ \t]*\|)[^|\n]*\|[^|\n]*\||"
                        r"\+-{3}|\|[-_]{3}|^(?=[^\n]*\|)(?=[^\n]*[-_=]{3})[\s|+:=_-]+$")
# print form template placeholders — template markup only: {{…}}, {%…%}, ${…}, [[…]], <%…%>, <ФИО>;
# underscores ("ИНН ________", "«___» ______ 20__ г.", "Заказчик ______") are paper fields and blanks, not placeholders
_PLACEHOLDER = re.compile(r"\{\{[^{}\n]{1,80}\}\}|\{%[^\n]{1,80}?%\}|\$\{[^{}\n]{1,80}\}|<%[^\n]{1,80}?%>|"
                          r"\[\[\s*[^\W\d][^\[\]\n]{0,60}\]\]|<[А-ЯЁа-яё][^<>\n]{0,60}>")
# real values with which placeholders mean a partly filled paper, not a template: passport, full name
_PASSPORT_VALUE = re.compile(r"паспорт\w*\s*(?:серии|серия|сер\.)?\s*\d{2}\s?\d{2}\s*(?:№|номер)?\s*\d{6}\b", re.I)
_FULL_FIO = re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:ович|евич|овн|евн|ичн|ьич|мич)[а-яё]*\b")


def _norm(s: str) -> str:
    s = s.replace(" ", " ").replace("‑", "-").replace("–", "-").replace("—", "-")
    s = s.lower().replace("ё", "е")
    return s


def _letters(s: str) -> int:
    return sum(1 for ch in s if ch.isalnum())


def _lines(text: str) -> list[tuple[str, bool]]:
    """Non-empty lines without formatting: (line, whether it is not a title — a list item or a table header)."""
    out = []
    for raw in text.splitlines():
        item = bool(_LIST_ITEM.match(raw) or (_HEADER_ROW.search(raw) and not _HEADER_NOT.search(raw))
                    or _TABLE_ROW.search(raw))
        ln = re.sub(r"\s+", " ", raw).strip(" \t#*_-•·|:—–>«»\"'")
        if ln:
            out.append((ln, item))
    return out


def _title_candidates(lines: list[tuple[str, bool]], limit: int) -> list[str]:
    """The first limit meaningful lines that may hold a title: not a list item, not longer than
    TITLE_MAX_CHARS, not a sentence (a long line ending with a period). Short lines (a date,
    "г. Москва", a number), lines where more than half of the characters are digits (except a kind
    title: "АКТ № 9 от 30.09.2025"), and requisite lines ("БИК", "ИНН …", "Банк получателя") do not
    use up the window."""
    out = []
    for ln, item in lines:
        alnum = _letters(ln)
        if alnum < TITLE_MIN_LETTERS:
            continue
        if _REQ_LINE.match(ln) or (sum(ch.isdigit() for ch in ln) * 2 > alnum
                                   and not any(rx.search(ln) for rx in _TITLE.values())):
            continue
        if len(out) >= limit:
            break
        if item or len(ln) > TITLE_MAX_CHARS or (len(ln) > TITLE_SENTENCE_CHARS and ln.endswith(".")
                                                 and not _YEAR_ABBR.search(ln)):
            out.append("")                              # the slot is used, but does not count as a title
            continue
        out.append(ln)
    return out


def _title_hits(cands: list[str], regexes: dict, chatter: bool) -> dict:
    """{kind: title line} from the candidate lines."""
    out = {}
    for ln in cands:
        if not ln or (chatter and (_TITLE_CHATTER_VERB.search(ln)
                                   or (_TITLE_CHATTER.search(ln) and not _TITLE_NUMDATE.search(ln)))):
            continue
        tail = _TITLE_TAIL.search(ln) if chatter else None
        if tail and _TAIL_DESC.search(tail.group("tail")):
            continue
        lab = _TITLE_NO_LABEL.search(ln) if chatter else None
        if lab and _TITLE_NO_WORD.match(ln, lab.start()):
            continue
        for kind, rx in regexes.items():
            if kind not in out and rx.search(ln):
                out[kind] = ln
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"[\t;]", line)]


def _req_value(t: str) -> bool:
    """A requisite value in text t (normalized, with lines): INN/KPP/BIK/OGRN with digits, bank and
    correspondent accounts, an account number, IBAN, the pair "ИНН/КПП … N/N" or an "ИНН" column with a
    10/12-digit number in one of the next two table lines; in an xlsx view cells go one per line —
    "ИНН" on its own line and the number in one of the next three."""
    if _REQ_VALUE.search(" ".join(t.split())):
        return True
    rows = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for i, ln in enumerate(rows):
        if "\t" not in ln and ";" not in ln:
            if _INN_HEAD_CELL.fullmatch(ln) and any(_INN_CELL.fullmatch(c) for c in rows[i + 1:i + 4]):
                return True
            continue
        cols = [j for j, c in enumerate(_cells(ln)) if _INN_HEAD_CELL.fullmatch(c)]
        for nxt in rows[i + 1:i + 3]:
            cells = _cells(nxt)
            if any(j < len(cells) and _INN_CELL.fullmatch(cells[j]) for j in cols):
                return True
    return False


def _paper_strong(raw: str, t: str, flat: str, kind: str, lines: list[str], title: str | None,
                  new: bool = True) -> bool:
    """A strong paper sign (see _STRONG_SIGNS) or — with a kind title — two kind phrases outside the
    title line (without a title phrases do not count: a test case table "Счет на оплату … | Итого к оплате").
    Without a title amounts in table cells do not count either — test data ("15⇥01.10.2025⇥1 000,00 руб.").
    new=False — without the newer signs (a role with an organization, "Итого: 60 000", a "Сумма, руб." column)."""
    money = flat if title is not None else " ".join(ln for ln in t.splitlines() if "\t" not in ln.strip())
    money_lines = "\n".join(ln for ln in t.splitlines() if title is not None or "\t" not in ln.strip())
    if (_PAPER_STRONG.search(flat) or _INLICE.search(flat) or _MONEY.search(money) or _VAT.search(money)
            or _req_value(t) or _SWIFT_VALUE.search(raw) or _STRONG_SIGNS.search(t)
            or _ROLE_FIO.search(raw) or (new and (_ROLE_ORG.search(t) or _TOTAL_AMOUNT.search(money_lines)
            or (_CURRENCY_COLUMN.search(money_lines) and _THOUSANDS.search(money_lines))))):
        return True
    if title is None:
        return False
    body = " ".join(ln for ln in lines if ln != title)
    return sum(1 for _label, rx in _BODY[kind] if re.search(rx, body, re.I | re.M)) >= 2


def placeholders(text: str) -> int:
    """How many template placeholders are in the text (template markup; underscores do not count)."""
    return len(_PLACEHOLDER.findall(text or ""))


def _cap(x: float, cap: float) -> float:
    return x if x <= cap else cap


def known_kinds() -> list[str]:
    """Kinds from config.DOC_KINDS that have rules here."""
    return [k for k in DOC_KINDS if k in _TITLE]


def classify_detail(text: str, filename: str) -> Detail:
    """Detailed result: kind, score, borderline flag and the list of triggered signs."""
    raw = (text or "")[:HEAD_CHARS]
    t = _norm(raw)
    flagged = _lines(t)
    lines = [ln for ln, _item in flagged]
    flat = " ".join(lines)
    fname = _norm(filename or "")
    has_text = _letters(t) >= MIN_TEXT_LETTERS
    features: list[str] = []

    titles = _title_hits(_title_candidates(flagged, TITLE_LINES), _TITLE, chatter=True)
    dev_title = _title_hits(_title_candidates(flagged, DEV_TITLE_LINES), {"dev": _DEV_TITLE}, chatter=False).get("dev")

    kinds = known_kinds()
    kind_points: dict[str, float] = {}
    kind_feats: dict[str, list[str]] = {}
    has_cue: dict[str, bool] = {}
    kind_cues: dict[str, list[str]] = {}
    body_points: dict[str, float] = {}
    for kind in kinds:
        pts = 0.0
        feats = []
        cues = []
        if kind in titles:
            pts += _TITLE_PTS
            feats.append(f"title «{titles[kind][:40]}»")
            cues.append("title")
        if _FNAME[kind].search(fname):
            pts += _FNAME_PTS_DEV if dev_title else _FNAME_PTS
            feats.append("file name")
            cues.append("filename")
        has_cue[kind] = pts > 0
        kind_cues[kind] = cues
        body = 0.0
        for label, rx in _BODY[kind]:
            if re.search(rx, flat, re.I | re.M):
                body += 2
                feats.append(f"phrase «{label}»")
        body = _cap(body, _BODY_CAP)
        body_points[kind] = body
        pts += body
        kind_points[kind] = pts
        kind_feats[kind] = feats

    generic = 0.0
    gf = []
    req = [label for label, rx in _REQUISITES if re.search(rx, flat, re.I)]
    if req:
        generic += _cap(len(req), 3)
        gf.append("requisites: " + ", ".join(req))
    if _VAT.search(flat):
        generic += 1
        gf.append("VAT")
    if _TOTAL.search(t):
        generic += 1
        gf.append("total due")
    if _SIGN.search(flat):
        generic += 1
        gf.append("signatures / seal")
    if _ADDR.search(flat):
        generic += 1
        gf.append("party requisites")
    if _NUMDATE.search(flat):
        generic += 1
        gf.append("number and date")
    if _INLICE.search(flat):
        generic += 2
        gf.append("«represented by … acting under»")
    roles = [m.group(1) for m in _ROLES.finditer(flat)]
    if len(set(roles)) >= 2:
        generic += 1
        gf.append("party roles")
        if len(roles) >= 6:
            generic += 1
    if _MONEY.search(flat):
        generic += 1
        gf.append("amounts")
    generic = _cap(generic, _GENERIC_CAP)

    dev = 0.0
    df = []
    if dev_title:
        dev += 4
        df.append(f"title «{dev_title[:40]}»")
    terms = [label for label, rx in _DEV_TERMS if rx.search(flat)]
    if terms:
        dev += _cap(len(terms), _DEV_TERMS_CAP)
        df.append("terms: " + ", ".join(terms[:6]))
    code = len(_CODE.findall(t)) >= 2
    if code:
        dev += 2
        df.append("code")

    if not kinds:
        return Detail(None, 0.0, False, [], 0.0, {}, [])
    best = max(kinds, key=lambda k: kind_points[k])
    kp = kind_points[best]
    points = kp + generic - dev
    score = max(0.0, min(1.0, points / _SCALE))
    if kp < 2:
        kind = None
    else:
        kind = best
        words = " ".join(t.split())
        if not has_cue[best] and (body_points[best] >= 4 and dev == 0 and _BANK_ACCOUNT.search(words)
                                  and _TAX_VALUE.search(words)):
            # no title, but two kind phrases with a bank account and INN/BIK with values — a 1C export
            # without a title line (a payment details header), not a letter about a paper; with any
            # development sign (title, terms, code) — a DB schema, an API fixture — not
            gf.append("bank details with values")
        elif not has_cue[best]:
            # phrases in the body alone — not above borderline
            score = min(score, _NO_TITLE_MAX)
        elif has_text and kind_cues[best] == ["filename"] and body_points[best] == 0 and generic == 0:
            # a file name alone with text without paper signs — below the borderline band
            score = min(score, _FNAME_ONLY_MAX)
        hold = _DEV_MARKERS.search(flat)
        subject = _DEV_SUBJECT.search(flat)
        dev_marker = hold or subject
        title = titles.get(best)
        if ("filename" not in kind_cues[best] and (hold or code or subject)
                and not _paper_strong(raw, t, flat, best, lines, title)):
            if ((not hold or _ITEM_AMOUNT.search(t)) and not code and title is not None
                    and _TITLE_NUMBERED.search(title) and _TITLE_DATED.search(title)):
                # a title "Счёт № N от ДАТА", and the text has only an IT studio's work subject ("Доработка API и
                # backend: 60 000") or a ticket word in a work item with an amount ("Bug fixing: 1,200.00") —
                # a paper, not below borderline
                score = max(score, BORDERLINE_THRESHOLD)
            else:
                # the hint is a title alone (or none), and the text has a ticket, a bug, README, test cases, API
                # documentation or code and not a single strong paper sign: the kind is not set
                score = min(score, _FNAME_ONLY_MAX)
                df.append("title only with " + (f"«{dev_marker.group(0)}»" if dev_marker else "code"))
        elif ("filename" not in kind_cues[best] and (hold or code or subject)
              and not _paper_strong(raw, t, flat, best, lines, title, new=False)):
            # of the strong signs — only a role with an organization, "Итого: 60 000" or a "Сумма, руб." column: an
            # invoice example in a spec, README or test case has them too — the kind is set, but not above borderline
            score = min(score, _NO_TITLE_MAX)
            df.append("newer paper signs with " + (f"«{dev_marker.group(0)}»" if dev_marker else "code"))
        if ("title" not in kind_cues[best] and "filename" in kind_cues[best] and (dev_marker or code)
                and not (body_points[best] >= 4 and _paper_strong(raw, t, flat, best, lines, None))):
            # the hint is a file name alone ("acceptance.txt"), and the text has development terms or code — not
            # above borderline: acceptance criteria, not an act; except a paper with ≥ 4 kind points in the text and
            # a strong sign (act act_15.docx with "доработка API" — confident)
            score = min(score, _NO_TITLE_MAX)
            df.append("file name only with " + (f"«{dev_marker.group(0)}»" if dev_marker else "code"))
        n_ph = placeholders(raw)
        if n_ph >= 2 and not (_req_value(t) or _SWIFT_VALUE.search(raw) or _MONEY.search(flat)
                              or _PASSPORT_VALUE.search(flat) or _FULL_FIO.search(raw)):
            # a print form template without real values (requisites, amounts, passport, full name) — not a paper,
            # even with a name hint
            score = min(score, _FNAME_ONLY_MAX)
            df.append(f"template ({n_ph} placeholders)")
    if kind is not None:
        features.extend(kind_feats[kind])
    features.extend(gf)
    features.extend("against: " + f for f in df)
    borderline = kind is not None and BORDERLINE_THRESHOLD <= score < DELETE_THRESHOLD
    if kind is not None and score < BORDERLINE_THRESHOLD:
        kind = None
        borderline = False
    cues = kind_cues[kind] if kind is not None else []
    return Detail(kind, round(score, 3), borderline, features, round(points, 2), kind_points, cues)


def classify(text: str, filename: str) -> tuple[str | None, float, bool]:
    """→ (kind from config.DOC_KINDS or None, score 0..1, whether the case is borderline)."""
    d = classify_detail(text, filename)
    return d.kind, d.score, d.borderline
