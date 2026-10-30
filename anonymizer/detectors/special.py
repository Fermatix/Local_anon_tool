"""Special-category (GDPR art. 9 / Russian 152-FZ special categories) — OPT-IN, conservative.

DISABLED by default (``config.detectors.special = False`` — an owner decision):
special-category data is the highest-risk class, but a naive sweep of health /
ethnicity / religion / criminal vocabulary shreds ordinary text (a support
ticket that merely says «взял больничный», an HR template that lists
«национальность» as a form field, «справка о несудимости» as onboarding
boilerplate). None of those is personal data about anyone.

So this module masks ONLY a genuine *tie* between a special-category FACT and a
SPECIFIC identifiable data subject expressed **in the same short clause**:

  * ``context`` tier — a person token already placed by the names pass
    (``PERSON_NAME_123`` / ``PERSON_123`` / ``USER_123``) next to the fact; a
    strong, identifiable tie, so it fires under the production gate.
  * ``low`` tier — a clause-initial 1st/2nd/3rd-person pronoun self-assertion
    (``я`` / ``у меня`` / ``мне`` / ``у него`` …): the speaker asserting the
    fact, but with no independent identifier in the clause; a weak tie, off by
    default.

The subject/pronoun is KEPT; only the fact is redacted → ``P.HEALTH`` (health,
disability, pregnancy, addiction, mental-health, HIV, sexual orientation) or
``P.ETHNICITY`` (ethnicity self-ID, religion, political affiliation, criminal
record, and an ethnic/orientation slur directed at a named person). The subject
is matched inline (it is part of the pattern), so a fact — including a slur,
which on its own is mere profanity, not personal data — can never fire without
an adjacent data subject.
Everything here is ``context`` (an anchored person↔fact tie) or ``low`` (bare
self-assertion sweeps) — never ``high`` — so even with the detector switched on
the production gate ("high","context") only fires the anchored rules, and the
weakest ones stay opt-in behind "low".

FP-critical invariant: no person/pronoun on the left within ~40 chars ⇒ nothing
fires. Generic vocabulary in isolation is left untouched by construction.
"""
from __future__ import annotations

import re

from ..engine import placeholders as P
from ..engine.boundary import CL, CR
from ..engine.rule import Rule, keep_line_breaks

# --- the data subject ---------------------------------------------------------
# Two strengths, mapped to two confidence tiers:
#
#   * A PLACED PERSON TOKEN (PERSON_NAME_/PERSON_/USER_) already anchored by the
#     names pass ⇒ a strong, identifiable tie ⇒ the "context" rules below.
#   * A bare 1st/2nd/3rd-person PRONOUN ("я", "у меня", "у него") ⇒ a weak tie
#     (no independent identifier in the clause) ⇒ the "low" self rules.
#
# The subject is the kept left group of every rule — the fact can never fire
# without a subject, because the subject is part of the match itself (so no
# separate requires_context is needed; the person token lives INSIDE the match,
# not to its left, and a left-window guard would wrongly reject a clause that
# begins with the subject, e.g. "PERSON_NAME_10 беременна"). Bare pronouns are
# Cyrillic-bounded (CL/CR) so `он` can't match inside `она`/`оно` and `я` can't
# match the tail of `имя`.
_PERSON_TOKEN = r"(?:PERSON_NAME_\d+|PERSON_\d+|USER_\d+)"

_PRON = (
    CL + r"(?:я|ты|мы|вы|он|она|оно|они|"
    r"мне|меня|тебе|тебя|ему|его|её|ей|нам|нас|вам|вас)" + CR
)
_PREP_PRON = (
    r"(?:у|о|об|про|за|на|по)\s+" + CL +
    r"(?:мен[яе]|теб[яе]|нег[оео]|не[её]|нас|вас|себя)" + CR
)
# Any pronoun subject, clause-initial (so it's the grammatical subject, not a
# mid-sentence object) — feeds the weak "low" self rules. In JSONL the in-string
# line break is the TWO characters ``\`` + ``n`` (see boundary.L), so the
# escaped ``\n`` is a clause start too; it is consumed into the kept subject
# group, so the redaction can never swallow it.
_SELF = (
    r"(?:^|[.;!?\n]\s*|\\n\s*|,\s*)(?:" + _PREP_PRON + r"|" + _PRON + r")"
)

# --- separator between subject and fact: same clause (no sentence break), and
#     it is KEPT verbatim so the redaction token never fuses onto the subject.
#     JSONL-aware: quotes/braces/brackets, sentence-final !/? and the escaped
#     two-character ``\n`` (see boundary.L) are boundaries too, so a person
#     token in one JSON field can never tie to vocabulary starting the next
#     field or the next escaped line.
_GAP = r'((?:(?!\\n)[^.\n"{}\[\]!?]){0,40})'

# Health / disability / pregnancy / addiction / mental-health / HIV / sexual
# orientation. Stems are chosen to be discriminating (not bare common words);
# each requires the subject tie. Bare stems (рак, ВИЧ, СПИД, птср) carry an
# explicit CR so they can never match a prefix of a longer word («ракету»,
# «спидометр»); delimiter checks are lookaheads, never consumed, so the token
# cannot weld onto the next word. Sexual orientation is a GDPR art. 9 / 152-FZ
# special category, so it lives with the other health-adjacent facts under
# ``P.HEALTH``.
_HEALTH_FACT = (
    r"(?:беремен\w+|"                                    # pregnancy
    r"инвалид\w*|инвалидност\w+|нетрудоспособн\w+|колясочник\w*|"  # disability
    r"диабет\w*|астм\w+|эпилепс\w+|онколог\w+|"          # chronic disease
    r"рак(?:а|ом|у|е)?" + CR + r"|опухол\w+|гепатит\w*|туберкул[её]з\w*|"
    r"псориаз\w*|рассеянн\w+\s+склероз\w*|"
    r"шизофрен\w+|деменц\w+|аутиз\w*|аутист\w*|"
    r"депресс\w+|биполярн\w+|психическ\w+\s+расстройств\w*|"  # mental health
    r"псих(?:ическ)?и\w*\s+заболеван\w*|расстройств\w+\s+психик\w*|"
    r"суицид\w*|самоубийств\w+|анорекси\w+|булими\w+|птср" + CR + r"|пограничн\w+\s+расстройств\w*|"
    r"ВИЧ" + CR + r"|СПИД" + CR + r"|вич-инфиц\w+|вич-полож\w+|спид-инфиц\w+|"  # HIV/AIDS
    r"наркоман\w*|наркозависим\w*|наркопотреб\w*|"       # addiction
    r"алкоголизм\w*|алкоголи[кч]\w*|алкозависим\w*|токсикоман\w*|"
    r"зависимост\w+\s+от\s+(?:наркотик\w+|алкогол\w+|психоактивн\w+)|"
    r"употребля\w+\s+(?:наркотик\w+|героин\w*|кокаин\w*)|"
    r"аборт\w*|бесплоди\w+|бесплодн\w+|выкидыш\w*|эко(?=\s|$)|"  # reproductive health
    r"гомосексуал\w*|гей(?=[\s,.]|$)|лесбиян\w+|бисексуал\w*|"  # sexual orientation
    r"транссексуал\w*|трансгендер\w*|небинарн\w+|асексуал\w*|"
    r"нетрадиционн\w+\s+ориентаци\w+|секс\w*\s+ориентаци\w+|"
    r"химиотерап\w+|диализ\w*|трансплантац\w+|пересадк\w+\s+(?:почк\w+|печен\w+|сердц\w+)|"
    r"диагноз\w*\s+\S+|"                                  # "диагноз <X>"
    r"наблюда\w+\s+(?:у|в)\s+(?:психиатр\w*|нарколог\w*|психоневролог\w*|"  # under care
    r"психотерапевт\w*|психоневрологи\w*|наркологи\w*)|"
    r"состои\w+\s+на\s+учёте\s+(?:у\s+)?(?:психиатр\w*|нарколог\w*|"
    r"в\s+(?:психоневролог\w*|наркологи\w*)))"
)

# Ethnicity / religion / political belief / criminal record. Ethnonym stems
# carry (?!...) lookaheads so their open \w* tails cannot swallow country/place
# derivations (Казахстан, Узбекистан, Таджикистан, Киргизия, Ингушетия,
# Башкирия, Бурятия) — a place name near a person token is travel prose, not a
# special-category fact. Дагестан/Азербайджан ARE the place stems, so those two
# are restricted to the person-noun/adjective paradigm instead of \w+.
_ETHNICITY_FACT = (
    r"(?:по\s+национальност\w+\s+\S+|"                    # "по национальности X"
    r"(?:еврей\w*|евре[йя]к\w*|армян\w+|азербайджан(?:ец|к\w+|ц\w+|ск\w+)|"  # ethnicity assertions
    r"чеченец|чеченк\w+|чечен\w+|дагестан(?:ец|к\w+|ц\w+|ск\w+)|цыган\w+|ром(?:а|ы)\b|"
    r"таджик(?!истан)\w*|узбек(?!истан)\w*|киргиз(?!и)\w*|казах(?!стан)\w*|грузин\w*|"
    r"осетин\w*|ингуш(?!ети)\w*|башкир(?!и)\w*|татар(?:ин|к\w+|ы)?" + CR + r"|бурят(?!и)\w*|"
    r"украин\w+\s+национальност\w+)|"
    r"(?:мусульман\w+|мусульманк\w+|христиан\w+|православн\w+|католик\w*|католичк\w*|"  # religion
    r"протестант\w*|иуде[йя]\w*|иудаизм\w*|буддист\w*|атеист\w*|агностик\w*|"
    r"баптист\w*|пятидесятник\w*|адвентист\w*|"
    r"верующ\w+\s+(?:мусульман\w+|христиан\w+|иуде[йя]\w*)|"
    r"свидетел\w+\s+иеговы|исповеду\w+\s+\S+)|"
    r"(?:судим\w+|осуждён\w*|осужден\w*|отбыва\w*\s+(?:срок|наказан\w+)|"  # criminal
    r"был\w*\s+в\s+(?:тюрьме|заключени\w+|колони\w+|СИЗО|местах\s+лишени\w+)|"
    r"сидел\w*\s+в\s+тюрьм\w+|условн\w+\s+срок\w*|под\s+следстви\w+|"
    r"привлека\w+\s+к\s+(?:уголовн\w+|ответственност\w+))|"
    r"(?:член\w*\s+(?:парти\w+|КПРФ|ЛДПР|Единой\s+России)|"  # political belief
    r"состои\w+\s+в\s+(?:парти\w+|КПРФ|ЛДПР)|"
    r"сторонник\w*\s+\S+|голосу\w+\s+за\s+\S+|"
    r"поддержива\w+\s+(?:партию|оппозиц\w+|власть|навальн\w+)))"
)

# Ethnic / orientation SLURS. A slur *directed at a named person* reveals a
# special category (perceived ethnicity / orientation) about that data subject —
# so it is masked, but ONLY when tied to a subject, exactly like every other rule
# (a slur floating in isolation is profanity, not personal data, and is left).
# Deliberately a short, unambiguous stem list (each token is a slur in ~all
# contexts, so no ordinary word collides), stored obfuscated-free but discrete.
_SLUR_FACT = (
    r"(?:жид\w*|хач\w*|чурк\w+|чурбан\w*|нигер\w*|негритос\w*|"      # ethnic slurs
    r"хох(?:о)?л\w*|москал\w+|азер\b|азер\w+|узкоглаз\w+|черно(?:жоп|мазр)\w*|"
    r"пидор\w*|пидар\w*|педик\w*|гомик\w*|пед\w*раст\w*)"            # orientation slurs
)


# Groups (identical across every rule): (1) subject, (2) kept separator, (3) the
# special-category fact = the only thing redacted. Keeping 1+2 verbatim means the
# token can never weld onto the subject or the intervening word.
# A multi-word fact ("сидел в\nтюрьме") can be split by a line break — the token keeps
# the fact's line breaks (keep_line_breaks), otherwise the line-break safeguard skipped the match.
def _keep_subject_health(m: re.Match) -> str:
    return m.group(1) + m.group(2) + keep_line_breaks(P.HEALTH, m.group(3))


def _keep_subject_ethnicity(m: re.Match) -> str:
    return m.group(1) + m.group(2) + keep_line_breaks(P.ETHNICITY, m.group(3))


RULES: list[Rule] = [
    # --- HEALTH / disability / pregnancy / addiction --------------------------
    # context tier: fact tied to a PLACED person token in the same clause. The
    # fact is CL/CR-bounded (like the slur rules) so a stem can never fire
    # mid-word («ракету», «спидометр») or weld the token onto a word remainder.
    Rule("special_health_person",
         re.compile(r"(?i)(" + _PERSON_TOKEN + r")" + _GAP + CL + r"(" + _HEALTH_FACT + r")" + CR),
         mask=_keep_subject_health, value_group=3, confidence="context"),

    # low tier: bare pronoun self-assertion only ("у меня диабет") — opt-in.
    Rule("special_health_self",
         re.compile(r"(?i)(" + _SELF + r")" + _GAP + CL + r"(" + _HEALTH_FACT + r")" + CR),
         mask=_keep_subject_health, value_group=3, confidence="low"),

    # --- ETHNICITY / religion / political belief / criminal record -----------
    Rule("special_ethnicity_person",
         re.compile(r"(?i)(" + _PERSON_TOKEN + r")" + _GAP + CL + r"(" + _ETHNICITY_FACT + r")" + CR),
         mask=_keep_subject_ethnicity, value_group=3, confidence="context"),

    Rule("special_ethnicity_self",
         re.compile(r"(?i)(" + _SELF + r")" + _GAP + CL + r"(" + _ETHNICITY_FACT + r")" + CR),
         mask=_keep_subject_ethnicity, value_group=3, confidence="low"),

    # --- ethnic / orientation SLUR tied to a person --------------------------
    # A slur reveals a (perceived) special category about a data subject, but
    # only counts when directed at one — so BOTH orderings require the person
    # token / pronoun adjacent, and the slur alone (profanity) never fires.
    # subject on the LEFT ("PERSON_NAME_1 — жид"); CL so `жид` can't fire
    # inside «поджидал»:
    Rule("special_slur_person",
         re.compile(r"(?i)(" + _PERSON_TOKEN + r")" + _GAP + CL + r"(" + _SLUR_FACT + r")" + CR),
         mask=_keep_subject_ethnicity, value_group=3, confidence="context"),
    # subject on the RIGHT ("этот чурка PERSON_NAME_1"): redact the slur (group 1),
    # keep the gap + subject verbatim so the token can't fuse onto the person.
    Rule("special_slur_person_pre",
         re.compile(CL + r"(" + _SLUR_FACT + r")" + CR + _GAP + r"(" + _PERSON_TOKEN + r")"),
         mask=lambda m: keep_line_breaks(P.ETHNICITY, m.group(1)) + m.group(2) + m.group(3),
         value_group=1, confidence="context"),
    # bare self-assertion ("я хохол") — opt-in low tier.
    Rule("special_slur_self",
         re.compile(r"(?i)(" + _SELF + r")" + _GAP + CL + r"(" + _SLUR_FACT + r")" + CR),
         mask=_keep_subject_ethnicity, value_group=3, confidence="low"),
]
