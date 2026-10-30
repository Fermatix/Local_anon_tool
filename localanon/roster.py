"""Roster of people from export metadata and the rules that mask them.

Sources:
  * Telegram (result.json and chat exports) — the engine's entity registry
    (``anonymizer.structured.build_registry``: ``from``/``from_id``, ``actor``, ``members``,
    ``forwarded_from``, ``contacts``, ``personal_information``, mentions) → ``Registry.roster()``:
    canonical full forms "Given Surname" (rank 0–1) and handles;
  * Discord (DiscordChatExporter: ``author.name``, ``author.nickname``, ``author.id``) and Slack
    (``users.json``: ``real_name``, ``profile.display_name``, ``profile.real_name``, ``name``) —
    a JSON walk without an engine profile.

Forms (as the engine builds them for known people — detectors/names + the main pipeline pack):
the full form in any order and case, with an optional patronymic ("Иван Сергеевич Петров",
"Петров Иван Сергеевич"), the surname in all cases (Capitalized or CAPS), initials with surname.
The token is the same one the engine gives the pair "Given Surname" in text: ``mint.pname`` by the
nominative surname (``names._surname_nominative`` + ``_seed_person``; Latin — ``_canon_lat``).
A given name alone, without a surname, is not masked as a literal ("Саша" in text is not
necessarily this person). Handles → ``HANDLE_…`` by the lowercase handle (like the engine's
``handle_at`` and the Telegram mention field), with and without "@".

Place in the engine chain (``rule_groups``): pairs and initials go BEFORE the name detector
(``Engine(before_names=…)``): these are exact metadata literals, they take the whole person before
``bare_surname_dict``/``dict_first_name`` split "Соколов Алексей" into two tokens. Handles and
single surnames go in the pack slot (after the detectors), handles before surnames: a Latin
surname inside a handle ("o.smirnov") does not cut it. A surname that is also a common noun
(registry rank 2: "Сокол", "Мир"), alone without a given name, is masked only Capitalized/CAPS
and NOT at the start of a sentence ("Сокол — птица", "Мир, труд, май") — like the engine, which
does not put rank 2 in the roster; the gate judges the same way. A file or directory name (each
path part) does not count as a sentence start: a file "Сокол.txt" with the person "Роза Сокол" in
the roster is a strong signal; renaming goes through ``path_context`` (Engine.mask_name), the
rule and the gate (path check) call the same predicate ``homonym_masked``.
"""
from __future__ import annotations

import contextlib
import contextvars
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from anonymizer.detectors.names import _canon_lat, _seed_person, _surname_nominative
from anonymizer.engine import mint
from anonymizer.engine import gazetteer as gaz
from anonymizer.engine.boundary import CL, CR, L, R, HWS_CLASS, alt_longest_first, is_own_token
from anonymizer.engine.rule import Rule
from anonymizer.engine.vocab import COMMON_EN, TECH_TERMS, TRANSLIT
from anonymizer.structured.entities import canonical, form_rank, person_parts

from . import rxdefer

_NICK = re.compile(r"[A-Za-z][A-Za-z0-9_.]{2,31}")
_NICK_STOP = frozenset(COMMON_EN) | frozenset(TECH_TERMS) | {
    "everyone", "here", "channel", "all", "bot", "null", "none", "true", "false", "deleted",
}
_HAS_CYR = re.compile(r"[А-Яа-яЁё]")
_SURN_SK = ("ский", "цкий")
_SURN_POSS = ("ов", "ев", "ёв", "ин", "ын")
_SURN_FIXED = ("енко", "ко", "ых", "их", "дзе", "швили", "ия", "о", "е", "и", "у", "ю", "э")
# endings by declension type (detectors/names: _POSSESSIVE_TAILS, _SK_FORMS, _NOUN_TAILS)
_POSS_TAILS = ("", "а", "у", "ым", "е", "ой", "ою", "ы", "ых", "ыми")
_SK_TAILS = ("ий", "ого", "ому", "им", "ом", "ая", "ой", "ую", "ие", "ых", "ими")
_NOUN_TAILS = ("", "а", "у", "ом", "е", "ы", "и", "ем", "я", "ю")
_A_TAILS = ("а", "и", "е", "у", "ой", "ою", "ы")
_YA_TAILS = ("я", "и", "е", "ю", "ей")
_SOFT_TAILS = ("ь", "я", "ю", "ем", "е", "и")
_Y_TAILS = ("й", "я", "ю", "ем", "е", "и")
_INITIALS = r"(?:[А-ЯЁ]\.[ \u00a0]?){1,2}"
_SP = r"[ \u00a0\u202f]+"
# a patronymic in any case (entities._PATRONYMIC + a case ending): Сергеевич(а|у|ем|е),
# Сергеевн(а|ы|е|у|ой|ою)
_PATR = (r"[А-ЯЁ][а-яё]{2,}(?:(?:ович|евич|ьич|ич)(?:а|у|е|ем|и)?"
         r"|(?:овн|евн|ичн|иничн)(?:а|у|е|ой|ою|ы|и))")
# sentence start: before the word only spaces after the text start, a sentence end, a newline,
# an opening quote/bracket or a list marker
_PHRASE_END = ".!?\n\r\t«\"'„“”([{<>—–-|*#•·"

# Discord: author object; Slack: users.json entry
_PERSON_KEYS = ("real_name", "display_name", "nickname", "global_name", "full_name")
_HANDLE_KEYS = ("name", "username")


@dataclass
class Person:
    canon: str                     # "Given Surname"
    given: str
    surname: str
    latin: bool
    homonym: bool = False          # the surname is a common noun (entities.form_rank == 2)

    @property
    def seed(self) -> str:
        if self.latin:
            return _canon_lat(self.surname)
        return _seed_person(_surname_nominative(self.surname))

    @property
    def token(self) -> str:
        return mint.pname(self.seed)


@dataclass
class Roster:
    persons: list = field(default_factory=list)      # Person
    handles: list = field(default_factory=list)      # lowercase handles, without "@"
    sources: dict = field(default_factory=dict)      # counters by source (no values)

    def add_person(self, name) -> bool:
        """Full form "Given Surname" → a roster person. Rank 2 (the surname is also a common noun:
        "Сокол", "Бондарь") is taken too: the rules mask only the Capitalized/CAPS form, lowercase
        "сокол" is left alone (in the engine rank 2 is judged by the regular path, absent here)."""
        c = canonical(name)
        if not c or any(p.canon == c for p in self.persons):
            return False
        given, surname = person_parts(c)
        self.persons.append(Person(c, given, surname, not _HAS_CYR.search(surname),
                                   homonym=form_rank(c) >= 2))
        return True

    def add_handle(self, nick) -> bool:
        if not isinstance(nick, str):
            return False
        raw = nick.strip().lstrip("@")
        if is_own_token(raw, part=True):          # already our token (repeated run)
            return False
        nick = raw.lower()
        if not _NICK.fullmatch(nick) or nick in _NICK_STOP or nick.endswith("bot"):
            return False
        if nick not in self.handles:
            self.handles.append(nick)
        return True

    def summary(self) -> dict:
        return {"persons": len(self.persons), "handles": len(self.handles), **self.sources}


# --------------------------------------------------------------------------- collection
def collect(root: Path, files, workers: int = 1) -> Roster:
    """Roster from the JSON files of the tree: Telegram — by the engine registry (the files are collected
    over ``workers`` processes, structured.build_registry), Discord/Slack — by a walk."""
    from anonymizer import structured
    root = Path(root)
    roster = Roster()
    tg, other = [], []
    for p in files:
        p = Path(p)
        if p.suffix.lower() not in (".json", ".jsonl", ".ndjson"):
            continue
        rel = p.relative_to(root).as_posix()
        if structured.detect(str(p), rel) is not None:
            tg.append((str(p), rel))
        else:
            other.append(p)
    if tg:
        reg = structured.build_registry(tg, workers)
        n_p = n_h = 0
        for lit, klass, _rel in reg.roster():
            if klass == "person":
                n_p += roster.add_person(lit)
            elif klass == "handle":
                n_h += roster.add_handle(lit)
        # rank 2 (homonym surname) is not in Registry.roster() — take the entity canon directly
        reg._derive()
        for canon in sorted(c for c in reg._canon.values() if c):
            n_p += roster.add_person(canon)
        for name in sorted(reg.loose):
            n_p += roster.add_person(name)
        roster.sources["telegram_persons"] = n_p
        roster.sources["telegram_handles"] = n_h
    for p in other:
        _collect_generic(p, roster)
    return roster


def _load_json(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.lstrip("\ufeff")
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except ValueError:
            return None
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _collect_generic(path: Path, roster: Roster) -> None:
    doc = _load_json(path)
    if doc is None:
        return
    n_p = n_h = 0
    for obj in _identity_objects(doc):
        for k in _PERSON_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                if roster.add_person(v):
                    n_p += 1
                elif roster.add_handle(v):
                    n_h += 1
        prof = obj.get("profile")
        if isinstance(prof, dict):
            for k in ("real_name", "display_name"):
                v = prof.get(k)
                if isinstance(v, str) and v.strip():
                    if roster.add_person(v):
                        n_p += 1
                    elif roster.add_handle(v):
                        n_h += 1
        for k in _HANDLE_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                if roster.add_person(v):
                    n_p += 1
                elif roster.add_handle(v):
                    n_h += 1
    if n_p or n_h:
        roster.sources["json_persons"] = roster.sources.get("json_persons", 0) + n_p
        roster.sources["json_handles"] = roster.sources.get("json_handles", 0) + n_h


def _identity_objects(doc):
    """Identity objects: Discord ``author``/``mentions[]``, Slack ``users.json`` (a list of
    entries with ``name`` and ``profile``/``real_name``), ``user_profile`` in Slack messages."""
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in ("author", "user_profile"):
                v = node.get(key)
                if isinstance(v, dict):
                    yield v
            ments = node.get("mentions")
            if isinstance(ments, list):
                for m in ments:
                    if isinstance(m, dict):
                        yield m
            if "profile" in node and isinstance(node.get("profile"), dict) \
                    and ("name" in node or "real_name" in node):
                yield node
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))


# --------------------------------------------------------------------------- forms
def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _yo_variants(s: str) -> list:
    out = {s}
    if "ё" in s:
        out.add(s.replace("ё", "е"))
    return sorted(out)


def surname_forms(nominative: str) -> list:
    """Case forms of a surname (lowercase, with ё and е) by the declension type of the nominative."""
    nom = nominative.lower()
    if "-" in nom:
        parts = [surname_forms(p) for p in nom.split("-")]
        return ["-".join(c) for c in itertools.product(*parts)][:64]
    if nom.endswith(_SURN_SK):
        forms = [nom[:-2] + t for t in _SK_TAILS]
    elif nom.endswith(_SURN_POSS) and not nom.endswith(("ых", "их")):
        forms = [nom + t for t in _POSS_TAILS]
    elif nom.endswith(_SURN_FIXED):
        forms = [nom]
    elif nom.endswith("а"):
        forms = [nom[:-1] + t for t in _A_TAILS]
    elif nom.endswith("я"):
        forms = [nom[:-1] + t for t in _YA_TAILS]
    elif nom.endswith("ь"):
        forms = [nom[:-1] + t for t in _SOFT_TAILS]
    elif nom.endswith("й"):
        forms = [nom[:-1] + t for t in _Y_TAILS]
    else:
        forms = [nom + t for t in _NOUN_TAILS]
    out = []
    for f in forms:
        for v in _yo_variants(f):
            if v not in out:
                out.append(v)
    return out


def given_forms(given: str) -> list:
    """Case forms of a given name by its declension class (gazetteer._CLASS_TAILS)."""
    stem = gaz._stem(given)
    tails = gaz._CLASS_TAILS.get(gaz._decl_class(given), ("",))
    out = []
    for t in tails:
        for v in _yo_variants((stem + t).lower()):
            if v not in out:
                out.append(v)
    return out


def _cased(forms) -> list:
    """Capitalized and CAPS — the surname case in text (like the detectors/names slots)."""
    out = []
    for f in forms:
        for v in (_cap(f), f.upper()):
            if v not in out:
                out.append(v)
    return out


def person_literals(p: Person) -> dict:
    """A person's literals for the rules and the check: {"surname": [...], "given": [...]}."""
    if p.latin:
        return {"surname": [p.surname], "given": [p.given]}
    return {"surname": surname_forms(_surname_nominative(p.surname)),
            "given": given_forms(p.given)}


def _guard(m) -> bool:
    return is_own_token(m.group(0), part=True)


def phrase_start(text: str, pos: int) -> bool:
    """The word at ``pos`` starts a sentence: on the left only spaces up to the text start, a
    sentence end, a newline, an opening quote/bracket or a list marker. A predicate shared by the
    homonym surname rule and the check gate."""
    i = pos
    while i > 0 and text[i - 1] in " \u00a0\u202f\t":
        i -= 1
    return i == 0 or text[i - 1] in _PHRASE_END


_IN_PATH = contextvars.ContextVar("localanon_in_path", default=False)


@contextlib.contextmanager
def path_context():
    """The masked text is a file or directory name (Engine.mask_name): context-dependent rules
    (homonym surname, lowercase contractor name) treat it as a path."""
    tok = _IN_PATH.set(True)
    try:
        yield
    finally:
        _IN_PATH.reset(tok)


def in_path() -> bool:
    return _IN_PATH.get()


def homonym_masked(text: str, pos: int, *, path: bool | None = None) -> bool:
    """A single homonym surname at ``pos`` is to be masked: in a path (a file or directory name is
    not a sentence start, a strong signal) always, in text — not at a sentence start. One
    predicate for the rule (``path`` None — by path_context) and the gate (``path`` — path check)."""
    if path is None:
        path = in_path()
    return path or not phrase_start(text, pos)


def _homonym_guard(m) -> bool:
    """A single homonym surname: already a token or not to be masked (homonym_masked) — leave it."""
    return is_own_token(m.group(1), part=True) or not homonym_masked(m.string, m.start(1))


def person_rules(p: Person, prefix: str = "roster") -> list:
    """Rules of one person → their token: the pair in any order and case (with an optional
    patronymic), initials + surname, surname alone (Capitalized/CAPS). List order: full form
    rules (pair, initials) first, the single surname rule last (rule_groups puts them into the
    chain slots). The patterns are compiled on first use or by the engine in a batch over its workers
    (rxdefer: a rule built and dropped is not compiled)."""
    tok = p.token
    lits = person_literals(p)
    rules = []
    tag = re.sub(r"[^a-z0-9]+", "-", "".join(TRANSLIT.get(ch, ch) for ch in p.seed.lower())) or "p"
    if p.latin:
        surn = alt_longest_first([p.surname])
        given = alt_longest_first([p.given])
        rules.append(Rule(f"{prefix}:pair:{tag}",
                          rxdefer.compile(L + r"(?:" + given + _SP + surn + r"|" + surn + _SP + given
                                          + r")" + R, re.IGNORECASE),
                          mask=lambda m, t=tok: t, guard=_guard, confidence="high"))
        # not inside a handle/email: "o.smirnov", "smirnov.o@…" — no "." on the left, no "@" on the right
        rules.append(Rule(f"{prefix}:surname:{tag}",
                          rxdefer.compile(L + r"(?<![.@-])(" + surn + r")" + R + r"(?![\w.-]*@)",
                                          re.IGNORECASE),
                          mask=lambda m, t=tok: t, value_group=1, guard=_guard,
                          confidence="high"))
        return rules
    surn = alt_longest_first(_cased(lits["surname"]))
    given = alt_longest_first(_cased(lits["given"]))
    patr = r"(?:" + _SP + _PATR + r")?"
    rules.append(Rule(f"{prefix}:pair:{tag}",
                      rxdefer.compile(CL + r"(?:(?:" + given + r")" + patr + _SP + r"(?:" + surn
                                      + r")|(?:" + surn + r")" + _SP + r"(?:" + given + r")" + patr
                                      + r")" + CR),
                      mask=lambda m, t=tok: t, guard=_guard, confidence="high"))
    rules.append(Rule(f"{prefix}:initials:{tag}",
                      rxdefer.compile(CL + r"(?:" + _INITIALS + r"(?:" + surn + r")|(?:" + surn + r")"
                                      + r"[ \u00a0]?" + _INITIALS + r")" + CR),
                      mask=lambda m, t=tok: t, guard=_guard, confidence="high"))
    rules.append(Rule(f"{prefix}:surname:{tag}",
                      rxdefer.compile(CL + r"(" + surn + r")" + CR),
                      mask=lambda m, t=tok: t, value_group=1,
                      guard=_homonym_guard if p.homonym else _guard, confidence="high"))
    return rules


def handle_rule(nick: str, prefix: str = "roster") -> Rule:
    """A handle from metadata → HANDLE token by the lowercase handle (the same seed as the engine's
    handle_at and the Telegram mention field). A handle with "_", a digit or a dot is masked
    without "@" too (like the engine roster); a letters-only one ("market") — only with "@": a
    bare word in text is left alone."""
    tok = mint.tag("HANDLE", nick)
    at = "@?" if ("_" in nick or "." in nick or any(c.isdigit() for c in nick)) else "@"
    rx = re.compile(r"(?:(?<=\\[ntrbfv])|(?<![\w@.\\-]))(" + at + r")(" + re.escape(nick)
                    + r")(?![\w.-]?[\w])", re.IGNORECASE)

    def mask(m, t=tok):
        return m.group(1) + t

    def guard(m):
        return is_own_token(m.group(2), part=True)

    return Rule(f"{prefix}:handle:" + re.sub(r"[^a-z0-9]+", "-", nick), rx, mask=mask,
                value_group=2, guard=guard, confidence="high")


def rule_groups(roster: Roster) -> tuple:
    """(before_names, pack): pairs and initials — before the engine's name detector; in the pack
    slot — first handles (HANDLE), then single surnames."""
    early, surnames = [], []
    for p in roster.persons:
        pr = person_rules(p)
        early.extend(pr[:-1])
        surnames.append(pr[-1])
    handles = [handle_rule(nick) for nick in roster.handles]
    return early, handles + surnames


def rules(roster: Roster) -> list:
    """All roster rules as one list in chain order (tests, a single pack)."""
    early, pack = rule_groups(roster)
    return early + pack


KIND_PERSON = "roster.person"
KIND_HOMONYM = "roster.person:homonym"      # single homonym surname: a hit not at a sentence start


def controls(roster: Roster) -> list:
    """Roster forms for the check: [(form, kind, case_sensitive)]. Surnames and pairs are case
    sensitive (like the rules: "Сокол"/"СОКОЛ" — yes, "сокол" — no), handles are not. A single
    homonym surname has kind KIND_HOMONYM: the gate does not count a hit at a sentence start (like the rule)."""
    out = []
    for p in roster.persons:
        lits = person_literals(p)
        if p.latin:
            out.append((p.canon, KIND_PERSON, False))
            out.append((p.surname, KIND_PERSON, False))
            continue
        for pair in (p.canon, f"{p.surname} {p.given}"):
            out.append((pair, KIND_PERSON, True))
            out.append((pair.upper(), KIND_PERSON, True))
        kind = KIND_HOMONYM if p.homonym else KIND_PERSON
        for f in _cased(lits["surname"]):
            out.append((f, kind, True))
    for nick in roster.handles:
        out.append(("@" + nick, "roster.handle", False))
        if "_" in nick or "." in nick or any(c.isdigit() for c in nick):
            out.append((nick, "roster.handle", False))
    return out


def to_data(roster: Roster) -> dict:
    """Roster for ctx.data (exchange between stages and the owner's report; not in the manifest)."""
    return {"persons": [p.canon for p in roster.persons], "handles": list(roster.handles),
            "sources": dict(roster.sources)}


def from_data(data: dict) -> Roster:
    roster = Roster()
    for name in data.get("persons", ()):
        roster.add_person(name)
    for nick in data.get("handles", ()):
        roster.add_handle(nick)
    roster.sources = dict(data.get("sources", {}))
    return roster


# --------------------------------------------------------------------------- identity fields
_SLACK_USER_FILE = "users.json"


def mask_identity_fields(path: Path, engine, roster: Roster) -> int:
    """Discord/Slack JSON without an engine profile: people fields (``author.name``/``nickname``, Slack
    ``name``/``real_name``/``profile.display_name``) → tokens per field, as the Telegram profile does
    for ``from``: a full form — the person token, otherwise HANDLE by handle / PERSON_NAME by author id.
    Returns the number of changed fields; the file is rewritten with the same indent."""
    doc = _load_json(path)
    if doc is None or path.suffix.lower() != ".json":
        return 0
    n = 0
    for obj in _identity_objects(doc):
        uid = obj.get("id")
        uid = str(uid) if isinstance(uid, (str, int)) and str(uid).isdigit() else None
        for k in _PERSON_KEYS + _HANDLE_KEYS:
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip() or is_own_token(v.strip(), part=True):
                continue
            tok = _field_token(v, k, uid, engine)
            if tok is not None and tok != v:
                obj[k] = tok
                n += 1
        prof = obj.get("profile")
        if isinstance(prof, dict):
            for k in ("real_name", "display_name", "real_name_normalized", "display_name_normalized",
                      "first_name", "last_name", "email", "phone", "title"):
                v = prof.get(k)
                if not isinstance(v, str) or not v.strip() or is_own_token(v.strip(), part=True):
                    continue
                tok = _field_token(v, k, uid, engine)
                if tok is not None and tok != v:
                    prof[k] = tok
                    n += 1
    if n:
        raw = path.read_text(encoding="utf-8", errors="replace")
        indent = 2 if re.search(r"\n[ \t]+\"", raw) else None
        out = json.dumps(doc, ensure_ascii=False, indent=indent)
        if raw.endswith("\n"):
            out += "\n"
        path.write_text(out, encoding="utf-8")
    return n


def _field_token(value: str, key: str, uid, engine):
    v = " ".join(value.split())
    if key in ("email",):
        from anonymizer.engine import placeholders as P
        return P.EMAIL
    if key in ("phone",):
        from anonymizer.engine import placeholders as P
        return P.PHONE
    if key == "title":
        return engine.mask_text(v) if engine is not None else None
    c = canonical(v)
    if c:
        given, surname = person_parts(c)
        return Person(c, given, surname, not _HAS_CYR.search(surname)).token
    masked = engine.mask_text(v) if engine is not None else v
    if masked != v and is_own_token(masked):
        return masked
    if key in _HANDLE_KEYS and _NICK.fullmatch(v):
        return mint.tag("HANDLE", v.lower())
    if uid:
        return mint.pname("user:" + uid)
    return mint.pname("name:" + v.lower().replace("ё", "е"))
