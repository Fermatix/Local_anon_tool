"""Entity registry of a structured profile: platform id → person → token.

Why. In a messenger export the same person appears under different display
names (profile name, name from the export owner's address book, Latin script)
in dozens of chats and in several exports. The stable platform id
(``from_id`` like ``user<number>``) is the LINK KEY: it collects all these
names into one entity. The token itself is NOT minted from the id (data owner
decision: a person in Telegram fields and the same person in the text are one
entity with one token): the entity token is the one the regular pipeline
gives this person by name.

How "field token == token in the text" is achieved:

  1. the canonical name of an entity is the most frequent FULL form
     "Given Surname" among its names (``person_parts``: a known given name +
     a word with surname morphology/from the surname dictionary), normalized
     to the "Given Surname" order;
  2. the canonical string goes through THE SAME rule chain as any text of the
     run (``Registry.prepare(chain)``; chain = Pipeline._apply_fixpoint:
     names.* detectors + pack) — the PERSON_NAME token is taken from the
     result. This is literally "what would happen to this name in the text":
     a "Given Surname" pair in a message, transcript or document is minted by
     the same rule (names.fullname_pair → mint.pname(_seed_person(surname))),
     oblique cases — by rules with the same seed (pack person-names /
     person-ci);
  3. the chain gave no PERSON_NAME (surname outside SURNAME_ENDINGS, no pack
     yet) — ``mirror_seed``: the seed the pack will mint the name with after
     the roster bulk verdict (tools/packgen.expand_person: stem of the first
     surname part, gazetteer._stem — shared with the pack's first_name_rx; a
     test keeps them equal);
  4. the entity has NO full form (only a given name / nick / "Mom") — nothing
     in the text matches it (single given names are not masked as literals,
     data owner rule), so the token is minted from the id: one person = one
     token in all chats and exports, different people with the same given
     name do not merge.

Username, phone and id are their own classes: HANDLE (mint.tag by the
lowercase nick, as handle_at / pack:handles), PHONE_REDACTED (a constant, as
contacts), USER_ID (mint.tag by the id digits, as discord_ids) — deterministic
and equal everywhere by construction.

Entity key (``users``) is a string: the Telegram peer id digits, "yt:<login>"
for a YouTrack user (structured/youtrack.py). The token does not depend on the
key while the entity has a full name form: "Ivan Petrov" as a YouTrack issue
author and "Ivan Petrov" as a Telegram sender in batches of one group (one
salt) get one PERSON_NAME. The platform login (``accounts``) and the original
display name (``people``) of a YouTrack user go to mining in the yt-users
cohort, as with the old harvester (tools/mine.harvest_structured_roster); for
a YouTrack user without a full name form ("Petrov I.O.", a single surname,
Latin script) the field token is what the rule chain (with the run pack) gives
the display name itself, if it masks it (_raw_token): a yt-users literal in
the text gets the same token.

The registry is built ONCE from the sources (``<run>/raw`` if present: work/
is already masked after the first round) and cached in ``<run>/state/
structured_registry.json`` (see engine.registry_for_tree); canonical full
names and nicks go to the mining roster (cohort ``tg-roster``).
"""
from __future__ import annotations

import re
from collections import Counter

from ..engine import gazetteer as gaz
from ..engine import mint, name_dicts
from ..engine.boundary import is_own_token
from ..engine.vocab import COMMON_EN, SURNAME_ENDINGS, TECH_TERMS, TRANSLIT

VERSION = 1

# one "letter" part of a name: a Capital + lowercase letters of one alphabet,
# a hyphen is allowed (double given names and surnames)
_CYR_WORD = re.compile(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")
_LAT_WORD = re.compile(r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?")
# display name decoration (emoji, brackets, badges) — not part of the name
_DECOR = re.compile(r"[^\w\s'’-]|_", re.UNICODE)
_SURN_END = re.compile(SURNAME_ENDINGS + r"$")
_PATRONYMIC = re.compile(r".{2,}(?:ович|евич|ьич|ич|овна|евна|ична|инична)$")
_PNAME_TOKEN = re.compile(r"PERSON_NAME_\d+")
_NICK = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,31}")

# separator for batch minting of canonical names: wider than any rule context
# window (Rule.window, default 120) — a neighbour name does not serve as context
_ISO = "\n" + "\ue000" * 200 + "\n"

_GIVEN_CYR = frozenset(gaz.RU_FIRST_NAMES_ALL) | frozenset(gaz.RU_FIRST_NAMES_DIM)
_EN_GIVEN = frozenset(n.lower() for n in gaz.EN_FIRST_NAMES)
_LAT_COMMON = frozenset(COMMON_EN) | frozenset(TECH_TERMS)
# mirror of tools/packgen._TOKEN_PREFIX_WORDS / _LATIN_STOPWORDS (seed of the latin part)
_TOKEN_PREFIX_WORDS = {"USER", "USER_ID", "PERSON", "PERSON_NAME", "ORG",
                       "HANDLE", "RESOURCE", "DOC_ID", "GEO", "DATE", "PHONE",
                       "EMAIL", "IP", "MAC", "STREET", "HOUSE", "FLAT",
                       "POSTCODE", "SECRET", "ENCODED", "CRM", "REDACTED"}
_LATIN_STOPWORDS = {"and", "or", "the", "of", "for", "not", "with", "from",
                    "van", "von", "der", "den", "de", "la", "le", "el",
                    "aka", "at", "in", "on", "to", "by", "as", "an", "a"}


def clean_name(name) -> str:
    """Display name → a string without edge spaces and repeated spaces. An
    already masked value (a token: the registry is collected over the tree
    after a round, when there are no raw/ sources) is not a name → ''."""
    if not isinstance(name, str):
        return ""
    name = " ".join(name.split())
    if is_own_token(name.lstrip("@")):          # only our own token
        return ""
    return name


def _lat_given(word: str) -> bool:
    low = word.lower()
    if low in _EN_GIVEN:
        return True
    # transliterated Russian given name (Sergey / Sergei / Ivan): key = TRANSLIT of the core
    return low in _RU_GIVEN_LAT


def _translit(word: str) -> str:
    return "".join(TRANSLIT.get(ch, ch) for ch in word.lower())


_RU_GIVEN_LAT = frozenset(
    k for n in gaz.RU_FIRST_NAMES for k in (
        _translit(n), _translit(n).replace("iy", "y").replace("ey", "ei"),
        _translit(n).replace("y", "i")))


def _cyr_given(word: str) -> bool:
    return word in _GIVEN_CYR


def _cyr_surname_like(word: str) -> bool:
    """Surname morphology (SURNAME_ENDINGS after cutting the case/feminine
    tail) or the russiannames surname dictionary. Common nouns used as address
    book notes ("neighbour", "marketing", "coach") do not pass; dictionary
    homonyms pass with rank 2 (see form_rank)."""
    return _cyr_morph_surname(word) or name_dicts.is_surname(word)


def person_parts(name):
    """Full form "Given Surname" → (given, surname) | None.

    Full form: 2–3 Capitalized words of one alphabet (decoration — emoji,
    brackets — does not count), among them a known given name and exactly one
    surname-like word (a patronymic, if any, is dropped). Any word order:
    "Petrov Sergey" → ("Sergey", "Petrov"). A single given name, a nick, a
    given name + common noun, "Mom" → None: there is no full form, such a
    name does not go to the roster and is not masked as a literal in the
    text. Reliability of the found form — ``form_rank``.
    """
    s = clean_name(_DECOR.sub(" ", name) if isinstance(name, str) else "")
    words = s.split()
    if not 2 <= len(words) <= 3:
        return None
    if all(_CYR_WORD.fullmatch(w) for w in words):
        given = [w for w in words if _cyr_given(w)]
        rest = [w for w in words if not _cyr_given(w)
                and not _PATRONYMIC.fullmatch(w)]
        if not given or len(rest) != 1:
            return None
        surname = rest[0]
        if not _cyr_surname_like(surname):
            return None
        return given[0], surname
    if len(words) == 2 and all(_LAT_WORD.fullmatch(w) for w in words):
        a, b = words
        if _lat_given(a):
            given, surname = a, b           # western order: Given Surname
        elif _lat_given(b):
            given, surname = b, a
        else:
            return None
        if len(surname) < 3 or surname.lower() in _LAT_COMMON:
            return None
        return given, surname
    return None


def canonical(name):
    """Normalized full form "Given Surname" or None."""
    parts = person_parts(name)
    return f"{parts[0]} {parts[1]}" if parts else None


def _cyr_morph_surname(word: str) -> bool:
    low = word.lower().replace("ё", "е")
    return any(_SURN_END.search(name_dicts.strip_case_tail(part))
               for part in low.split("-"))


def form_rank(canon: str) -> int:
    """Reliability of a full form as a person's name (lower is more reliable):
    0 — Cyrillic, the surname has surname morphology or is from the dictionary
        and is NOT a common noun ("Sergey Petrov", "Marina Smirnova");
    1 — Latin ("Sergey Petrov");
    2 — Cyrillic, the "surname" is a dictionary homonym (a real surname that is
        also a common word, or an address book note like "Sasha Work").
    Ranks 0–1 go to the mining roster (bulk mask without an LLM); rank 2 does
    not (the regular pipeline judges it: NER + verdict), but the entity token
    is still minted by name — if the pipeline masks the surname in the text,
    the tokens match."""
    surname = canon.split()[-1]
    if not re.search(r"[А-Яа-яЁё]", surname):
        return 1
    if _cyr_morph_surname(surname) or not name_dicts.is_homonym(surname):
        return 0
    return 2


def mirror_seed(canon: str) -> str:
    """The seed tools/packgen.expand_person will mint the person literal
    ``canon`` (a full form) with: the stem of the first Cyrillic surname part
    (gazetteer._stem — it is also group(1) of first_name_rx in the
    pack:person-names rule), otherwise the first usable Latin part, otherwise
    the stem of the given name, otherwise the literal itself. The packgen
    corpus guard "part = common word" is not available here (it needs the run
    frequency list) — a mismatch is possible only for a common-noun part, and
    such names do not pass as a full form."""
    surn = latin = given = None
    for part in canon.split():
        if not re.search(r"[А-Яа-яЁё]", part):
            if part.upper() in _TOKEN_PREFIX_WORDS \
                    or len(part) < 3 or part.lower() in _LATIN_STOPWORDS \
                    or len(part) <= 3 or part.lower() in _LAT_COMMON:
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9._'\-]*", part):
                latin = latin or part.lower()
        elif len(gaz._stem(part)) <= 2:
            continue
        elif part in gaz.RU_FIRST_NAMES or part in gaz.RU_FIRST_NAMES_DIM:
            given = given or gaz._stem(part).lower()
        elif re.fullmatch(r"[А-ЯЁ][А-Яа-яЁё-]+", part):
            surn = surn or gaz._stem(part).lower()
    return surn or latin or given or canon.lower()


def client_role_token(name):
    """A display name / value of a people field of a profile (Telegram ``from``,
    YouTrack ``fullName``/``name``) that as a whole equals a client role label
    ("Contractor 12") → the same ORG token the engine rule
    orgs.client_role_placeholder gives in the text of the same export (one
    label — one token in all fields and in the text); otherwise None."""
    from ..detectors.orgs import client_role_seed
    seed = client_role_seed(name) if isinstance(name, str) else None
    return mint.tag("ORG", seed) if seed is not None else None


def client_roles_mixed(name, rest_fn):
    """A label INSIDE a people field value ("Client 2 (decision maker)",
    "Contractor 12 (backend)") → the same ORG token as the rule in the text,
    the rest of the value — ``rest_fn(core of the rest)``: the regular path of
    the same field (orgs.client_roles_split); None — no labels. Repeated pass:
    a value with our token as a word is already masked (telegram/youtrack
    ``_masked``) and does not get here."""
    from ..detectors.orgs import client_roles_split
    return client_roles_split(name, rest_fn) if isinstance(name, str) else None


def pick_token(chain_out: str, canon: str) -> str:
    """The entity PERSON_NAME token from the rule chain result over the
    canonical string. One token (the whole pair / surname + the kept first
    word / a pair of equal ones) — that one; several different ones (the pack
    mints given name and surname separately) — the surname token = the mirror
    seed; none — the mirror seed."""
    toks = _PNAME_TOKEN.findall(chain_out)
    if toks and len(set(toks)) == 1:
        return toks[0]
    mirror = mint.pname(mirror_seed(canon))
    if not toks or mirror in toks:
        return mirror
    return toks[-1]                 # canon order "Given Surname": the surname is last


class Registry:
    """id → names → canonical name → token. Serialized to JSON (the run
    cache); tokens are not serialized — they are a function of the run rules."""

    def __init__(self):
        self.users: dict = {}       # id digits -> Counter(display name)
        self.titles: Counter = Counter()   # channel/group names (not people)
        self.loose: Counter = Counter()    # people names from fields without an id
        self.handles: Counter = Counter()  # nicks (lowercase)
        self.owners: set = set()    # nicks of the export owners
        # YouTrack: account logins and original display names
        # (as in the field, not normalized) -> count
        self.accounts: Counter = Counter()
        self.people: Counter = Counter()
        self._raw: dict = {}        # display name -> token | None (_raw_token)
        self.rels: dict = {}        # canon/nick -> relpath of the first occurrence
        self._canon: dict | None = None    # id digits -> canon | None
        self._by_name: dict | None = None  # display name -> {id digits}
        self._tok: dict = {}        # canon -> token
        self._chain = None
        # tokens minted lazily (not in prepare) — canon -> token; the
        # incremental computation checks them against the old run
        self._lazy: dict = {}

    # ---- collecting ----
    def add_user(self, digits: str, name, rel=None) -> None:
        name = clean_name(name)
        if not digits or not name:
            return
        c = self.users.get(digits)
        if c is None:
            c = self.users[digits] = Counter()
        c[name] += 1
        if rel is not None and name not in self.rels:
            self.rels[name] = rel
        self._canon = None

    def add_title(self, name) -> None:
        name = clean_name(name)
        if name:
            self.titles[name] += 1

    def add_loose(self, name, rel=None) -> None:
        name = clean_name(name)
        if name:
            self.loose[name] += 1
            if rel is not None and name not in self.rels:
                self.rels[name] = rel
            self._canon = None

    def add_handle(self, nick, rel=None, owner=False) -> None:
        if not isinstance(nick, str):
            return
        nick = nick.strip().lstrip("@")
        if is_own_token(nick):
            return                      # an already masked nick (our token)
        nick = nick.lower()
        if _NICK.fullmatch(nick):
            self.handles[nick] += 1
            if owner:
                self.owners.add(nick)
            if rel is not None and "@" + nick not in self.rels:
                self.rels["@" + nick] = rel

    def add_account(self, login, rel=None) -> None:
        """Platform account login (YouTrack) — as in the field (mining:
        cohort yt-users, _handle_ok); the token is a HANDLE by the nick."""
        if not isinstance(login, str):
            return
        login = login.strip()
        if not login or is_own_token(login.lstrip("@")):
            return
        self.accounts[login] += 1
        if rel is not None and "@" + login not in self.rels:
            self.rels["@" + login] = rel

    def add_person(self, name, rel=None) -> None:
        """Original display name of a YouTrack user (mining: cohort yt-users
        via _person_route — "Surname I.O.", caps, short ones)."""
        if not isinstance(name, str) or not name.strip() \
                or is_own_token(name.strip()):
            return
        name = name.strip()
        self.people[name] += 1
        if rel is not None and name not in self.rels:
            self.rels[name] = rel

    def merge(self, other: "Registry") -> None:
        for d, c in other.users.items():
            self.users.setdefault(d, Counter()).update(c)
        self.titles.update(other.titles)
        self.loose.update(other.loose)
        self.handles.update(other.handles)
        self.owners |= other.owners
        self.accounts.update(other.accounts)
        self.people.update(other.people)
        for k, v in other.rels.items():
            self.rels.setdefault(k, v)
        self._canon = None

    # ---- derived ----
    def _derive(self) -> None:
        if self._canon is not None:
            return
        canon, by_name = {}, {}
        for digits, names in self.users.items():
            votes: Counter = Counter()
            for name, n in names.items():
                by_name.setdefault(name, set()).add(digits)
                c = canonical(name)
                if c:
                    votes[c] += n
            # the most frequent full form among the most reliable ones
            # (form_rank: a Cyrillic full name before Latin and before an
            # address book "surname" homonym); on a tie — the lexicographically
            # first (determinism between runs and processes)
            canon[digits] = (min(votes, key=lambda k: (form_rank(k),
                                                       -votes[k], k))
                             if votes else None)
        self._canon, self._by_name = canon, by_name

    def name_kind(self, name) -> str:
        """'user' — a known user's name; 'title' — a channel/group name;
        'person' — a full form "Given Surname"; '' — not recognized."""
        self._derive()
        name = clean_name(name)
        if name in self._by_name:
            return "user"
        if name in self.titles:
            return "title"
        return "person" if canonical(name) else ""

    # ---- minting ----
    def prepare_names(self) -> list:
        """Names that prepare mints (sorted, no duplicates): canonical names
        and original YouTrack display names (``people``, spaces normalized)."""
        self._derive()
        names = {c for c in self._canon.values() if c}
        for name in self.loose:
            c = canonical(name)
            if c:
                names.add(c)
        names.update(" ".join(n.split()) for n in self.people)
        return sorted(names)

    def prepare(self, chain, outs=None) -> None:
        """Batch minting of canonical name tokens in ONE call of the rule
        chain (``chain(text) -> text``): each name on its own line, with a
        separator wider than the context window between them. Called in the
        parent before forking workers; a repeated call with another chain
        resets the cache.

        ``outs`` — chain outputs for prepare_names(), already computed (a pool
        over name groups — Pipeline._prepare_names_parallel, with a check that
        grouping does not change the output); None — one call here, as before."""
        names = self.prepare_names()
        self._chain = chain
        self._tok = {}
        self._lazy = {}
        self._raw = {}
        if not names:
            return
        if outs is not None and len(outs) != len(names):
            raise SystemExit(f"structured: prepare got {len(outs)} "
                             f"outputs for {len(names)} names")
        if outs is None and chain is not None:
            outs = chain(_ISO.join(names)).split(_ISO)
            if len(outs) != len(names):       # a rule ate the separator
                outs = [chain(n) for n in names]
        people = {" ".join(n.split()) for n in self.people}
        for i, name in enumerate(names):
            out = outs[i] if outs else ""
            self._tok[name] = pick_token(out, name)
            if name in people:
                self._raw[name] = self._tok[name] if _PNAME_TOKEN.search(out) \
                    else None

    def _canon_token(self, canon: str) -> str:
        tok = self._tok.get(canon)
        if tok is None:
            out = self._chain(canon) if self._chain is not None else ""
            tok = self._tok[canon] = pick_token(out, canon)
            self._lazy[canon] = tok
        return tok

    def user_token(self, digits: str, name=None) -> str:
        """A person's token by id: the entity canonical name → the pipeline
        token; no full form → minted from the id (see the module header, item 4)."""
        self._derive()
        role = client_role_token(name) if name else None
        if role is not None:
            return role
        if name:
            mixed = client_roles_mixed(name, lambda rest: self.user_token(digits, rest))
            if mixed is not None:
                return mixed
        canon = self._canon.get(digits)
        if digits.startswith("yt:") and name:
            # a YouTrack user: the chain token for the display name ITSELF, if the
            # chain masks it (a yt-users cohort literal in the pack, a detector pair) —
            # the same as this name gets in the text; otherwise — canonical form / id
            tok = self._raw_token(" ".join(name.split()))
            if tok is not None:
                return tok
        if canon is None and digits not in self.users and name:
            # the id was not there when the registry was built (cache older than the file): by name
            canon = canonical(name)
        if canon:
            return self._canon_token(canon)
        return mint.pname("tg:user" + digits)

    def _raw_token(self, name: str):
        """The chain PERSON_NAME token for a display name | None (the chain
        does not mask the name). A lazy token — only a found one: the
        incremental check (_canon_token) gives the same answer for it."""
        if name in self._raw:
            return self._raw[name]
        tok = None
        if self._chain is not None:
            out = self._chain(name)
            if name not in self._tok:
                # lazy token: the same as _canon_token(name) gives at the check
                self._tok[name] = self._lazy[name] = pick_token(out, name)
            if _PNAME_TOKEN.search(out):
                tok = self._tok[name]
        self._raw[name] = tok
        return tok

    def name_token(self, name) -> str:
        """A person's token from a field without an id: the only known user
        with this name → their entity; a full form → the pipeline token;
        otherwise — minted from the name itself (namesakes without an id are
        indistinguishable)."""
        self._derive()
        name = clean_name(name)
        role = client_role_token(name)
        if role is not None:
            return role
        mixed = client_roles_mixed(name, self.name_token)
        if mixed is not None:
            return mixed
        ids = self._by_name.get(name)
        if ids and len(ids) == 1:
            return self.user_token(next(iter(ids)))
        canon = canonical(name)
        if canon:
            return self._canon_token(canon)
        return mint.pname("tg:name:" + name.lower().replace("ё", "е"))

    # ---- mining roster ----
    def roster(self) -> list:
        """[(literal, class, relpath|None)] — known literals of the run:
        canonical full names (person) and nicks (handle). Single given names
        without a surname do not get here (data owner rule)."""
        self._derive()
        out, seen = [], set()
        for digits in sorted(self._canon):
            c = self._canon[digits]
            if c and c not in seen and form_rank(c) < 2:
                seen.add(c)
                rel = next((self.rels[n] for n in self.users[digits]
                            if n in self.rels), None)
                out.append((c, "person", rel))
        for name in sorted(self.loose):
            c = canonical(name)
            if c and c not in seen and form_rank(c) < 2:
                seen.add(c)
                out.append((c, "person", self.rels.get(name)))
        for nick in sorted(self.handles):
            # a nick literal is masked over the whole corpus WITHOUT "@" — the
            # roster gets only the export owners' nicks and nicks that do not
            # look like a word (with "_" or a digit). A purely alphabetic
            # mentioned nick ("@market") is masked in place (mention entity /
            # handle_at rule), but the word "market" in texts is left alone.
            if nick in self.owners or "_" in nick \
                    or any(c.isdigit() for c in nick):
                out.append((nick, "handle", self.rels.get("@" + nick)))
        return out

    def yt_roster(self) -> tuple:
        """YouTrack → mining in the yt-users cohort (as the old harvester):
        ([(login, relpath)], [(display name, relpath)]), sorted."""
        return ([(x, self.rels.get("@" + x)) for x in sorted(self.accounts)],
                [(x, self.rels.get(x)) for x in sorted(self.people)])

    def counts(self) -> dict:
        self._derive()
        return {"users": len(self.users),
                "users_full_name": sum(1 for c in self._canon.values() if c),
                "users_multi_name": sum(1 for c in self.users.values()
                                        if len(c) > 1),
                "titles": len(self.titles), "loose": len(self.loose),
                "handles": len(self.handles), "accounts": len(self.accounts),
                "people": len(self.people)}

    # ---- serialization ----
    def to_json(self) -> dict:
        doc = {"version": VERSION,
               "users": {d: dict(c) for d, c in sorted(self.users.items())},
               "titles": dict(self.titles), "loose": dict(self.loose),
               "handles": dict(self.handles), "owners": sorted(self.owners),
               "rels": self.rels}
        if self.accounts:
            # platform logins (YouTrack) — only if present: the registry cache of
            # Telegram exports stays byte for byte the same (test_own_token: an old code cache is taken)
            doc["accounts"] = dict(self.accounts)
        if self.people:
            doc["people"] = dict(self.people)
        return doc

    @classmethod
    def from_json(cls, doc: dict) -> "Registry":
        reg = cls()
        reg.users = {d: Counter(c) for d, c in doc.get("users", {}).items()}
        reg.titles = Counter(doc.get("titles", {}))
        reg.loose = Counter(doc.get("loose", {}))
        reg.handles = Counter(doc.get("handles", {}))
        reg.owners = set(doc.get("owners", ()))
        reg.accounts = Counter(doc.get("accounts", {}))
        reg.people = Counter(doc.get("people", {}))
        reg.rels = dict(doc.get("rels", {}))
        return reg
