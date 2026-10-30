"""Secret detectors.

Two tiers here (the entropy tier lives in engine.secret + the pipeline's
format-aware value pass):
  1. VENDOR SIGNATURES — self-identifying tokens (JWT, AWS/Google/GitHub/Slack/
     Stripe/OpenAI/Telegram/HF/…). Mask always, key-independent.
  2. LABEL-ANCHORED — a `пароль/password/api-key/token: <value>` where the value
     also passes the entropy gate; masks the value, keeps the label.
  3. UNLABELLED (Telegram audit) — a password as a line/message below a label, a session cookie and
     bare_key_token: a RANDOM key without a label (base62/base64 ≥24, all three classes, scattered
     digits and capitals, not an identifier made of words, not base64 of text). The label-free tier
     used to be handed to the masking agent (see «REMOVED: risky_token» below); here it is
     narrow and has Unicode boundaries — glues with Cyrillic do not match.
  4. (Telegram re-run audit, SECRET critical) — «Bearer <token>» without
     a header, a key in a link parameter (?api=/?token=/&authkey=), an environment variable
     *_PASSWORD_*/*_KEY=, a password after a label without a colon and in brackets, a key/token label
     on the line above, a meeting PIN and passcode, the Figma share parameter (t=), a bot token in
     a Bot API link, ids of shared links (Dropbox, Yandex Disk, Google Drive folders).
     A value made entirely of our tokens is not considered a secret (_has_token); a token with
     a meaningful rest is judged whole, as before.
  5. (owner's decision) — a password WITHOUT a label: a message of one
     "word" of password form (lone_password_message), a password line under an email line
     (credential_after_email) and an access block "label/link → login → password"
     (credential_login_password).
"""
from __future__ import annotations

import re

import base64
import binascii

from ..engine import mint
from ..engine import placeholders as P
from ..engine.boundary import JSON_SAFE as _JS, L, OWN_TOKEN_RE, is_placeholder
from ..engine.rule import Rule
from ..engine.vocab import TECH_TERMS
from ..engine.secret import is_secret_value

_SIG = [
    ("jwt", r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("aws_akid", r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    ("google_api", r"AIza[\w-]{30,}"),
    ("github_pat", r"gh[pousr]_[A-Za-z0-9]{30,}"),
    ("slack", r"xox[abprs]-[\w-]{10,}"),
    ("sendgrid", r"SG\.[\w-]{16,}\.[\w-]{16,}"),
    ("stripe", r"(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    ("openai", r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    ("telegram_bot", r"\d{8,10}:AA[\w-]{30,}"),
    ("twilio", r"(?:AC|SK)[0-9a-f]{32}"),
    ("digitalocean", r"do[a-z]_v1_[a-f0-9]{64}"),
    ("huggingface", r"hf_[A-Za-z0-9]{20,}"),
    ("vk_token", r"vk1\.a\.[A-Za-z0-9_-]{50,}"),
    ("google_play", r"AO-J1Oy[A-Za-z0-9_-]{20,}"),
]

# the gap "label … <hex>" ends with a digest marker — that is a hash,
# not a key: «commit <sha>», «sha=<sha>» (CI log), «sha256:<hex>» (docker),
# «md5 <hex> archive.zip», «checksum/hash/хеш/контрольная сумма: <hex>».
# The tail [\s:=()\[\]`"']* is punctuation between the marker and the value.
_DIGEST_TAIL = re.compile(
    r"(?i)\b(?:commit\w*|sha-?\d*(?:sums?)?|md5(?:sums?)?|checksum\w*|"
    r"digest\w*|hash\w*|хеш\w*|хэш\w*|сумм\w*|revision|rev)"
    r"[\s:=()\[\]`\"']*$")

# needs for signatures from which the prefilter cannot derive a probe: github_pat
# «gh[pousr]_» — the match must contain one of the five 4-char prefixes
# (for the other signatures the literal prefix is derived automatically).
_SIG_NEEDS = {"github_pat": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_")}


def _decodes_to_text(v: str) -> bool:
    """A base64/base64url blob that decodes to printable text is the business of detectors/encoding
    (the decoding judges it by a PII signal); left alone here so as not to mask harmless text."""
    s = v.rstrip("=")
    if len(s) % 4 == 1:
        return False
    for dec in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            bs = dec(s + "=" * (-len(s) % 4))
        except (binascii.Error, ValueError):
            continue
        try:
            t = bs.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if t and sum(ch.isprintable() or ch in "\t\n\r" for ch in t) / len(t) >= 0.85:
            return True
    return False


_WORDISH = re.compile(r"[A-Z]?[a-z]{3,}")
_WORD_SEG = re.compile(r"[A-Z]?[a-z]{3,}|[A-Z]{3,}(?![a-z])")


def _made_of_words(v: str) -> bool:
    """An identifier made of words, not a random key: word-like segments cover ≥55% of the string
    («MyComponent2Test3Spec4Helper») or ≥2 segments are English words of the SCOWL dictionary
    («iPhone12ProMax256GB2023Edition», «Win10Pro64Bit2Update3Build»). Measured on random
    base62/base64/base64url keys of length 24–100: this filters out ≤0.65% (before this filter
    identifiers with ≥3 groups of digits and capitals went to SECRET)."""
    if sum(len(x) for x in _WORDISH.findall(v)) >= 0.55 * len(v):
        return True
    from ..engine import lexicon
    en = lexicon._en("words")
    hits = 0
    for s in _WORD_SEG.findall(v):
        if s.lower() in en:
            hits += 1
            if hits >= 2:
                return True
    return False


def _token_rest(v: str) -> str:
    """Letters and digits of the value outside our tokens."""
    return re.sub(r"[\W_]+", "", OWN_TOKEN_RE.sub(" ", v))


def _has_token(v: str) -> bool:
    """The value is already covered — it is not a secret: it consists entirely of our tokens and separators
    («ORG_N/ORG_N», «RESOURCE_N», «SECRET_REDACTED»). Rules see each other's text on the fixpoint
    passes.
    A token with a meaningful rest is NOT considered covered («ORG_N-RTRG-1234567-1aBcd»,
    «пароль: ORG_N-2024», «ORG_N_admin» under an access label, «SECRET_REDACTED-tail» of a long
    key): the rest is part of the same key or password, the rule judges the value whole, as before.
    A token is recognised by known prefixes (OWN_TOKEN_RE): a piece «xF_1» of a random key is not our
    token. The previous version (any «CAPITALS_digits» or «_REDACTED» inside
    the value) left API keys, a pixel id and tails of compound keys open on raw tg
    sources. Rules that already skipped a value with a
    *_REDACTED constant (a key without a label, a password on the line below and after words) check it themselves."""
    return OWN_TOKEN_RE.search(v) is not None and not _token_rest(v)


def _token_list(v: str) -> bool:
    """A list of our tokens with a short tail is not a key without a label («ORG_N/ORG_N/ORG_N/YouTube»,
    «…/Ab9Cd» after a pack were folded by bare_key_token into SECRET_REDACTED whole): outside
    the tokens there are no digits or fewer than 8 letters and digits. A long rest with digits is part of an identifier,
    the value is judged whole."""
    if OWN_TOKEN_RE.search(v) is None:
        return False
    rest = _token_rest(v)
    return len(rest) < 8 or not re.search(r"\d", rest)


def _bare_key_guard(m) -> bool:
    """True => do not mask. A key without a label: all three classes (lowercase, capitals, digits),
    the entropy form of a secret (is_secret_value, hash_ok=False — bare md5/sha digests and
    hex ids are left alone), not an identifier made of words, not a placeholder token and not base64 of printable
    text."""
    v = m.group(1)
    if is_placeholder(v) or "_REDACTED" in v or _has_token(v) or _token_list(v):
        return True
    if not (re.search(r"[a-z]", v) and re.search(r"[A-Z]", v) and re.search(r"\d", v)):
        return True
    # a random key: digits and capitals are scattered across the string (≥3 separate groups of each);
    # in a code identifier («ConstantName_WITH_digits_123») the digits are clustered — not a secret
    if len(re.findall(r"\d+", v)) < 3 or len(re.findall(r"[A-Z]+", v)) < 3:
        return True
    if _made_of_words(v):
        return True
    if not is_secret_value(v):
        return True
    return _decodes_to_text(v)


def _pw_next_ok(v: str) -> bool:
    """A value as a line/message below a password label looks like a password: not a placeholder, not
    a link/email/path, an entropy form or "word+digits" (is_secret_value, hash_ok=True)."""
    if is_placeholder(v) or "_REDACTED" in v or _has_token(v) or "://" in v or "@" in v \
            or v.startswith("/"):
        return False
    return is_secret_value(v, hash_ok=True)


_HASH_WORDS = re.compile(r"(?i)(?<![\w])(?:hash\w*|digest|checksum|md5\w*|sha\d*|хеш\w*|хэш\w*|"
                         r"контрольн\w*)(?![\w])")


def _digest_ctx(label: str, v: str) -> bool:
    """«password hash: <hex>», «пароль (md5): …» — a digest, not a password (bare md5/sha are not
    a secret); pure hex ≥16 under a word password label — also a digest/identifier."""
    return bool(_HASH_WORDS.search(label)) or re.fullmatch(r"[0-9a-fA-F]{16,}", v) is not None


def _pw_next_guard(m) -> bool:
    """True => do not mask. A label line with a table separator («id;login;password») and
    a value with the same separator — a CSV/table header and a data row, not a password on the line below
    (the password cell is masked by the column/entropy pass)."""
    v = m.group(2)
    s, a = m.string, m.start(1)
    head = s[s.rfind("\n", 0, a) + 1:a] + m.group(1).split("\n", 1)[0]
    if any(d in head and d in v for d in (";", ",", "\t", "|")):
        return True
    # the first line of the text is one label word («password»), the value is on the next line: the header
    # of a single-column CSV (cells are taken by the column entropy pass). The same pair inside
    # a message or as the next message (the slot separator) — a password, masked.
    if _HASH_WORDS.search(head):          # «password hash:» on the line above — a digest
        return True
    if s.rfind("\n", 0, a) < 0 and _PW_BARE.fullmatch(head) and "\ue000" not in m.group(1):
        return True
    return not _pw_next_ok(v)


# a key/token label also leads a value on the line below («apiKey» as a message, the key as the next one)
_PW_LABELS = (r"парол\w*|password|passwd|pwd|доступ\w*|креды|credentials|"
              r"api[_ -]?key|apikey|token|токен\w*|secret|секрет\w*")
_PW_BARE = re.compile(r"(?i)\ufeff?[^\S\n]*(?:" + _PW_LABELS + r")[^\S\n]*")


# --- Telegram re-run audit 2026-10-04: SECRET critical -----------------------------------------
_PW_SPECIAL = re.compile(r"[!@#$%^&*?+=~]")
_EMAILISH = re.compile(r"[\w.+-]@[\w-]+\.[A-Za-z]{2,}")
_FILE_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,4}$")
# a file name with a known extension is not a login or a password of an access block (raw
# tg sources: a block «link → file name → file name» went to HANDLE and SECRET). A list of
# extensions, not _FILE_EXT: a login «name.surname» with a short surname is not a file
_KNOWN_FILE = re.compile(
    r"(?i)\.(?:pdf|docx?|xlsx?|pptx?|txt|rtf|csv|json|xml|html?|md|log|sql|py|js|ts|zip|rar|7z|tar|gz|"
    r"png|jpe?g|gif|svg|webp|heic|bmp|psd|ai|fig|mp3|mp4|m4a|mov|avi|mkv|wav|ogg|webm|apk|exe|dmg)$")


def _not_value(v: str) -> bool:
    """Not a secret value: our token, a link/email/path/mention/hashtag, a version number."""
    return (_has_token(v) or "://" in v or v.startswith(("/", "@", "#", "www.", "+", "./", "~/"))
            or _EMAILISH.search(v) is not None or re.match(r"v?\d+(?:\.\d+)+(?![\w])", v) is not None)


def _strong_pw(v: str) -> bool:
    """A password form without a reliable label nearby: ≥8 chars, Latin and a digit and (a capital or
    a special char) — «Abc12345», «qwerty12!». A word, a number, a link, a token — no."""
    if len(v) < 8 or _not_value(v) or re.search(r"[А-Яа-яЁё]", v):
        return False
    return bool(re.search(r"[A-Za-z]", v) and re.search(r"\d", v)
                and (re.search(r"[A-Z]", v) or _PW_SPECIAL.search(v)))


# Branch (a) of _lone_pw_shape requires only different character classes, and the slots
# of a dev chat «PostgreSQL16!», «React18+Next14», «Ctrl+Alt+Del1», «10:00?Ok» went to SECRET.
# Point exceptions (not _made_of_words — it would open «Password1!», «Welcome2024!»): a compound
# slot of ≥2 technical names joined by "+", "/", ":" or "-" (a name — letters and up to 4 digits with a version
# dot, a numeric part is allowed), a time at the start, a technical brand with a version tail and one "!".
# The first version opened ordinary passwords — the brand list
# contained English dictionary words («Spring2024!», «Oracle12!», «Windows10!», «Office365!»,
# «Word2024!»), the year «2024» was taken for a version («Java2024!»), the compound slot took any two
# Latin words («Pass+Word1», «Secret+Key99», «Hello+World1»). Now: a single slot — only
# a brand NOT from the dictionary (_TECH_BRANDS; a test checks against SCOWL), a version — 1–3 digits or 4 not of the form
# 19xx/20xx; a compound slot — EACH word part from technical names (_TECH_NAMES: brands,
# including dictionary ones — «Win», «Next», «Office», — TECH_TERMS, keys).
# brands NOT from the English dictionary — a single slot «Brand<version>!» (tests: ∩ SCOWL = ∅)
_TECH_BRANDS = frozenset("""
postgresql postgres mysql mariadb sqlite mongodb redis kafka nginx macos ios ipados ubuntu debian
centos linux nodejs nextjs nuxt java kotlin php dotnet csharp django laravel iphone ipad xbox
playstation ps photoshop figma firefox html css ecmascript javascript js golang scala perl zsh
xcode intellij pycharm webstorm vscode jira gitlab github bitbucket jenkins ansible terraform
kubernetes k8s openshift elasticsearch kibana grafana prometheus rabbitmq docker graphql grpc
webpack eslint npm vue
""".split())
# technical names that are also English words — only as a PART of a compound slot
_TECH_WORDY = frozenset("""
android angular bash celery chrome confluence edge es excel fedora flask galaxy go helm net next
node office oracle pixel rails ruby rust safari spring svelte swift tomcat ts typescript unity
unreal vim win windows word
""".split())
_KEY_NAMES = frozenset("ctrl alt del shift cmd esc tab fn ins home end enter space backspace "
                       "opt option pgup pgdn".split())
_TECH_NAMES = _TECH_BRANDS | _TECH_WORDY | _KEY_NAMES | frozenset(TECH_TERMS)
_SLOT_PART = re.compile(r"([A-Za-z]{2,})\d{0,4}(?:\.\d{1,3})?|\d{1,4}(?:\.\d{1,3})?")
_BRAND_TAIL = re.compile(r"([A-Za-z]{2,}?)(\d{1,3}|(?!(?:19|20)\d\d)\d{4})(?:\.\d{1,3})?!?")


def _lone_name_slot(v: str) -> bool:
    """The slot is a name/compound name/time, not a password."""
    if re.match(r"\d{1,2}:\d{2}(?!\d)", v):
        return True                                   # «10:00?Ok» — a time at the start
    m = _BRAND_TAIL.fullmatch(v)
    if m and m.group(1).lower() in _TECH_BRANDS:
        return True                                   # «PostgreSQL16!», «Debian12!»
    parts = re.split(r"[+/:-]", v)
    if len(parts) < 2:
        return False
    words = 0
    for p in parts:
        pm = _SLOT_PART.fullmatch(p)
        if pm is None:
            return False
        if pm.group(1) is not None:
            if pm.group(1).lower() not in _TECH_NAMES:
                return False                          # «Pass+Word1» — not technical names
            words += 1
    return words >= 2                                 # «React18+Next14», «Ctrl+Alt+Del1»


# --- a password without a label (owner's decision) -------------------------------------------
def _lone_pw_shape(v: str) -> bool:
    """A message (a projection slot) of one "word" is a password if (a) it has lowercase, capitals,
    a digit and a special char («Password1!», «Xxxxxx@?xxx1234»), or (b) it is a random string of ≥12
    chars: lowercase, capitals and digits mixed (≥2 digit groups, ≥3 capital groups — as
    in bare_key_token), not made of dictionary words, an entropy form. «MacBookPro2021», «iPhone15»,
    «SALE2024!», versions, hashtags, nicknames, links, email, file names are left alone. A value
    made entirely of our tokens (_has_token via _not_value) is already covered; a list of tokens with
    a short tail («ORG_N/ORG_N/YouTube» after a pack) is not a password without a label (_token_list, as
    in bare_key_token: digits and capitals of the tokens themselves are not a password form)."""
    if not (8 <= len(v) <= 64) or _not_value(v) or _token_list(v) or _FILE_EXT.search(v):
        return False
    if not (re.search(r"[a-z]", v) and re.search(r"[A-Z]", v) and re.search(r"\d", v)):
        return False
    if _PW_SPECIAL.search(v) and len(v) <= 40:
        return not _lone_name_slot(v)
    return (len(v) >= 12 and len(re.findall(r"\d+", v)) >= 2 and len(re.findall(r"[A-Z]+", v)) >= 3
            and not _made_of_words(v) and is_secret_value(v))


def _cred_pw(v: str) -> bool:
    """A password line under an email/login line: a password form (_strong_pw) or an entropy form
    with a digit; Cyrillic, a link, an email, a file name, our token (and a list of tokens with a short
    tail — the label is weak here, as for a key without a label) — no."""
    if _not_value(v) or _token_list(v) or re.search(r"[А-Яа-яЁё]", v) \
            or not re.search(r"[A-Za-z]", v) or _FILE_EXT.search(v):
        return False
    return _strong_pw(v) or (re.search(r"\d", v) is not None and is_secret_value(v, hash_ok=True))


_LOGIN_STOP = frozenset("admin administrator root user test guest demo login password null none "
                        "default anonymous example".split())


def _cred_login_guard(m) -> bool:
    """True => do not mask. The login line is our constant of another class («SECRET_REDACTED»
    on the line above: a password without a label from a previous pass — not a login; HANDLE_N — a login from a previous
    round, the password under it is masked); the password is a list of tokens or not of a strong form."""
    login = m.group(2)
    if _has_token(login) and not login.startswith("HANDLE_"):
        return True
    if _KNOWN_FILE.search(login) or _KNOWN_FILE.search(m.group(4)):
        return True                     # a file name in place of the login or password — not an access block
    return _token_list(m.group(4)) or not _strong_pw(m.group(4))


def _cred_login_password_mask(m) -> str:
    """A login line (not an email, not a service word, not our token) → HANDLE (the seed is lowercase, the
    same as for @nickname and login_label), the password below → SECRET."""
    login = m.group(2)
    if not (_has_token(login) or "@" in login or login.lower() in _LOGIN_STOP):
        login = mint.tag("HANDLE", login.lower())
    return m.group(1) + login + m.group(3) + P.SECRET


def _url_secret_guard(m) -> bool:
    """True => do not mask. The value of a key parameter of a link: not a token/template placeholder
    ({key}, <token>, ${X}, %7B…), with a digit (or ≥16 chars in mixed case), not words
    («google_oauth2» after auth=) — then it is an access key/token."""
    v = m.group(2)
    if _has_token(v) or v[:1] in "{<$%[" or v.lower().startswith(("%7b", "your", "xxx")):
        return True
    if not (re.search(r"\d", v) or (len(v) >= 16 and re.search(r"[a-z]", v)
                                     and re.search(r"[A-Z]", v))):
        return True
    # words joined by "_/./-" with a version number at the end — a provider/mode name, not a key. Coverage
    # by "word-like" pieces (_made_of_words) does not fit here: a random key of lowercase letters and
    # digits (Dropbox rlkey) consists entirely of "words" of 3+ letters
    return re.fullmatch(r"[A-Za-z]+(?:[_.-][A-Za-z]+)*\d{0,2}", v) is not None


# the name is the evidence: segments PASSWORD/PASS/PWD/SECRET/TOKEN/APIKEY; a bare _KEY (SORT_KEY,
# PARTITION_KEY, CACHE_KEY) — only a name from the explicit list of secret keys or a value of a secret
# form
_ENV_STRONG_SEG = re.compile(r"_(?:PASSWORD|PASSWD|PASS|PWD|SECRET|TOKEN|APIKEY)(?:_|$)")
_ENV_KEY_SEG = re.compile(r"_KEY(?:_|$)")
_ENV_SECRET_KEY = re.compile(r"(?:^|_)(?:API|SECRET|ACCESS|PRIVATE|AUTH|AUTHORIZATION|ENCRYPTION|"
                             r"ENCRYPT|SIGNING|SIGNATURE|SIGN|LICENSE|MASTER|CLIENT|APP|SERVICE_ROLE|"
                             r"ANON|HMAC|JWT|SESSION|CRYPT|CIPHER|DEPLOY|ADMIN|SECURITY|SERVER|"
                             r"PUBLISHABLE|WEBHOOK|SSH)_KEY(?:_|$)")
_ENV_SAFE_TAIL = re.compile(r"_(?:PATH|FILE|FILES|DIR|URL|URI|HOST|PORT|NAME|TTL|EXPIRE\w*|LENGTH|"
                            r"SIZE|TYPE|HEADER|ALG\w*|ID|PREFIX|SUFFIX|FORMAT|ENABLED|MODE)$")
_ENV_PLACEHOLDER = frozenset("changeme change_me changeit password secret token null none true "
                             "false empty example your_password your_token your_secret xxxxxx "
                             "******".split())


def _mixed_alnum(v: str) -> bool:
    """Letters mixed with digits (≥3 letter↔digit junctions, ≥8 chars) — a key form, not a field
    name or a cache template («created_at», «v1_users_list»)."""
    return len(v) >= 8 and len(re.findall(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", v)) >= 3


def _env_secret_guard(m) -> bool:
    """True => do not mask. A secret variable (*_PASSWORD*, *_SECRET*, *_TOKEN*, a key from
    the list *_API_KEY/SECRET_KEY/…): the value is not a reference to another variable/template/path, not
    a placeholder and not a service tail of the name (SSH_KEY_PATH, TOKEN_TTL, JWT_SECRET_ALGORITHM).
    A bare _KEY outside the list (SORT_KEY, CACHE_KEY) is a secret only by the value form (entropy
    or letters mixed with digits, and not an identifier made of words)."""
    name = m.group(1).split("=")[0].split(":")[0].strip().strip("\"'")
    v = m.group(2)
    strong = _ENV_STRONG_SEG.search(name) is not None
    if not (strong or _ENV_KEY_SEG.search(name)) or _ENV_SAFE_TAIL.search(name):
        return True
    if _has_token(v) or v[:1] in "$%<{[/~." or v.lower() in _ENV_PLACEHOLDER:
        return True
    if len(set(v)) <= 2:                        # «******», «000000»
        return True
    if strong or _ENV_SECRET_KEY.search(name):
        return False
    # a bare _KEY: a value of a key form and not an identifier made of words («v1_users_list»)
    return not ((is_secret_value(v, hash_ok=True) or _mixed_alnum(v)) and not _made_of_words(v))


def _bearer_bare_guard(m) -> bool:
    """True => do not mask. After «Bearer» without a header — a token: ≥8 chars, letters and
    digits; «Bearer token», «Bearer <TOKEN>», «Bearer $TOKEN», our token — no."""
    v = m.group(2).rstrip("=")
    if len(v) < 8 or _has_token(v):
        return True
    return not (re.search(r"\d", v) and re.search(r"[A-Za-z]", v))


def _pw_inline_guard(m) -> bool:
    v = m.group(2)
    if v.isdigit():                             # «Пароль 12345678» (Wi-Fi, account): ≥8 digits
        return len(v) < 8 or _digest_ctx(m.group(1), v)
    return not _strong_pw(v) or _digest_ctx(m.group(1), v)


def _pw_paren_guard(m) -> bool:
    """True => do not mask. «(пароль X)»: X is not a token/link and not an ordinary lowercase word
    («(пароль стандартный)»); capitals, digits or Latin — a password."""
    v = m.group(2)
    if _not_value(v) or re.fullmatch(r"[а-яё-]+", v):
        return True
    return _digest_ctx(m.group(1), v)


def _pw_context_guard(m) -> bool:
    """password_context guard: a value with a digit/special char is a password; letters
    only — only after a colon and in CamelCase form ≥8 («Пароль: XxxXxxxxXxxx»)."""
    v = m.group(2)
    if is_placeholder(v) or _has_token(v) or not re.search(r"[A-Za-z]", v):
        return True
    if re.search(r"[\d!@#$%^&?*_\-]", v):
        return False
    return not (m.group(1).rstrip().endswith((":", "=")) and len(v) >= 8
                and re.search(r"[a-z][A-Z]", v) is not None)


def _passcode_guard(m) -> bool:
    """A meeting passcode: digits or mixed case; an explanatory word is not a code."""
    v = m.group(2)
    if _has_token(v) or "://" in v:
        return True
    return not (re.search(r"\d", v) or (re.search(r"[a-z]", v) and re.search(r"[A-Z]", v)))


_HOST_TOK: dict = {}


def host_token(*hosts) -> frozenset:
    """RESOURCE tokens of hosts (as orgs.domain mints them: the lowercase domain) — the rule recognises
    a link on an ALREADY masked host too («RESOURCE_N/<meeting code>» in a round's output, which
    the next round runs over). The salt is set before the run — the cache is keyed by (salt, hosts)."""
    key = (mint.require_tenant(), mint.mint_version(), hosts)
    hit = _HOST_TOK.get(key)
    if hit is None:
        hit = _HOST_TOK[key] = frozenset(mint.tag("RESOURCE", h) for h in hosts)
    return hit


def _tok_host_bad(m, *hosts) -> bool:
    """The link was recognised by a host RESOURCE token (group 1 starts with the host), but the token is
    not the token of any of ``hosts``: someone else's site with a similar path — left alone."""
    h = m.group(1).split("/", 1)[0]
    return h.startswith("RESOURCE_") and h not in host_token(*hosts)


def _figma_host_guard(m) -> bool:
    return _tok_host_bad(m, "figma.com", "www.figma.com")


def _dropbox_mask(m) -> str:
    """Dropbox link id → RESOURCE; the second segment «/scl/fo/<id>/<id2>» is also an id (no dot,
    with a digit or in mixed case); a file name «/scl/fi/<id>/<name.ext>» is left alone."""
    tail = m.group(3) or ""
    v2 = tail[1:]
    if v2 and not _has_token(v2) and (re.search(r"\d", v2) or (re.search(r"[a-z]", v2)
                                                               and re.search(r"[A-Z]", v2))):
        tail = "/" + mint.tag("RESOURCE", v2)
    return m.group(1) + mint.tag("RESOURCE", m.group(2)) + tail


RULES: list[Rule] = [
    Rule(name, re.compile(L + r"(" + pat + r")" + r"(?![A-Za-z0-9])"),
         mask=lambda m: P.SECRET, value_group=1, confidence="high",
         needs=_SIG_NEEDS.get(name))
    for name, pat in _SIG
] + [
    # facebook graph token — guard the RSA/PNG-tail false positives
    Rule("facebook_graph", re.compile(r"(?<![A-Za-z0-9])(EAA[A-Za-z0-9]{20,})(?![A-Za-z0-9])"),
         mask=lambda m: P.SECRET, value_group=1, confidence="context",
         guard=lambda m: bool(re.search(r"A{8,}|RU5ErkJgg|==$|EAAAADAQAB", m.group(1)))),

    # PEM private key block. The body limit was raised 6000→20000: RSA-8192 (~6.5KB) and
    # keys with encrypted-PEM headers did not fit into 6000 — the whole private
    # key (including the armour) leaked entirely (all-or-nothing). The limit stays
    # BOUNDED, the lazy quantifier keeps the scan linear for each BEGIN.
    # multiline=True: a PEM block legitimately collapses into one token —
    # the apply_rules safeguard "a mask does not change the line count" leaves it alone
    Rule("pem_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,20000}?-----END [A-Z ]*PRIVATE KEY-----"),
         mask=lambda m: P.SECRET, confidence="high", multiline=True),
    # [^\S\r\n]: \s+ matched across a line break and merged records
    Rule("ssh_pubkey", re.compile(r"\bssh-(?:rsa|ed25519|dss)[^\S\r\n]+[A-Za-z0-9+/=]{50,}"),
         mask=lambda m: P.SECRET, confidence="high"),

    # basic-auth in a connection URL — mask only the :password@ part
    Rule("basic_auth_url",
         re.compile(r"((?:https?|ftp|postgres(?:ql)?|mysql|mongodb|redis|amqp)://[^\s:/@]{2,}:)([^\s@/]{3,})(@[^\s/]{3,})"),
         mask=lambda m: m.group(1) + P.SECRET + m.group(3), value_group=2, confidence="high"),
    # bearer / authorization header — mask the token after the scheme
    Rule("bearer", re.compile(r"(?i)((?:authorization|x-api-key|x-auth-token|api-key)\s*[:=]\s*(?:Bearer |Basic |Token )?)([A-Za-z0-9._+/=-]{16,400})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="high",
         guard=lambda m: not is_secret_value(m.group(2), hash_ok=True)
         and len(m.group(2)) < 24),
    # (Telegram re-run audit: SECRET critical ×8 — «Authorization Bearer <token>» without
    # a colon, «Bearer <token>» as a line/message, a short token in curl -H): after the word
    # Bearer — a token of ≥8 chars with letters and digits; «Bearer token»/«Bearer $TOKEN» — no
    Rule("bearer_bare",
         re.compile(r"(?<![\w-])((?i:bearer)[ \t]+)([A-Za-z0-9._~+/-]{8,4096}=*)(?![\w.~+/=-])"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_bearer_bare_guard, needs=re.compile(r"(?i)bearer[ \t]")),

    # label = value (пароль/password/token/api-key/secret/ключ) — keep label.
    # (Telegram audit: a PHP/Ruby config in a message — «'password' => '…'»): after the label
    # a key quote and «=>» are allowed, and a single quote — also before «:»/«=». The JSON form «"key": "…"»
    # is NOT included: in .json this is the zone of the entropy_secret value pass (it is blocked by a human verdict —
    # tests/test_round: test_blocked_pseudo_rule_not_applied)
    # «API key: <hex>» — a label with a space (api[_ -]?key)
    # The value is up to 512 chars (was 120) — a key longer than 120 was split into
    # «SECRET_REDACTED» and an open tail (raw tg sources: a 129-char key under a label)
    Rule("label_value",
         re.compile(r"(?i)((?:парол\w*|password|passwd|pwd|passcode|секрет|secret|"
                    r"token|токен|api[_ -]?key|access[_-]?key|accesskey|client[_-]?secret|"
                    r"ключ)(?:[\"']?\s*=>|'?\s*[:=])\s*[`\"'*]{0,2})([^\s`\"',;<>\]}\\*]{5,512})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         # skip when the "value" is not high-entropy OR is already an anonymised
         # token (ORG_/PERSON_/…_REDACTED, incl. markdown-wrapped **ORG_2015**).
         # hash_ok=True: the label is the evidence, hex 32/40/64 under it is
         # a key (Mailchimp/GitHub PAT/VirusTotal), not a digest.
         # a value made entirely of our tokens («пароль: ORG_N/ORG_N») is not a secret
         guard=lambda m: not is_secret_value(m.group(2), hash_ok=True)
         or is_placeholder(m.group(2)) or _has_token(m.group(2))),
    # (Telegram re-run audit: «XX_PASSWORD_XXX=…», «XXXX_XXXXXXXXXX_KEY=…» in an .env block of
    # a message): an environment variable whose name has a segment PASSWORD/PASS/PWD/SECRET/
    # TOKEN/KEY — the value is a secret, however "simple" (the name is the evidence). Service
    # name tails (_PATH/_FILE/_URL/_TTL/_ID…), references to variables and placeholders — no.
    Rule("env_secret",
         # the name is one bounded run of [A-Z0-9_] (no nested repetition: «X_PASSWORD_…» ×10⁴
         # gave a quadratic search over splits); the guard checks the secret segment in the name
         # the value does not eat «\» (``\"`` in raw JSON — the quote stayed closing)
         re.compile(r"((?<![\w.$-])[A-Z][A-Z0-9_]{1,120}[^\S\n]*[:=][^\S\n]*[\"']?)"
                    r"([^\s\"'`" + _JS + r"]{6,512})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_env_secret_guard,
         needs=re.compile(r"_(?:PASSWORD|PASSWD|PASS|PWD|SECRET|TOKEN|KEY|APIKEY)(?![a-z])")),

    # --- passwords in PROSE («qaz123456» after the word «пароль» and
    # «email/Ab1cD2eF3gH4!» survived a release: label_value requires :=
    # and an entropy gate, while human passwords are low-entropy). A marker
    # right on the left = evidence; the value is ASCII without spaces, a letter AND
    # (a digit OR a special char); placeholders are not re-minted (idempotency).
    # The value class [!#-+\--:<-\[\]-~] is printable ASCII WITHOUT space/comma/
    # semicolon and WITHOUT " and \: in JSONL the value lives
    # inside a string, and the old class greedily ate the closing quote (»broken_
    # jsonl«) or the escape \n (»password: SECRET_REDACTEDвот« — the token welded to
    # Cyrillic), failing the structural gate. A password with literal "/\ in
    # escaped JSON is behind a backslash anyway — the match honestly stops at \.
    Rule("password_context",
         # (?<![/.]) — the label is not in a path/filename: «/password-reset2024» and
         # «docs/password-policy2021.pdf» are URL segments/file names, not
         # credentials; without the guard the «-» separator played the special class and
         # the rule tore the path. Prose «пароль …», «password: …», mid-word
         # «mypassword=…» match as before (a letter on the left, not «/» and not «.»).
         re.compile(r"(?i)(?<![/.])((?:парол\w*|пассворд\w*|password|passwd|pwd|"
                    r"пин-?код|код\s+доступа)\s*[:=\-–—/ ]\s*)"
                    r"([!#-+\--:<-\[\]-~]{6,24})(?![!#-+\--:<-\[\]-~])"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2,
         confidence="context",
         # letters only — a password only after a colon in CamelCase form ≥8
         # («Пароль: XxxXxxxxXxxx»); a value made entirely of our tokens is not a password
         guard=_pw_context_guard),
    # (Telegram audit: «Пароль от почты: …», «пароль от личного кабинета: …»): between the label and
    # the colon 1–3 words; the value has the same form as in password_context, not a link.
    # Up to 5 words («пароль от <service> <something> аккаунта: …»)
    Rule("password_after_words",
         re.compile(r"(?i)(?<![/.\w])((?:парол\w*|password|passwd)(?:[^\S\n]+[a-zа-яё]{1,15}){1,5}"
                    r"[^\S\n]*[:=][^\S\n]*)([!#-+\--:<-\[\]-~]{6,64})(?![!#-+\--:<-\[\]-~])"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=lambda m: (is_placeholder(m.group(2)) or "_REDACTED" in m.group(2)
                          or _has_token(m.group(2)) or "://" in m.group(2)
                          or not re.search(r"[A-Za-z]", m.group(2))
                          or not re.search(r"[\d!@#$%^&?*_\-]", m.group(2))
                          or _digest_ctx(m.group(1), m.group(2)))),
    # (Telegram re-run audit: «пароль такой Xxxx…!», «У нас пароль сейчас xXxx9xxx…»,
    # «Пароль 12345678» — without a colon, 0–2 Russian words between the label and the value): the value
    # only of a strong password form (Latin, a digit and a capital/special char, ≥8) or ≥8 digits —
    # an ordinary word is left alone
    Rule("password_inline",
         re.compile(r"(?i)(?<![/.\w])((?:парол\w*|password)(?:[^\S\n]+[а-яё]{2,15}){0,2}[^\S\n]+)"
                    r"([!#-+\--:<-\[\]-~]{8,64})(?![!#-+\--:<-\[\]-~])"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_pw_inline_guard, needs=re.compile(r"(?i)(?:парол|password)")),
    # (Telegram re-run audit: «(пароль ЖЖЖЖЖЖЖЖЖЖ)» under a disk link): a password in brackets
    # right after the label — any "word" (Cyrillic capitals too); a lowercase Russian word is an explanation
    Rule("password_paren",
         re.compile(r"(?i)(\((?:парол\w*|password|pass|pwd)[^\S\n]{0,3}[:=—–-]?[^\S\n]{0,3})"
                    r"([^\s()]{4,64})(?=[^\S\n]{0,3}\))"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_pw_paren_guard, needs=re.compile(r"(?i)\((?:парол|pass|pwd)")),
    # (Telegram audit: a Cookie header in a message — «PHPSESSID=…; BITRIX_SM_…»): the value of
    # a session cookie is live access, SECRET; the cookie name stays
    Rule("session_cookie",
         re.compile(r"(?i)(?<![\w.-])((?:PHPSESSID|JSESSIONID|ASP\.NET_SessionId|SESS[0-9a-f]{6,40}|"
                    r"sessionid|session_id|_session_id|connect\.sid|laravel_session|ci_session|"
                    r"ssid|sid)=)([^;\s\"'<>\\,&]{8,4096})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=lambda m: is_placeholder(m.group(2)) or "_REDACTED" in m.group(2)),
    # an email/password glue («ivan@example.com/Ab1cD2eF3gH4!»):
    # no detector saw the tail after «/» behind an email. The email itself
    # is minted by contacts (it runs after secrets) → EMAIL_REDACTED/SECRET.
    # PERF (critical of a global review): without a left boundary and limits
    # [\w.+-]+ gave O(n²) on contiguous megabyte runs (11 min CPU on
    # 560KB — reproduced); the lookbehind cuts the run interior in O(1),
    # RFC limits 64/255 bound the scan per position by a constant.
    Rule("password_email_glue",
         re.compile(r"(?<![\w.+-])"
                    r"((?:EMAIL_(?:REDACTED|\d+)"
                    r"|[\w.+-]{1,64}@[\w.-]{1,255}\.\w{2,})/)"
                    r"([!#-+\--:<-\[\]-~]{6,24})(?![!#-+\--:<-\[\]-~])"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2,
         confidence="context",
         guard=lambda m: is_placeholder(m.group(2))
         or not re.search(r"[A-Za-z]", m.group(2))
         or not re.search(r"[\d!@#$%^&?*]", m.group(2)),
         # group 1 is either an EMAIL_… token or an address with «@»
         needs=("EMAIL_", "@")),
    # REMOVED: risky_token — a label-free heuristic "token-like string"
    # («a1B2c3D4@eF5gH6»). The left boundary (?<![!-~]) excluded only ASCII,
    # so an ASCII secret GLUED to Cyrillic was masked with the token welded
    # to the word («сделатьSECRET_REDACTED») and broke JSON — the structural
    # gate failed a run. Label-free secrets in prose are the zone of the masking
    # agent (tools/llm_hotfix, the owner's design "hard to catch → AI,
    # not regexes"), NOT of a deterministic rule: too many false
    # boundaries. Label-anchored password_context/password_email_glue stay —
    # a marker/email on the left gives a reliable boundary without welding.

    # signed-url signature params (presigned S3, Discord CDN hm= etc.).
    # hm — the HMAC signature of cdn.discordapp.com (audit, critical: 334 live
    # signed attachments = live access to non-anonymized files).
    Rule("signed_url",
         re.compile(r"(?i)([?&](?:X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|"
                    r"signature|sig|access_token|api_key|AWSAccessKeyId|hm)=)"
                    r"([^&\s\"'" + _JS + r"]{12,})"),       # «\» never occurs in a URL
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="high"),
    # (Telegram re-run audit: a bot token in a Bot API link «…/bot<id>:<secret>/getUpdates» —
    # the telegram_bot signature does not see it right after «bot», the word boundary is wrong)
    Rule("telegram_bot_url",
         re.compile(r"(?i)(/bot)(\d{8,10}:AA[\w-]{30,})(?![\w-])"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="high",
         needs=re.compile(r"(?i)/bot\d{8}")),
    # (Telegram re-run audit: SECRET critical ×17 — «…/api/…/?api=<32 hex>&id=…» in curl,
    # «view.aspx?resid=…&authkey=!…» OneDrive, «email?token=…», «?rlkey=…» Dropbox): a key/token
    # as a link parameter.
    # The parameter name is the evidence; the value has a digit (or ≥16 in mixed case), is not
    # a template placeholder ({key}/<token>/${X}) and not words (auth=google_oauth2)
    Rule("url_secret_param",
         re.compile(r"(?i)([?&](?:api|api[_-]?key|apikey|key|token|access[_-]?key|auth|auth[_-]?key|"
                    r"authkey|rlkey|secret|client[_-]?secret|private[_-]?token|pass|password|passwd|pwd)=)"
                    r"([^&\s\"'#<>\\^]{8,2048})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_url_secret_guard,
         needs=re.compile(r"(?i)[?&](?:api|key|token|access|auth|rlkey|secret|client|private|pass|pwd)")),

    # External document IDs in a path (audit, critical: 58 live Google
    # Docs/Sheets + Figma keys = direct access to sources). Minted as a
    # RESOURCE token (it is a resource link, not a password); idempotent —
    # the token does not match the value class.
    # The path may have an account segment «/u/0/» («/document/u/0/d/<id>»,
    # «/spreadsheets/u/1/d/<id>») and «/d/e/<id>» of a published copy — previously
    # such ids slipped through (a whole Google doc id in the text survived
    # a round; the gdocs corpus has 18 links with /u/N/).
    Rule("gdoc_file_id",
         re.compile(r"(/(?:spreadsheets|document|presentation|file|forms|drawings)"
                    r"(?:/u/\d{1,2})?/d/(?:e/)?)"
                    r"([A-Za-z0-9_-]{20,})"),
         mask=lambda m: m.group(1) + mint.tag("RESOURCE", m.group(2)),
         value_group=2, confidence="high",
         needs="/d/"),
    # The same id in a query parameter («docs.google.com/open?id=<id>»,
    # «drive.google.com/uc?export=download&id=<id>» — 40 links in the gdocs
    # corpus) and a Drive folder id («drive.google.com/drive/folders/<id>»).
    # The Google host is mandatory: a bare «id=<20+ chars>» is not evidence.
    Rule("gdoc_open_id",
         re.compile(r"((?:docs|drive)\.google\.com/(?:a/[\w.-]{1,60}/)?"
                    r"(?:u/\d{1,2}/)?(?:open|uc|file)\?"
                    r"(?:[A-Za-z0-9_=%.-]{1,40}&(?:amp;)?){0,4}id=)"
                    r"([A-Za-z0-9_-]{20,})"),
         mask=lambda m: m.group(1) + mint.tag("RESOURCE", m.group(2)),
         value_group=2, confidence="high",
         needs="google.com/"),
    # a mobile link «drive/mobile/folders/<id>» and a host already masked by a previous
    # round («RESOURCE_N/drive/folders/<id>» in old outputs): the host token — drive/docs
    Rule("gdrive_folder_id",
         re.compile(r"((?:(?:docs|drive)\.google\.com|RESOURCE_\d{1,12})/(?:drive/)?(?:u/\d{1,2}/)?"
                    r"(?:mobile/)?folders/)([A-Za-z0-9_-]{20,})"),
         mask=lambda m: m.group(1) + mint.tag("RESOURCE", m.group(2)),
         value_group=2, confidence="high",
         guard=lambda m: _has_token(m.group(2))
         or _tok_host_bad(m, "drive.google.com", "docs.google.com"),
         needs="folders/"),
    # (proofreading of tg output: shared links — the host is closed by a token, the path is
    # open): Dropbox «…/scl/fo|fi/<id>/<id2>?rlkey=…» and Yandex Disk «…/i|d/<id>»,
    # «…/edit/d/<id>» — the link id → RESOURCE, as for Google documents; the second segment
    # of a Dropbox folder is also an id, a file name (with an extension) stays. The host is the service or its token.
    Rule("dropbox_share_id",
         re.compile(r"(?i)((?:www\.)?(?:dropbox\.com|RESOURCE_\d{1,12})/(?:scl/f[io]|s|sh)/)"
                    r"([A-Za-z0-9_-]{6,80})(/[A-Za-z0-9_-]{16,80}(?![\w.-]))?(?![\w-])"),
         mask=_dropbox_mask, value_group=2, confidence="high",
         guard=lambda m: _has_token(m.group(2))
         or _tok_host_bad(m, "dropbox.com", "www.dropbox.com"),
         needs=re.compile(r"(?i)/(?:scl/f[io]|s|sh)/[A-Za-z0-9_-]{6}")),
    Rule("yadisk_share_id",
         re.compile(r"(?i)((?:disk\.yandex\.(?:ru|com|by|kz|ua)|disk\.360\.yandex\.ru|yadi\.sk|"
                    r"RESOURCE_\d{1,12})/(?:edit/)?[id]/)([A-Za-z0-9_-]{6,80})(?![\w-])"),
         mask=lambda m: m.group(1) + mint.tag("RESOURCE", m.group(2)),
         value_group=2, confidence="high",
         guard=lambda m: _has_token(m.group(2))
         or _tok_host_bad(m, "disk.yandex.ru", "disk.yandex.com", "disk.yandex.by", "disk.yandex.kz",
                          "disk.yandex.ua", "disk.360.yandex.ru", "yadi.sk"),
         needs=re.compile(r"(?i)/(?:edit/)?[id]/[A-Za-z0-9_-]{6}")),
    Rule("figma_file_id",
         re.compile(r"(figma\.com/(?:file|design|proto|board)/)"
                    r"([A-Za-z0-9]{15,})"),
         mask=lambda m: m.group(1) + mint.tag("RESOURCE", m.group(2)),
         value_group=2, confidence="high"),
    # («…?node-id=1-2&p=f&t=<16 letters-digits>-0» in "share" links):
    # t= of a Figma "share" link is a sharing identifier (it ties the link to whoever
    # copied it) → DOC_ID. The host is figma.com or its RESOURCE token (the next round over the output)
    Rule("figma_share_param",
         re.compile(r"(?i)((?:figma\.com|RESOURCE_\d{1,12})/(?:file|design|proto|board)/"
                    r"[^\s\"'<>]{0,300}?[?&]t=)([A-Za-z0-9]{10,32})(?=-\d|[&#\s\"'<>)]|$)"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", m.group(2)),
         value_group=2, confidence="high", guard=_figma_host_guard,
         needs=re.compile(r"(?i)[?&]t=[A-Za-z0-9]{10}")),

    # (meeting invitations in messages): a dial-in PIN
    # («PIN: 999 999 999 9999#», «PIN-код: …»), pin= in a tel.meet link, a Zoom passcode and a Zoom
    # recording by link — access to a meeting, SECRET. The meeting code and the conference number (DOC_ID) —
    # detectors/discord_ids.
    Rule("meeting_pin",
         re.compile(r"(?i)((?<![\w])(?:PIN|ПИН)(?:[- ]?(?:код|code))?[^\S\n]{0,3}:[^\S\n]{0,3})"
                    r"(\d{3}(?: ?\d{3}){2}(?: ?\d{3,4})?|\d{9,16})(?!\d)"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         needs=re.compile(r"(?i)(?:pin|пин)")),
    Rule("meeting_link_pin",
         re.compile(r"(?i)(/(?:[a-z]{3}-[a-z]{4}-[a-z]{3}|DOC_ID_\d{1,12})\?(?:[^\s&#\"'<>]{0,100}&)?"
                    r"pin=)(\d{6,16})(?!\d)"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="high",
         needs=re.compile(r"(?i)[?&]pin=\d")),
    # The value class does not eat the closing quote of a JSON string
    # («passcode: a12&bc3d"» → broken_jsonl) and does not pull a tail through the «\n» escape (boundary.JSON_SAFE)
    Rule("meeting_passcode",
         re.compile(r"(?i)((?<![\w])(?:код доступа|код конференции|пароль конференции|пароль встречи|"
                    r"код встречи|passcode|meeting password|access code)[^\S\n]{0,3}:[^\S\n]{0,3})"
                    r"([^\s,;" + _JS + r"]{4,32})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_passcode_guard,
         needs=re.compile(r"(?i)(?:код доступа|код конференции|пароль конференции|пароль встречи|"
                          r"код встречи|passcode|meeting password|access code)")),
    Rule("zoom_recording_link",
         re.compile(r"(?i)(zoom\.us/rec/(?:share|play)/)([\w.~-]{10,512})"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="high",
         needs=re.compile(r"(?i)zoom\.us/rec/")),

    # a hex key 32/40/64 next to a secret label; without a label the same string is a git SHA/
    # digest. Previously only 40 and the labels token|api-key|secret|
    # ключ|auth|bearer — «api_key=<32hex>» (Mailchimp/Trello), «secret=<64hex>»
    # (VirusTotal), «"api_key": "<40hex>"» (GitHub classic PAT in a JSON vault)
    # went out open. Now the labels are as in label_value + auth/bearer, lengths
    # {32,40,64}. Also catches the JSON key «"api_key": "<hex>"» (label_value requires
    # the label right before «:=», the quote gets in the way) and prose «токен вот: <hex>».
    # Guard: the gap between the label and the hex ends with a digest marker
    # (_DIGEST_TAIL) or it is a path segment «…/<hex>» (URL/file) → not a secret.
    # The rule name is historical: verdicts/tiers live by the name (decisions,
    # rule_tiers) — do not rename.
    Rule("hex40_labeled",
         re.compile(r"(?i)(?:token|токен|api[_ -]?key|access[_-]?key|accesskey|"
                    r"password|парол\w*|passwd|pwd|passcode|secret|секрет|ключ|"
                    r"auth|bearer)"
                    r"([^\n]{0,60}?)(?:(?<=\\n)|(?<![A-Za-z0-9\\]))"
                    r"([0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{32})(?![A-Za-z0-9])"),
         mask=lambda m: m.group(0)[:-len(m.group(2))] + P.SECRET,
         value_group=2, confidence="context",
         guard=lambda m: bool(_DIGEST_TAIL.search(m.group(1)))
         or m.group(1).endswith("/")),

    # (Telegram audit, critical SECRET: long keys/tokens WITHOUT a label inside the text —
    # 17 of the confirmed ones): a standalone token of 24–200 chars of the base64/url alphabet with
    # all three classes (lowercase+capitals+digits) and the entropy form of a secret. Not part
    # of a link/path/email/parameter (neighbours «/ . @ = ? & % :»), bare digests and printable
    # base64 are left alone (_bare_key_guard).
    Rule("bare_key_token",
         # boundaries by \w (Unicode: Cyrillic too): a key glued to a Russian word does not
         # match at all, rather than being masked with the token welded to the word (the reason
         # risky_token was removed, see above)
         re.compile(r"(?<![\w+/=.@:?&%\\-])([A-Za-z0-9][A-Za-z0-9+/_-]{23,199}={0,2})"
                    r"(?![\w+/=.@-])"),
         mask=lambda m: P.SECRET, value_group=1, confidence="context",
         guard=_bare_key_guard,
         needs=re.compile(r"[A-Za-z0-9+/_-]{24}")),

    # (Telegram re-run audit: SECRET/HANDLE critical — an access block
    # «<link or label>\n<login>\n<password>»: «https://…/admin/», «Логин и пароль это:», «Доступы»
    # and two lines below): the login line → HANDLE (the same seed as for a nickname), the password on the next
    # line → SECRET. Only a strong password form (_strong_pw); it comes BEFORE password_next_line —
    # otherwise that one would take the login line for a password. The lines are inside a message or adjacent messages
    # (the slot separator). A value made entirely of our tokens (a repeated round) is not a password.
    # The match starts at the label (not at the line start: the lazy prefix «^[^\n]{0,300}?» gave
    # a quadratic search on every line), the line tail after the label — up to 300 chars.
    Rule("credential_login_password",
         re.compile(r"(?i)((?:://|логин|login|парол|password|доступ|креды|credentials|учетк|учётк|"
                    r"админк|admin|вход|sign[ -]?in|аккаунт|account)[^\n]{0,300}\n"
                    r"(?:{2}\n)?[^\S\n]*)([A-Za-z0-9][A-Za-z0-9._@+-]{2,63})"
                    r"([^\S\n]*\n(?:{2}\n)?[^\S\n]*)([^\s]{6,64})(?=[^\S\n]*(?:\n|\Z))"),
         mask=_cred_login_password_mask, value_group=4, confidence="context",
         guard=_cred_login_guard,
         needs=re.compile(r"(?i)(?:://|логин|login|парол|password|доступ|креды|credentials|учетк|"
                          r"учётк|админк|admin|вход|sign[ -]?in|аккаунт|account)")),

    # (Telegram audit: the password is sent on the NEXT line/message after the label — «Пароль от
    # админки:» and the password itself below; in the Telegram structural profile message values go
    # on separate lines via the slot separator «\n\n»): the value is one "word form"
    # without spaces occupying the whole line. The label and the separator stay as they were (the mask covers only the value).
    # A key/token label («apiKey» as a message, the key as the next one) — the same pair (_PW_LABELS)
    # Our constant «SECRET_REDACTED» on the line above is not a «secret» label
    # (a password as a separate message without a label became SECRET, and the next message was judged
    # as "a line under a label" — «MacBookPro2021» after a password went to SECRET)
    Rule("password_next_line",
         re.compile(r"(?i)((?:" + _PW_LABELS + r")(?!_REDACTED)[^\n]{0,40}\n"
                    r"(?:\ue000{2}\n)?[^\S\n]*)([^\s\ue000]{8,64})(?=[^\S\n]*(?:\n|\Z))"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=_pw_next_guard),
    # (Telegram re-run audit: SECRET critical ×5+5 — an access list
    # «<email>\n<password>»): a line made entirely of an email address (or already EMAIL_REDACTED), the next
    # line/message is one "word" of password form (_cred_pw) → SECRET. A phone, a name, a link,
    # a word without digits, our token — not a password.
    Rule("credential_after_email",
         re.compile(r"(?m)^([^\S\n]*(?:[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){0,5}\.[A-Za-z]{2,24}"
                    r"|EMAIL_REDACTED)[^\S\n]*\n(?:\ue000{2}\n)?[^\S\n]*)"
                    r"([^\s\ue000]{6,64})(?=[^\S\n]*(?:\n|\Z))"),
         mask=lambda m: m.group(1) + P.SECRET, value_group=2, confidence="context",
         guard=lambda m: not _cred_pw(m.group(2)), needs=("@", "EMAIL_REDACTED")),
    # (Telegram re-run audit: SECRET critical ×8+10 — a password as a SEPARATE
    # message without a label: «Password1!», «Xxxxxx@?xxx1234», a random string): a projection slot
    # of the structural profile (a message) made entirely of one "word" of password form (_lone_pw_shape).
    # Only a slot between «» separators (or a text of one word): a line of an ordinary file
    # — no. Code/JSON/links/email/versions/our token — left alone.
    Rule("lone_password_message",
         re.compile(r"(?:\A|(?<=\ue000\n))([^\S\n]*)([^\s\ue000{}\[\]()<>\"'\\=&]{8,64})([^\S\n]*)"
                    r"(?=\n\ue000|\Z)"),
         mask=lambda m: m.group(1) + P.SECRET + m.group(3), value_group=2, confidence="context",
         guard=lambda m: not _lone_pw_shape(m.group(2)),
         # prefilter: a slot of one "word" of 8–64 chars — a superset of the rule's matches
         needs=re.compile(r"(?:\A|\ue000\n)[^\S\n]*[^\s\ue000]{8,64}[^\S\n]*(?:\n\ue000|\Z)")),
]
