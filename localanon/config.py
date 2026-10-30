"""Run config (TOML). The schema is shared by all stages: stages only read it.

Created by ``localanon init <path>``: a template with a fresh salt. The data owner generates and
keeps the salt; only its fingerprint goes to the manifest.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

# Hosts whose links are replaced with the RESOURCE_REDACTED placeholder as a whole. Subdomains are
# included: "docs.google.com" covers both "docs.google.com/..." and "xxx.docs.google.com".
DEFAULT_DELETE_HOSTS = [
    "figma.com",
    "docs.google.com", "drive.google.com",
    "disk.yandex.ru", "disk.yandex.com", "disk.yandex.by", "disk.yandex.kz", "disk.yandex.ua",
    "disk.360.yandex.ru", "yadi.sk",
    "notion.so", "notion.site",
    "miro.com",
    "dropbox.com", "dropboxusercontent.com", "db.tt",
    "sharepoint.com",
    "atlassian.net",
    # other external sources (sheets/slides live under docs.google.com)
    "sheets.google.com", "slides.google.com", "drive.usercontent.google.com",
    "docs.yandex.ru", "docs.360.yandex.ru",
    "onedrive.com", "onedrive.live.com", "1drv.ms",
    "trello.com", "box.com", "boxcloud.com", "airtable.com", "lucid.app", "whimsical.com",
    "wetransfer.com", "we.tl", "cloud.mail.ru", "mega.nz", "mega.io",
]
# Hosts by name prefix: a company's own Confluence ("confluence.company.ru").
DEFAULT_DELETE_HOST_PREFIXES = ["confluence."]

# Document kinds the classifier tells apart (localanon/docclass.py).
DOC_KINDS = ["contract", "power_of_attorney", "supplementary_agreement", "act", "invoice",
             "upd", "waybill",
             # letters about delivery deadlines and penalties: claim, guarantee letter,
             # disagreement protocol
             "claim", "guarantee_letter", "disagreement_protocol"]
# Removed by default: contracts, supplementary agreements, powers of attorney, acts, invoices
# (and UPD, a combined act and invoice), plus claims, guarantee letters and disagreement
# protocols (letters about delivery deadlines and penalties). Other kinds (waybill) stay, are
# anonymized like any document and are marked in the report.
DEFAULT_DELETE_KINDS = ["contract", "supplementary_agreement", "power_of_attorney", "act",
                        "invoice", "upd", "claim", "guarantee_letter", "disagreement_protocol"]
MSG_CATEGORIES = ["dev", "commercial", "personal", "other"]
# commercial covers delivery deadlines as obligations, penalties, money;
# personal — private matters unrelated to development. dev and other stay.
DEFAULT_DROP_CATEGORIES = ["commercial", "personal"]
IMAGE_MODES = ("keep", "avatars", "delete")


@dataclass
class Project:
    name: str = ""
    salt: str = ""                       # 64 hex; required


@dataclass
class Executor:
    """Contractor: everything about it is replaced with a placeholder (names → ORG token, sites →
    RESOURCE)."""
    names: list[str] = field(default_factory=list)          # «Ромашка», «Romashka»
    legal_names: list[str] = field(default_factory=list)    # «ООО «Ромашка»»
    brands: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)        # romashka.ru
    email_domains: list[str] = field(default_factory=list)
    inn: list[str] = field(default_factory=list)            # INN, OGRN, KPP (Russian company registration numbers)
    extra_terms: list[str] = field(default_factory=list)


@dataclass
class Links:
    delete_hosts: list[str] = field(default_factory=lambda: list(DEFAULT_DELETE_HOSTS))
    delete_host_prefixes: list[str] = field(
        default_factory=lambda: list(DEFAULT_DELETE_HOST_PREFIXES))
    mask_other_domains: bool = True      # other links: domain → RESOURCE_…, the path stays


@dataclass
class Documents:
    delete_kinds: list[str] = field(default_factory=lambda: list(DEFAULT_DELETE_KINDS))
    borderline: str = "delete"           # delete (doubtful match is removed, flagged) | keep
    header_images: str = "delete"        # delete (logos of headers/footers/slide masters) | keep
    restore: list[str] = field(default_factory=list)   # relative paths: do not remove


@dataclass
class Chats:
    exclude: list[str] = field(default_factory=list)       # chat names or ids — remove whole
    include_only: list[str] = field(default_factory=list)  # non-empty — keep only these
    drop_personal_chats: bool = False


@dataclass
class Messages:
    # auto (model if installed, else rules) | rules | model | off (nothing is dropped)
    filter: str = "auto"
    drop_categories: list[str] = field(default_factory=lambda: list(DEFAULT_DROP_CATEGORIES))
    model_dir: str = ""                  # filter model directory (ONNX + tokenizer)
    # threshold on p(delete) = sum of drop_categories probabilities: "auto" — the tier's own
    # (index.json calibration) | a number in (0, 1). 0.5 (the old default) is read as "auto".
    threshold: float | str = "auto"


@dataclass
class Media:
    images: str = "avatars"              # avatars (profile photos only) | keep | delete (all)
    transcribe: bool = True              # transcript next to the media file; the file is removed
    asr_model: str = "medium"            # tiny | base | small | medium | large-v3
    asr_device: str = "auto"             # auto (by [compute] and hardware) | cpu | cuda | cuda:N
    diarize: bool = False


@dataclass
class Ner:
    model: str = "auto"                  # auto (by hardware: hw.plan_nermodel) | off | tier from models/ner (tiny2, base, large)
    natasha: bool | str = "auto"         # auto — only without the model | true — together with the model | false
    orgs: bool = True                    # organizations the name model finds → ORG (a model with the ORG class)
    gliner: bool = False                 # gliner-x-large on top of the model (extra gliner)
    gliner_threshold: float = 0.97       # at 0.5 — hundreds of false hits per 1000 messages
    device: str = "auto"                 # auto | cpu | cuda | cuda:N


@dataclass
class Compute:
    """Hardware (localanon/hw.py): every "0" / "auto" / empty — chosen by the detected hardware."""
    devices: str = "auto"                # auto | cpu | list: "cuda:0,cuda:1", "rocm:0", "dml:0", "cpu"
    cpu_workers: int = 0                 # worker processes per CPU stage; 0 — auto
    threads_per_worker: int = 0          # threads inside a worker; 0 — auto, no oversubscription
    gpu_workers_per_device: int = 0      # model instances per GPU; 0 — auto by memory
    batch_size: int = 0                  # 0 — auto by stage and device
    reserve_cores: int = -1              # cores left to the system; -1 — auto (1–2)
    max_ram_frac: float = 0.6            # share of available RAM for workers
    max_gpu_mem_frac: float = 0.8        # share of free GPU memory for models
    time_budget_hours: float = 0.0       # time budget for the message filter; 0 — no budget
    msgmodel_tier: str = ""              # filter model tier from index.json; empty — auto
    asr_auto: bool = True                # media.asr_model = "medium" (default) or "auto" — by hardware


@dataclass
class Output:
    archive_name: str = ""               # empty — <input name>_clean.zip
    report: bool = True


@dataclass
class Config:
    project: Project = field(default_factory=Project)
    executor: Executor = field(default_factory=Executor)
    links: Links = field(default_factory=Links)
    documents: Documents = field(default_factory=Documents)
    chats: Chats = field(default_factory=Chats)
    messages: Messages = field(default_factory=Messages)
    media: Media = field(default_factory=Media)
    ner: Ner = field(default_factory=Ner)
    compute: Compute = field(default_factory=Compute)
    output: Output = field(default_factory=Output)

    def salt_fingerprint(self) -> str:
        """Salt fingerprint for the manifest: it shows "same salt" but not the salt itself."""
        return hashlib.sha256(("localanon:" + self.project.salt).encode()).hexdigest()[:16]


_SECTIONS = {f.name: f.type for f in fields(Config)}


def _section(cls, raw: dict, name: str):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"[{name}]: unknown keys {', '.join(unknown)}")
    return cls(**raw)


def load(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"unknown config sections: {', '.join(unknown)}")
    if "rubert" in (raw.get("ner") or {}):
        raise ValueError("[ner].rubert is gone: ruBERT collection3 is replaced by the name model "
                         "([ner].model, directory models/ner) — delete the rubert line from the config")
    kw = {}
    for f in fields(Config):
        cls = type(getattr(Config(), f.name))
        kw[f.name] = _section(cls, raw.get(f.name, {}), f.name)
    cfg = Config(**kw)
    s = cfg.project.salt
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s.lower()):
        raise ValueError("[project].salt: a salt of 64 hex digits is required (localanon init creates it)")
    if cfg.documents.borderline not in ("delete", "keep"):
        raise ValueError("[documents].borderline: delete or keep")
    if cfg.documents.header_images not in ("delete", "keep"):
        raise ValueError("[documents].header_images: delete or keep")
    if cfg.media.images not in IMAGE_MODES:
        raise ValueError("[media].images: avatars, keep or delete")
    bad = sorted(set(cfg.documents.delete_kinds) - set(DOC_KINDS))
    if bad:
        raise ValueError(f"[documents].delete_kinds: unknown kinds {', '.join(bad)}; "
                         f"allowed: {', '.join(DOC_KINDS)}")
    bad = sorted(set(cfg.messages.drop_categories) - set(MSG_CATEGORIES))
    if bad:
        raise ValueError(f"[messages].drop_categories: unknown categories {', '.join(bad)}")
    if cfg.messages.filter not in ("auto", "rules", "model", "off"):
        raise ValueError("[messages].filter: auto, rules, model or off")
    thr = cfg.messages.threshold
    if not (thr == "auto" or (not isinstance(thr, bool) and isinstance(thr, (int, float)) and 0 < thr < 1)):
        raise ValueError('[messages].threshold: "auto" or a number in (0, 1)')
    _check_compute(cfg)
    _check_ner(cfg)
    return cfg


_TIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _check_ner(cfg: Config) -> None:
    n = cfg.ner
    if not isinstance(n.model, str) or not _TIER_RE.match(n.model):
        raise ValueError('[ner].model: "auto", "off" or a tier name from models/ner ("base", "large")')
    if n.natasha not in (True, False, "auto"):
        raise ValueError('[ner].natasha: true, false or "auto"')
    if not isinstance(n.orgs, bool):
        raise ValueError("[ner].orgs: true or false")
    if not isinstance(n.gliner, bool):
        raise ValueError("[ner].gliner: true or false")
    if isinstance(n.gliner_threshold, bool) or not isinstance(n.gliner_threshold, (int, float)) \
            or not 0 < n.gliner_threshold < 1:
        raise ValueError("[ner].gliner_threshold: a number in (0, 1)")
    if not isinstance(n.device, str) or not _DEV_RE.match(n.device.strip().lower()):
        raise ValueError("[ner].device: auto, cpu, cuda or cuda:N")


_DEV_RE = re.compile(r"^(auto|cpu|cuda|rocm|dml)(:\d+)?$")


def _check_compute(cfg: Config) -> None:
    c = cfg.compute
    items = [x.strip().lower() for x in c.devices.split(",") if x.strip()]
    if not items or any(not _DEV_RE.match(x) for x in items) or (len(items) > 1 and "auto" in items):
        raise ValueError("[compute].devices: auto, cpu or a list like \"cuda:0,cuda:1,cpu\"")
    if not 0 < c.max_ram_frac <= 1 or not 0 < c.max_gpu_mem_frac <= 1:
        raise ValueError("[compute]: max_ram_frac and max_gpu_mem_frac are fractions (0, 1]")
    if min(c.cpu_workers, c.threads_per_worker, c.gpu_workers_per_device, c.batch_size) < 0 \
            or c.reserve_cores < -1 or c.time_budget_hours < 0:
        raise ValueError("[compute]: negative values are not allowed")


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return '"' + str(v) + '"'


def _toml_list(xs: list[str]) -> str:
    return "[" + ", ".join('"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"' for x in xs) + "]"


def template(name: str = "") -> str:
    """Text of a new config with a fresh salt."""
    c = Config()
    return f'''# LocalAnonPipeline config. The salt below is the data owner's secret: do not share it.
#
# The local run removes what the brief lists: contracts, powers of attorney, supplementary
# agreements, acts, invoices; messages about delivery deadlines, penalties and private matters;
# avatars and profile photos; logos in document headers and footers; links to external resources
# (figma, google, yandex…). Full names, emails, logins, phones, domains and everything about the
# contractor are replaced with placeholders. Whatever the script could not process stays as is
# with a mark in report.html — our main pipeline cleans it up.
[project]
name = "{name}"
salt = "{secrets.token_hex(32)}"

# Contractor: everything listed is replaced with a placeholder in text, file names and links.
[executor]
names = []          # e.g. ["Ромашка", "Romashka"]
legal_names = []    # e.g. ["ООО «Ромашка»"]
brands = []
domains = []        # e.g. ["romashka.ru"]
email_domains = []
inn = []
extra_terms = []

# Links do not take the message with them: a URL to a host from delete_hosts (external storage
# and boards) is replaced with the RESOURCE_REDACTED placeholder as a whole; for other links
# (github, gitlab…) the domain → RESOURCE_…, the path stays.
[links]
delete_hosts = {_toml_list(c.links.delete_hosts)}
delete_host_prefixes = {_toml_list(c.links.delete_host_prefixes)}
mask_other_domains = true

# Documents of these kinds are removed. All kinds: {", ".join(DOC_KINDS)}.
# Other kinds (waybill) and other documents (specification, commercial offer) stay and are
# anonymized, the kind is marked in the report. Unreadable and broken documents, scans without
# OCR, old formats without LibreOffice are not removed — they stay as is with the
# "not anonymized" mark.
[documents]
delete_kinds = {_toml_list(c.documents.delete_kinds)}
borderline = "{c.documents.borderline}"   # doubtful match: delete (remove, with the "check" mark) or keep
header_images = "{c.documents.header_images}"   # images in headers/footers and slide masters (logos), in PDF — repeated in the header/footer band: delete or keep
restore = []            # paths not to remove even if they look like a removed kind

# Whole chats: by default all stay, messages are filtered by [messages].
[chats]
exclude = []            # chats to remove whole (name or id)
include_only = []       # if non-empty — keep only these chats
drop_personal_chats = false

# Message filter by meaning: the drop_categories are removed (categories:
# {", ".join(MSG_CATEGORIES)}; commercial — money, delivery deadlines, penalties; personal —
# private matters unrelated to development). Development (dev) and service (other) messages stay.
[messages]
filter = "{c.messages.filter}"         # auto (model if installed, else rules) | rules | model | off
drop_categories = {_toml_list(c.messages.drop_categories)}
model_dir = ""
# Delete a message when the model's probability of drop_categories (summed) reaches the threshold.
# "auto" — the threshold chosen for the model (index.json); a number in (0, 1) — your own: lower
# deletes more, higher keeps more. 0.5 from older configs means "auto".
threshold = "{c.messages.threshold}"

[media]
images = "{c.media.images}"      # avatars — remove avatars and profile photos | keep | delete — all images
transcribe = true       # audio and video: the transcript stays, the media file is removed
asr_model = "medium"
asr_device = "auto"     # auto | cpu | cuda | cuda:N
diarize = false

[ner]
# Name model from models/ner. auto: large on an NVIDIA card, base on CPU, tiny2 on a slow CPU (speed probe); off: none.
model = "{c.ner.model}"
natasha = {_toml_scalar(c.ner.natasha)}        # "auto" — Natasha only without the model; true — together with the model (finds a little more, hours of CPU on a large corpus); false — none
orgs = {_toml_scalar(c.ner.orgs)}             # organizations the name model finds → ORG tokens (rules like [executor]); false — names only
gliner = false          # gliner-x-large on top of the model, needs an NVIDIA card: .venv/bin/python -m pip install -e ".[gliner]",
                        # then .venv/bin/python -m localanon.fetch --gliner (Windows: .venv\\Scripts\\python.exe)
gliner_threshold = {c.ner.gliner_threshold}  # below 0.9 — many false names
device = "auto"         # auto | cpu | cuda | cuda:N

# Hardware: found automatically by default (localanon hw shows what was found and how stages are laid out).
[compute]
devices = "auto"        # auto | cpu | "cuda:0,cuda:1" | "dml:0" …
cpu_workers = 0         # 0 — auto
threads_per_worker = 0  # 0 — auto
gpu_workers_per_device = 0
batch_size = 0
reserve_cores = -1      # -1 — auto (1–2 cores for the system)
max_ram_frac = 0.6
max_gpu_mem_frac = 0.8
time_budget_hours = 0   # time budget for the message filter; 0 — no budget
msgmodel_tier = ""      # empty — auto (models/msgfilter/index.json)
asr_auto = true         # [media].asr_model = "medium" (default) or "auto" — model by hardware

[output]
archive_name = ""
report = true
'''
