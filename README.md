# LocalAnonPipeline — local pre-cleaning of project exports

The script runs on your computer, on the original export (chats, tasks, documents). It is the first
stage of anonymization: what it is not sure about is finished by us later.

What it removes:
- contracts, supplementary agreements, powers of attorney, acts, invoices and UPD (a combined act
  and invoice); claims, guarantee letters and disagreement protocols;
- messages about money, deadlines and penalties, personal messages and other personal information
  that is not about development (a local message filter model decides, see "Message filter");
  messages about development stay;
- the contractor's logos: pictures in document headers and footers and in slide masters;
- profile photos and avatars;
- audio and video: a text transcript stays in the archive instead; a recording that could not be
  transcribed is removed and marked "check";
- everything you list about the contractor in `[executor]`: company names, websites, e-mail domains.

What it replaces:
- names, e-mails, phone numbers, logins in social networks and elsewhere, domains — with tokens;
- links to Figma, Google Drive, Yandex Disk, Notion, Miro and other external resources — with the
  placeholder `RESOURCE_REDACTED`; in links to GitHub, GitLab and other sites the domain is replaced
  with a token and the path stays.

Screenshots and other pictures, tasks, specifications and other documents stay and are anonymized.

A file that cannot be read or processed (a broken document, a password-protected archive, a file of an
unknown kind) stays as it is and is marked "not anonymized" in the report; we finish it later. A
doubtful contract or invoice is removed and marked "check" — it can be brought back (see "Restoring a
file"). While processing, the script **never goes online**: all models are local, the network is used
only by the first start (installation).

The report `report.html` lists what was removed and what was kept with a mark. It stays with you.

## Quick start

You need:
- **Python 3.11 or newer** ([python.org](https://www.python.org/downloads/); on Windows tick
  "Add python.exe to PATH");
- **Git**;
- the **models archive** `localanon-models-<date>.zip` — we send it separately from the repository;
- about 12 GB of free disk space (17 GB with an NVIDIA card): the models archive and the unpacked
  models, the Python environment and the speech recognition model.

```bash
git clone <repository URL> localanon
cd localanon
# put the models archive into this folder as it is (do not unpack it), then:
./run.sh /path/to/export             # Linux, macOS
make run IN=/path/to/export          # the same with make; OUT=<folder> sets the output folder
```

```powershell
git clone <repository URL> localanon
cd localanon
# put the models archive into this folder as it is (do not unpack it), then:
powershell -ExecutionPolicy Bypass -File run.ps1 C:\Exports\Project       # Windows
```

The input is a folder or an archive (`.zip`, `.tar.gz`) with the export. The second argument is the
output folder; by default it is `out/<export name>` inside the repository folder.

The **first start** unpacks the models archive into `models/` (each file is checked against
`models.lock.json`; afterwards the archive is no longer needed and can be deleted) and sets everything up
in `.venv` inside the repository folder (this takes a while and needs the network):
- the package with all its dependencies;
- `pyahocorasick`, an optional accelerator of the anonymization engine; if it cannot be installed (no
  wheel for a fresh Python), the script prints one line and goes on — the engine works without it,
  only slower;
- with an NVIDIA card: `onnxruntime-gpu` with the CUDA libraries, then a check that the card really
  computes (`localanon hw --probe-cuda`); if the check fails, processing runs on CPU;
- the speech recognition model (whisper) of a size that suits this machine.

Then it creates the settings file `localanon.toml` (see below) and processes the export. Later starts
skip the installation (it is repeated only when `pyproject.toml` changes, e.g. after `git pull`, or when
an NVIDIA card appears or disappears) and run fully offline. If the hardware changed and this machine now
needs another whisper size (more memory, another card), the script downloads just that model before
processing; without the network it warns and uses the model already downloaded.

If the models are not unpacked and there is no archive in the repository folder, the script stops and
names the archive it needs. The same happens after an update (`git pull`) that comes with new models: put
the new archive into the folder and start again — only the changed files are unpacked.

Options of the start scripts:
- `LOCALANON_GPU=no ./run.sh ...` (PowerShell: `$env:LOCALANON_GPU = "no"`) — do not install
  `onnxruntime-gpu` even if there is a card (`yes` — always install it);
- `PYTHON=python3.12 ./run.sh ...` — the interpreter for `.venv` (Linux, macOS; on Windows the
  `py` launcher picks 3.13, 3.12 or 3.11).

To start from scratch (e.g. if the installation was interrupted or broken), delete `.venv` and run the
script again.

## Results

In the output folder:

| File | What it is | For whom |
|---|---|---|
| `<input name>_clean.zip` | the cleaned and pseudonymized export | hand over to us |
| `local_manifest.json` | versions, salt fingerprint, counters — no paths and no contents | hand over to us |
| `report.html` | what was removed and what was kept with a mark (paths, reasons), chat list, notes, check problems | stays with you |

**Hand over only `_clean.zip` and `local_manifest.json`.** Do not hand over the report, the settings
file with the salt or the original export.

Exit codes: `0` — the archive is built (also when the checks at the end found problems: they are
listed in the report); `1` — a stage failed (its name is in the message); `2` — an error in the
settings or in the input (the message says what).

## Settings

`localanon.toml` in the repository folder is created by the first start with a **fresh salt**. The
salt is your secret: tokens are derived from it, so the file is never handed over (only its
fingerprint goes to the manifest). Do not change the salt between runs of the same export. The file
is ignored by git, so `git pull` does not touch it.

What to fill in:

- `[executor]` — everything about the contractor that must disappear: `names` (names and their Latin
  spellings), `legal_names` (legal entities), `brands`, `domains` (websites), `email_domains`, `inn`
  (tax and registration numbers), `extra_terms` (anything else that must go).
- `[chats]` — chats to remove completely: `exclude = ["Private chat", "Accounting"]` (name or id), or
  `include_only` — keep only the listed ones. After the first run the report lists the chats with
  message counts.
- `[documents]` — which kinds of papers to delete (`delete_kinds`: contracts, supplementary
  agreements, powers of attorney, acts, invoices, UPD, claims, guarantee letters and disagreement
  protocols by default; waybills and other papers such as specifications stay), `borderline` — what to do with doubtful ones (`delete` — remove and mark
  "check", `keep` — keep), `header_images` — whether to cut pictures out of headers and footers
  (`delete` / `keep`) and `restore` — paths of files that must not be deleted (see below).
- `[messages]` — the message filter (see below). `[media]` — `images`: `avatars` (default: only
  avatars and profile photos are removed), `keep` or `delete` (all pictures); `transcribe = false` —
  do not transcribe audio and video (they are removed then).
- `[ner]` — can stay as it is:
  - `model` — the name model tier: `auto` (chosen by the hardware, see below), `tiny2`, `base`,
    `large` or `off`;
  - `natasha` — `"auto"` (default: Natasha searches only when there is no name model), `true`
    (together with the model: finds a little more, but takes hours of CPU on a large corpus) or `false`;
  - `orgs` — `true` (default): organization names the name model finds are replaced with ORG tokens
    like the contractor from `[executor]` — in all cases and in lowercase, one token per name; a name
    that is a common word — only Capitalized and only if the model found it at least twice.
    Works with a model that knows organizations (`labels.json` with `ORG`): the chosen tier itself or
    the organizations tier of the models archive (`org_tier` in `models/ner/index.json` — a second, light
    pass for organizations only; names still come from the chosen tier); with neither the report says
    so. `false` — names only;
  - `gliner` — gliner-x-large on top of the model, `false` by default. Needs an NVIDIA card,
    `.venv/bin/python -m pip install -e ".[gliner]"` and `.venv/bin/python -m localanon.fetch --gliner`
    (about 7 GB). Keep `gliner_threshold` at 0.9 or higher, otherwise there are many false names;
  - `device` — `auto`, `cpu`, `cuda` or `cuda:N`.

The other keys are described by comments in the file itself. After editing, run the script again.

### Message filter

Every message of chats, tasks and transcripts falls into one of the categories: `dev` (development),
`commercial` (money, payments, deadlines, penalties), `personal` (private matters, not about work),
`other` (service messages). The categories in `[messages].drop_categories` are removed — `commercial`
and `personal` by default. `filter = "auto"` — the model if it is in the repository, otherwise rules;
`rules` — rules only; `off` — the filter is off. A removed message takes its attachments with it. A
chat with no messages left after the filter is removed completely; to remove a chat by hand, use
`[chats].exclude`. The report shows how many messages were removed and in which category.

With the model, a message is removed when the model's probability of the removed categories (summed)
reaches `[messages].threshold`. `"auto"` (default) takes the threshold chosen for the model: on our
test set it wrongly removes about 8 % of development messages and catches about 79 % of the messages
to remove (the rest is caught later, by the main pipeline). A number in (0, 1) sets your own threshold:
lower removes more (0.55 — about 85 % caught, 14 % of development messages lost), higher keeps more
(0.8 — about 70 % caught, 4 % lost). The value `0.5` from earlier versions is read as `"auto"` with a
note; write `0.501` to really use 0.5. The report counts `messages:removed:<category>` and
`messages:doubtful` — kept messages just below the threshold. The model tier (`base`, or the lighter
`tiny2` on a slow CPU) is chosen by the hardware, see "Hardware".

## Restoring a file

The report lists every removed file with its path, category and reason. Doubtful cases are marked
"check" — the document looks like a contract or an invoice, but the signs are ambiguous. If a file was
removed by mistake, add its path (as in the report table) to the settings and run the script again:

```toml
[documents]
restore = ["Documents/Specification v2.pdf", "Chats/Project/files/requirements.docx"]
```

This restores documents (category `document`). To stop deleting documents of some kind at all, remove
the kind from `[documents].delete_kinds`. Archives that were not unpacked and files of unknown kinds
(`.sketch`, `.epub`, files without an extension) are never deleted — they are in the archive, listed
under "Kept with a mark". Messages come back through `[messages]` (a category — remove it from
`drop_categories`, the whole filter — `filter = "off"`), pictures — `[media].images = "keep"`, a whole
chat — remove it from `[chats].exclude`.

## Check problems

The checks at the end do not stop the archive. The report section "Check problems" lists files that
turned out broken after processing (JSON does not parse, an Office document does not open) and known
values left in the output; `local_manifest.json` carries their count, we finish these places on our
side — hand over the archive as usual. A JSON file that was already broken in your export is not a
problem of processing: it is masked as text and listed under "Kept with a mark".

## If the archive is not built

This happens only when a stage failed (exit code `1`) or the settings or the input are wrong (exit
code `2`). The report is always written; tell us about the problem and attach `local_manifest.json`
(it contains no data, only counters).

## Hardware

Models and devices are chosen automatically, no flags needed:
- the name model: on an NVIDIA card with enough memory — the largest tier (`large`), on CPU — `base`;
- the message filter: `base` both on a card and on CPU (it is the largest filter tier);
- on a slow CPU both take the lightest tier (`tiny2`). The CPU speed is measured by a short probe at the
  start of every run (the best of up to 3 rounds, about 3 seconds);
- speech recognition (whisper): `large-v3` on a card, `medium` or `small` on CPU.

The choice and the probe result are in the notes of `report.html`. To see the hardware and the plan:

```bash
.venv/bin/localanon hw --config localanon.toml            # Windows: .venv\Scripts\localanon.exe hw ...
```

Lines with `!` are notes; if a note contains a fix command (`python -m pip ...`), run it with
`.venv/bin/python`.

NVIDIA card details: `onnxruntime` (CPU) and `onnxruntime-gpu` are different packages with the same
module, so the start script installs the GPU build last, removes the CPU build and reinstalls the GPU
build on top. faster-whisper needs cuBLAS 12 and cuDNN 9 for CUDA 12 (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`); they share files with the CUDA 13 cuDNN of `onnxruntime-gpu`, so they are
installed as the very last step. After changing packages by hand, check with
`.venv/bin/localanon hw --probe-cuda`.

## Optional tools

- **LibreOffice** — old formats `.doc`, `.xls`, `.ppt`. Without it such files stay as they are,
  marked "not anonymized".
- **tesseract** (with Russian) — scanned PDFs without a text layer. Without it scans stay as they
  are, marked "not anonymized".
- **py7zr**, **rarfile** — `.7z` and `.rar` archives inside the export
  (`.venv/bin/python -m pip install py7zr rarfile`; rar also needs the `unrar` program). Without them
  such archives stay unpacked with a mark; `.zip`, `.tar`, `.tar.gz` are always unpacked.

## Running by hand

`run.sh` / `run.ps1` call:

```bash
.venv/bin/localanon run --in /path/to/export --out out/export --config localanon.toml
```

The original files are not changed: the script works on a copy in a temporary folder (`--tmp <dir>` —
your own, e.g. on a disk with enough space; `--keep-work` — keep the copy after the run). If there is
not enough space to unpack an archive, the script says so: unpack it yourself and pass the folder.
Text files not in UTF-8 (e.g. Windows-1251) are converted to UTF-8; their number is in the report.

## Several machines: shards

The input is split into N parts by a stable hash of the path. Every machine runs its shard with the
same settings file (the same salt), then the outputs are merged:

```bash
localanon run --in export --out out-1 --config localanon.toml --shard 1/2   # machine 1
localanon run --in export --out out-2 --config localanon.toml --shard 2/2   # machine 2
localanon merge out-1 out-2 --out out
```

- The unit of splitting is a project or chat folder. Top-level files (e.g. a shared CSV) are one
  common unit. A level with files only is split by files. A Telegram export (`result.json`) is not
  split: it goes to one shard as a whole.
- OS service files (`desktop.ini`, `Thumbs.db`, `.DS_Store`, `__MACOSX`, `._*`, `~$*`) are not counted;
  those next to projects stay only in shard 1.
- If there are fewer units than N or a shard is empty, `run` prints a warning to stderr and to the
  report.
- `merge` accepts only a complete set: every k from 1 to N of one split, each exactly once, all from
  the same input (the same number of units and files). Otherwise it fails and builds nothing.

**Shards are independent.** The roster (people and organizations collected from tables) is built
within a run. A person from the roster of another shard may be missed in this shard. For full
consistency run one shard.
