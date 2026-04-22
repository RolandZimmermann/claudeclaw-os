# Obsidian Vault Integration via Self-hosted LiveSync

This guide shows how to give ClaudeClaw read/write access to one or more
Obsidian vaults that you already sync across your devices with the
[Self-hosted LiveSync](https://github.com/vrtmrz/obsidian-livesync) plugin
(CouchDB backend). The Mac running the bot becomes *another* linked client
of those vaults: LiveSync mirrors the `.md` files to a local folder, the
bot reads and writes those files directly, and your edits flow back to
every other device (phone, tablet, other desktops) automatically.

This approach beats the alternative of running a CouchDB-MCP bridge
because:

- **Zero extra daemons.** No additional HTTP servers, ports, or OAuth flows.
- **Offline resilience.** The bot keeps working if your CouchDB server is
  briefly unreachable; LiveSync catches up when the link is back.
- **Obsidian parity.** You see exactly what the bot sees — same folder
  structure, same plugin-rendered views, same search index.
- **Free.** No new subscription; you already pay for (or self-host) your
  LiveSync CouchDB.

## Architecture

```
             Phone (Obsidian + LiveSync)
                    │
                    ▼
         CouchDB (e.g. on a Netcup/Hetzner VPS)
                    │
                    ▼
             Mac (Obsidian + LiveSync)
                    │
                    ▼
          ~/Documents/Obsidian/<Vault>/...  (local .md files)
                    │
                    ▼
         ClaudeClaw bot
             │
             │ reads/writes via:
             │   1. `obsidian` CLI          (preferred — respects cache)
             │   2. filesystem MCP server    (fallback — plain I/O)
             │
             ▼
          Claude Code
```

LiveSync is the mirror driver; ClaudeClaw does not talk to CouchDB at all.

## Prerequisites

- **Obsidian Desktop 1.12.7+** (`brew install --cask obsidian` on macOS,
  or the official download for Linux/Windows). 1.12.7 ships the built-in
  CLI binary described below.
- **A working LiveSync setup** on at least one of your other devices,
  with the CouchDB URL, credentials, and end-to-end-encryption passphrase
  at hand. New to LiveSync? Follow
  [the plugin's own setup guide](https://github.com/vrtmrz/obsidian-livesync#quick-start)
  first — you want sync running between your phone and CouchDB before
  adding the Mac.
- **ClaudeClaw** cloned and `npm install`ed.

## Step 1 — Install Obsidian on the bot's machine

```bash
brew install --cask obsidian
```

After install:

```bash
which obsidian   # → /opt/homebrew/bin/obsidian
```

The brew cask automatically links the official CLI binary
(`Obsidian.app/Contents/MacOS/obsidian-cli`). On other platforms the CLI
is enabled through **Settings → General → Command line interface**, which
creates an equivalent symlink.

## Step 2 — Create local vault folders

Pick one parent folder for all vaults so paths stay consistent:

```bash
mkdir -p ~/Documents/Obsidian/Work ~/Documents/Obsidian/Personal
```

Keep these folders **outside** any other sync service (Dropbox, iCloud,
Syncthing for source code, etc.) — two sync engines editing the same
files is a recipe for conflict spam.

## Step 3 — Configure LiveSync per vault

For each vault:

1. Launch Obsidian. On the first run, choose **"Open folder as vault"**
   and point it at `~/Documents/Obsidian/Work` (or whatever folder).
2. **Settings → Community plugins → Browse → Self-hosted LiveSync →
   Install → Enable**.
3. **Settings → Self-hosted LiveSync → Setup Wizard**:
   - URI: your CouchDB URL, e.g. `https://sync.example.com:5984`
   - Username / Password: the CouchDB user credentials
   - Database name: the per-vault database name (e.g. `obsidian_work`)
   - Passphrase: the E2EE passphrase used on your other devices (must
     match *exactly*, case-sensitive).
   - Apply settings, then run the initial replication. Depending on
     vault size this can take seconds to minutes.
4. Once replication finishes, you should see all your notes appear under
   `~/Documents/Obsidian/Work/...`. Sanity-check with `ls` or by opening
   a few notes in Obsidian.

Repeat for every vault you want the bot to see.

**Leave Obsidian running.** The LiveSync plugin only watches CouchDB
changes while the app is open — the Mac's role is "linked client", same
as your phone. (The CLI also requires the app running; see below.)

## Step 4 — Tell ClaudeClaw where the vaults live

ClaudeClaw uses the Claude Agent SDK's MCP support to expose the vault
files to the agent. The simplest configuration is a single
`@modelcontextprotocol/server-filesystem` entry covering every vault plus
any other project folders the bot should be able to read.

Create or edit `.claude/settings.json` at the repo root:

```json
{
  "mcpServers": {
    "vaults-and-work": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/Documents/Obsidian/Work",
        "/Users/you/Documents/Obsidian/Personal",
        "/Users/you/Code/my-project"
      ]
    }
  }
}
```

That file is **user-specific** and should stay out of version control; the
bot's `loadMcpServers()` (`src/agent.ts`) reads it on every agent call.

**Warning:** passing a huge directory tree (e.g. your entire
`~/Sync/workspaces` with gigabytes of node_modules) to the filesystem
server makes the first agent turn noticeably slower while the SDK
advertises tools over the full scope. Limit the allowlist to folders the
bot actually needs.

## Step 5 — Teach the agent about the CLI

The `obsidian` CLI gives richer, cache-aware access than raw filesystem
reads: it talks to the running Obsidian app over IPC, so it sees
frontmatter, tags, task-state changes, and per-vault aliases as the app
does. Every tool response also includes an
`obsidian://` deep link that renders as a tappable "Open in Obsidian" on
mobile.

List known commands with:

```bash
obsidian help                      # all commands
obsidian help <command>            # details for one command
```

Relevant subset for an AI assistant:

```
read                 Read a note
create               Create a new note (with optional content or template)
append / prepend     Add to an existing note
search               Full-text search
search:context       Search with matching-line context
tasks / task         List or toggle tasks (Obsidian Tasks plugin format)
tags / tag           Tag inventory
daily:read / append  Interact with the daily note
files / folders      List files / folders
outline              Show headings
properties / property:*   Read/write frontmatter
backlinks / links / orphans     Knowledge-graph navigation
random / random:read    Pull an arbitrary note (surprise-me mode)
```

All commands accept `vault="<name>"` to pick a specific vault by its
Obsidian display name (not the folder name). Vault display names come
from `obsidian vaults verbose`.

Add hints to your agent's `CLAUDE.md` so the model knows the CLI exists
and when to reach for it rather than raw filesystem reads. A sanitized
template is provided in this repo at **`CLAUDE.md.example`** — use it as a
starting point.

Minimum snippet for the agent's personality file:

```markdown
## Obsidian vaults

I have access to the following Obsidian vaults on this machine:

- "Work"     — business/project notes
- "Personal" — private notes, journal, life admin

Vaults are synced across devices via Self-hosted LiveSync. The official
Obsidian CLI is on the PATH as `obsidian` and expects the Obsidian
desktop app to be running. Use it for note-aware operations:

    obsidian vault="Personal" read file="index"
    obsidian vault="Work" search query="client-x onboarding" limit=5
    obsidian vault="Work" create name="2026-04-22_meeting" content="..." path="inbox/Bot"

For reads, plain filesystem access via the Read tool also works and is
faster if I don't need frontmatter/tag parsing.

Writing convention: bot-generated notes go under `<inbox>/Bot/` or
`<captures>/` of the respective vault, depending on the vault's existing
folder layout. Check `obsidian vault="X" folders` first and respect
casing (Personal uses lowercase `inbox/`, Work might use `Inbox/`).
```

## Step 6 — Verify end-to-end

Start the bot (`npm start` or the launchd service) and send a message
from your Signal / Telegram / whichever messenger you configured:

1. **Read:** *"Read the index note from my Personal vault"* — the agent
   should invoke `obsidian vault="Personal" read file="index"` and show
   you the frontmatter + content.
2. **Search:** *"Find all my notes about HRV"* — agent uses
   `obsidian search query="HRV"` or `search:context`.
3. **Write:** *"Create a note called 2026-04-22_test in the Personal
   vault under inbox/Bot with content 'Hello from the bot'"* — agent
   uses `obsidian create` (or the filesystem MCP). Within ~10 seconds the
   note should appear on your phone via LiveSync.
4. **Daily note:** *"Add a task 'Buy milk' to my daily note"* — agent
   uses `obsidian daily:append content="- [ ] Buy milk"`.

## Writing etiquette

To keep agent-generated content from colliding with human edits:

- **Write into a dedicated subfolder.** Recommended convention: create
  `<vault>/<inbox-folder>/Bot/` in every vault (use the vault's existing
  casing, e.g. `inbox/Bot/` or `Inbox/Bot/`). The example CLAUDE.md in
  this repo documents this.
- **Add provenance frontmatter** to every bot-authored note:

  ```markdown
  ---
  created: 2026-04-22
  source: claudeclaw
  via: signal
  ---
  ```

- **Avoid editing notes the user actively maintains** unless the user
  explicitly asked. Prefer `append` over `overwrite` / `write_note` when
  modifying existing notes.

## Multiple vaults

LiveSync uses one CouchDB database per vault. Repeat Step 3 for every
vault: new local folder, new plugin configuration, matching database and
passphrase. The `obsidian vaults verbose` command lists everything the
Obsidian desktop app is currently aware of. The filesystem MCP entry
just adds more paths — one server, many roots.

## Troubleshooting

### `The CLI is unable to find Obsidian. Please make sure Obsidian is running and try again.`

Exactly what it says: start Obsidian.app. The CLI is a thin wrapper that
talks to the running app via a local socket; it does not embed the app.

If the app *is* running but you still get this, try:

- Toggle **Settings → General → Command line interface** off and on
  again (rebuilds the symlink).
- Check you're using the bundled binary:
  `ls -la /opt/homebrew/bin/obsidian` — should be a symlink into
  `/Applications/Obsidian.app/Contents/MacOS/obsidian-cli`.

### Notes are on the Mac but not on my phone

LiveSync's status bar (bottom-right in Obsidian) shows the replication
direction. If it's red, open **Settings → Self-hosted LiveSync → Remote
database → Check database configuration**. Common causes:

- Passphrase mismatch. The bot sees plaintext files locally, but if the
  passphrase is wrong LiveSync can't push them back to CouchDB.
- CouchDB user lacks `_writer` role for the target database.
- CouchDB disk full.

### Paths with spaces don't work in `launchd`

macOS `launchd` crash-loops silently (exit 78) when `StandardOutPath` or
`StandardErrorPath` contain spaces. If your vault folder path contains
spaces, the paths in `settings.json` are still fine (Node quotes them
automatically), but logs go to `/tmp/*.log` regardless.

### The agent writes files with the wrong casing

The agent adapts to the vault's existing folder convention on its own —
but only if it has already seen other folders in that vault. First write
into a fresh vault may land at the wrong case if nothing else hints.
Mitigate with an explicit hint in `CLAUDE.md`:

```markdown
The Personal vault uses lowercase folder names (inbox/, captures/, ...).
The Work vault uses Title-case (Inbox/, Captures/, ...).
```

### Agent queries stall when a new filesystem-MCP path is added

The first call after adding a new path does a full tool re-discovery
across every root. Huge roots (millions of files) can take tens of
seconds. Keep the allowlist minimal and specific.

## Related files in this repo

- `CLAUDE.md.example` — starter template for the agent's personality
  file, with the CLI usage block already wired up.
- `src/obsidian.ts` — legacy helper that pulls open `- [ ]` tasks from a
  configured vault directory into the agent's system prompt. Optional;
  works alongside the MCP-based access described here.
