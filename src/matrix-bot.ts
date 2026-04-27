import fs from 'fs';
import os from 'os';
import path from 'path';

import yaml from 'js-yaml';
import { Marked } from 'marked';
import { MatrixClient, SimpleFsStorageProvider } from 'matrix-bot-sdk';

import { runAgentWithRetry, AgentProgressEvent } from './agent.js';
import { AgentError } from './errors.js';
import {
  AGENT_ID,
  AGENT_TIMEOUT_MS,
  CLAUDECLAW_CONFIG,
  DASHBOARD_PORT,
  DASHBOARD_TOKEN,
  DASHBOARD_URL,
  EXFILTRATION_GUARD_ENABLED,
  MATRIX_ACCESS_TOKEN,
  MATRIX_AUTHORIZED_USERS,
  MATRIX_HOMESERVER_URL,
  MATRIX_USER_ID,
  MODEL_FALLBACK_CHAIN,
  PROJECT_ROOT,
  PROTECTED_ENV_VARS,
  SHOW_COST_FOOTER,
  SMART_ROUTING_CHEAP_MODEL,
  SMART_ROUTING_ENABLED,
  STORE_DIR,
  agentDefaultModel,
  agentMcpAllowlist,
  agentSystemPrompt,
} from './config.js';
import { buildCostFooter } from './cost-footer.js';
import {
  clearSession,
  getRecentMemories,
  getRecentTaskOutputs,
  getSession,
  pinMemory,
  setSession,
  unpinMemory,
} from './db.js';
import { scanForSecrets, redactSecrets } from './exfiltration-guard.js';
import { logger } from './logger.js';
import {
  MEMORY_NUDGE_TEXT,
  buildMemoryContext,
  evaluateMemoryRelevance,
  saveConversationTurn,
  shouldNudgeMemory,
} from './memory.js';
import { classifyMessageComplexity } from './message-classifier.js';
import { messageQueue } from './message-queue.js';
import { delegateToAgent, parseDelegation } from './orchestrator.js';
import {
  audit,
  checkKillPhrase,
  executeEmergencyKill,
  getSecurityStatus,
  isLocked,
  lock,
  touchActivity,
  unlock,
} from './security.js';
import { emitChatEvent, setActiveAbort, setProcessing } from './state.js';
import { synthesizeSpeech, transcribeAudio, voiceCapabilities } from './voice.js';

// Matrix has no hard upper bound (events up to ~64 KB), but very long
// single messages render poorly on mobile and cost more cache. Split here.
const MATRIX_MAX_MESSAGE_LENGTH = 4000;

// Typing indicators on Matrix expire after the timeout passed in
// setTyping(); refresh ahead of that.
const MATRIX_TYPING_TIMEOUT_MS = 30_000;
const MATRIX_TYPING_REFRESH_MS = 25_000;

// Per-chat (roomId) model override, same pattern as bot.ts/signal-bot.ts.
const chatModelOverride = new Map<string, string>();

/**
 * Single-writer lock for rooms.yaml updates. Concurrent /cwd commands from
 * different rooms would race on read-mutate-write; we chain mutations onto
 * one promise so they run sequentially. In-process only — fine for this
 * single-bot deployment, would need a real lock in a multi-writer setup.
 */
let cwdWriteLock: Promise<void> = Promise.resolve();
function withCwdWriteLock<T>(fn: () => T | Promise<T>): Promise<T> {
  const next = cwdWriteLock.then(() => fn());
  cwdWriteLock = next.then(() => undefined, () => undefined);
  return next;
}

/**
 * Markdown → HTML renderer for Matrix `formatted_body`. Claude's responses
 * use markdown (bold, lists, code blocks, headers, links, tables); Element
 * renders Matrix custom-html, so we ship both. `gfm: true` enables GitHub
 * extensions (tables, fenced code, autolinks); `breaks: true` turns single
 * newlines into <br> so chat-style line breaks survive.
 *
 * The sender is trusted (Claude itself), so we don't pull in DOMPurify —
 * just defensively strip <script> tags from the output as a last line of
 * defence. Element/Matrix clients also sanitize on render.
 */
const markdown = new Marked({ gfm: true, breaks: true });

/** Render markdown text to a Matrix-safe HTML string. */
function renderMarkdownToHtml(text: string): string {
  let html: string;
  try {
    html = markdown.parse(text, { async: false }) as string;
  } catch (err) {
    logger.warn({ err }, 'markdown render failed — falling back to escaped text');
    html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
  // Defensive: strip <script>...</script> blocks. Marked doesn't emit them
  // from markdown but raw HTML inside the markdown could.
  return html.replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi, '');
}

/**
 * Per-room config loaded from rooms.yaml. Currently only `cwd` (working
 * directory the agent runs in for this room) and an optional human label.
 * Extend here if more per-room knobs are needed (model, system prompt, etc.).
 */
interface RoomConfig {
  cwd: string;
  label?: string;
}

/**
 * Load `${CLAUDECLAW_CONFIG}/rooms.yaml` into a Map<roomId, RoomConfig>.
 * Override path with MATRIX_ROOM_CWD_FILE for tests. Missing/empty file
 * returns an empty map (callers fall back to default cwd silently).
 *
 * Schema (room IDs use Conduit's short form — no `:homeserver` suffix —
 * because that's what matrix-bot-sdk hands us in event handlers):
 *   rooms:
 *     "!abc...":
 *       cwd: /home/customers/foo
 *       label: "Foo customer"
 */
/** Resolve the rooms.yaml path the same way the loader does. */
function roomConfigFilePath(): string {
  return process.env.MATRIX_ROOM_CWD_FILE || path.join(CLAUDECLAW_CONFIG, 'rooms.yaml');
}

/**
 * Serialise a Map<roomId, RoomConfig> back to rooms.yaml. The write is
 * atomic-ish (write-to-tmp + rename). NOTE: js-yaml does not preserve YAML
 * comments — every save clobbers them. The /cwd reply warns the user.
 *
 * Concurrency: callers must serialise calls (cwdWriteLock below). We do not
 * do file-locking; multiple bot processes writing to the same file at once
 * would still race. In practice all writes come from the main bot only.
 */
function saveRoomConfigs(map: Map<string, RoomConfig>): void {
  const file = roomConfigFilePath();
  const rooms: Record<string, RoomConfig> = {};
  for (const [roomId, cfg] of map.entries()) {
    rooms[roomId] = cfg.label ? { cwd: cfg.cwd, label: cfg.label } : { cwd: cfg.cwd };
  }
  const yamlStr = yaml.dump({ rooms }, { indent: 2, lineWidth: 120, quotingType: '"' });
  const tmp = `${file}.tmp`;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(tmp, yamlStr, 'utf-8');
  fs.renameSync(tmp, file);
}

function loadRoomConfigs(): Map<string, RoomConfig> {
  const roomCfgFile = roomConfigFilePath();
  const map = new Map<string, RoomConfig>();
  let raw: string;
  try {
    raw = fs.readFileSync(roomCfgFile, 'utf-8');
  } catch {
    logger.info({ roomCfgFile }, 'No rooms.yaml found — per-room cwd disabled');
    return map;
  }
  try {
    const parsed = yaml.load(raw) as { rooms?: Record<string, Partial<RoomConfig>> } | null;
    const rooms = parsed?.rooms;
    if (!rooms || typeof rooms !== 'object') {
      logger.warn({ roomCfgFile }, 'rooms.yaml has no top-level "rooms" map — ignoring');
      return map;
    }
    for (const [roomId, cfg] of Object.entries(rooms)) {
      if (!cfg || typeof cfg !== 'object') continue;
      if (!cfg.cwd || typeof cfg.cwd !== 'string') {
        logger.warn({ roomId }, 'rooms.yaml entry missing string `cwd` — skipping');
        continue;
      }
      map.set(roomId, { cwd: cfg.cwd, label: typeof cfg.label === 'string' ? cfg.label : undefined });
    }
    logger.info({ roomCfgFile, count: map.size }, 'Loaded room configs from rooms.yaml');
  } catch (err) {
    logger.warn({ err, roomCfgFile }, 'Failed to parse rooms.yaml — per-room cwd disabled');
  }
  return map;
}

// Per-chat voice-reply preference. Three states:
//   'on'    → always reply as TTS, even for typed prompts
//   'off'   → never reply as TTS, even for voice notes (hard-mute)
//   unset   → default: mirror the incoming modality
const voiceMode = new Map<string, 'on' | 'off'>();

// Cache of room encryption status. `true` = E2EE (skip), `false` = plain
// (process). Avoids one /state round-trip per incoming message.
const roomEncryptionCache = new Map<string, boolean>();

interface MatrixMessageEvent {
  event_id: string;
  sender: string;
  origin_server_ts?: number;
  type: string;
  content?: {
    msgtype?: string;
    body?: string;
    url?: string;
    file?: { url?: string; iv?: string; key?: unknown };
    info?: { mimetype?: string; size?: number; duration?: number };
    'm.relates_to'?: { rel_type?: string; event_id?: string };
  };
}

/**
 * Write a Buffer to a temp file with the given suffix and return its path.
 * Caller is responsible for cleanup.
 */
function writeTempFile(buffer: Buffer, suffix: string): string {
  const p = path.join(
    os.tmpdir(),
    `claudeclaw-matrix-${Date.now()}-${Math.random().toString(36).slice(2, 8)}${suffix}`,
  );
  fs.writeFileSync(p, buffer);
  return p;
}

/** Best-effort unlink — logs warnings instead of throwing. */
function tryUnlink(p: string): void {
  try { fs.unlinkSync(p); } catch (err) { logger.warn({ err, path: p }, 'tmp unlink failed'); }
}

/** Map a Matrix audio mimetype to its on-disk file extension. */
function audioExtFromMime(mime: string | undefined): string {
  if (!mime) return '.ogg';
  if (mime.includes('aac')) return '.aac';
  if (mime.includes('mpeg') || mime.includes('mp3')) return '.mp3';
  if (mime.includes('ogg') || mime.includes('opus')) return '.ogg';
  if (mime.includes('wav')) return '.wav';
  if (mime.includes('m4a') || mime.includes('mp4')) return '.m4a';
  if (mime.includes('webm')) return '.webm';
  if (mime.includes('flac')) return '.flac';
  return '.ogg';
}

/** Map a Matrix media mimetype to a sensible file extension. */
function fileExtFromMime(mime: string | undefined, fallback: string): string {
  if (!mime) return fallback;
  if (mime.startsWith('audio/')) return audioExtFromMime(mime);
  if (mime === 'image/jpeg') return '.jpg';
  if (mime === 'image/png') return '.png';
  if (mime === 'image/webp') return '.webp';
  if (mime === 'image/gif') return '.gif';
  if (mime === 'application/pdf') return '.pdf';
  return fallback;
}

/** Parse comma-separated MXIDs into a Set, dropping empty entries. */
function isAuthorised(sender: string): boolean {
  if (MATRIX_AUTHORIZED_USERS.length === 0) return false;
  return MATRIX_AUTHORIZED_USERS.includes(sender);
}

/** Split a long response so Matrix doesn't render one giant wall of text. */
function splitMessage(text: string): string[] {
  if (text.length <= MATRIX_MAX_MESSAGE_LENGTH) return [text];
  const parts: string[] = [];
  let remaining = text;
  while (remaining.length > MATRIX_MAX_MESSAGE_LENGTH) {
    const chunk = remaining.slice(0, MATRIX_MAX_MESSAGE_LENGTH);
    const lastNewline = chunk.lastIndexOf('\n');
    const splitAt = lastNewline > MATRIX_MAX_MESSAGE_LENGTH / 2 ? lastNewline : MATRIX_MAX_MESSAGE_LENGTH;
    parts.push(remaining.slice(0, splitAt));
    remaining = remaining.slice(splitAt).trimStart();
  }
  if (remaining) parts.push(remaining);
  return parts;
}

interface FileMarker {
  type: 'document' | 'photo';
  filePath: string;
  caption?: string;
}

/** Extract [SEND_FILE:...] and [SEND_PHOTO:...] markers. Same shape as bot.ts. */
function extractFileMarkers(text: string): { text: string; files: FileMarker[] } {
  const files: FileMarker[] = [];
  const pattern = /\[SEND_(FILE|PHOTO):([^\]\|]+)(?:\|([^\]]*))?\]/g;
  const cleaned = text.replace(pattern, (_, kind: string, filePath: string, caption?: string) => {
    files.push({
      type: kind === 'PHOTO' ? 'photo' : 'document',
      filePath: filePath.trim(),
      caption: caption?.trim() || undefined,
    });
    return '';
  });
  return { text: cleaned.replace(/\n{3,}/g, '\n\n').trim(), files };
}

const AVAILABLE_MODELS: Record<string, string> = {
  opus: 'claude-opus-4-6',
  sonnet: 'claude-sonnet-4-5',
  haiku: 'claude-haiku-4-5',
};

export interface MatrixBot {
  start(): Promise<void>;
  stop(): Promise<void>;
  /** For outside code (scheduler / memory callbacks) to push messages. */
  sendTo(roomId: string, text: string): Promise<void>;
}

/**
 * Create the Matrix bot. Does NOT connect yet — call `start()` to open the
 * sync stream and begin receiving messages.
 */
export function createMatrixBot(): MatrixBot {
  if (!MATRIX_HOMESERVER_URL) {
    throw new Error('MATRIX_HOMESERVER_URL not set in .env.');
  }
  if (!MATRIX_ACCESS_TOKEN) {
    throw new Error('MATRIX_ACCESS_TOKEN not set in .env.');
  }
  if (!MATRIX_USER_ID) {
    throw new Error('MATRIX_USER_ID not set in .env.');
  }
  if (MATRIX_AUTHORIZED_USERS.length === 0) {
    throw new Error(
      'MATRIX_AUTHORIZED_USERS is empty — refusing to start a bot that would talk to anyone.',
    );
  }

  // Persistent sync token + per-room state survives restarts.
  const storageDir = path.join(STORE_DIR, 'matrix');
  fs.mkdirSync(storageDir, { recursive: true });
  const storage = new SimpleFsStorageProvider(path.join(storageDir, 'bot.json'));

  const client = new MatrixClient(MATRIX_HOMESERVER_URL, MATRIX_ACCESS_TOKEN, storage);

  // Per-room cwd map, loaded once at startup. Rooms not in this map fall
  // back to the agent's default cwd (PROJECT_ROOT).
  const roomConfigs = loadRoomConfigs();

  // Voice STT/TTS capability is fixed at process start (env-driven); cache
  // once instead of probing on every message.
  const voiceCaps = voiceCapabilities();

  /**
   * Send a markdown message: rendered to HTML for Matrix-formatted clients
   * (Element renders bold/code/lists/etc.) with the original markdown kept
   * as plaintext fallback. Splits long text first, then renders per chunk
   * so the HTML stays well-formed in each part.
   */
  const sendMessage = async (roomId: string, text: string): Promise<void> => {
    if (!text || !text.trim()) return;
    for (const part of splitMessage(text)) {
      const html = renderMarkdownToHtml(part);
      try {
        await client.sendMessage(roomId, {
          msgtype: 'm.text',
          body: part,
          format: 'org.matrix.custom.html',
          formatted_body: html,
        });
      } catch (err) {
        logger.error({ err, roomId }, 'matrix send failed');
        // Don't throw — a dropped message must not crash the receive loop.
      }
    }
  };

  /** Upload a local file and send as document/photo/audio. */
  const sendFile = async (roomId: string, filePath: string, caption?: string): Promise<void> => {
    if (!fs.existsSync(filePath)) {
      await sendMessage(roomId, `Could not send file: ${filePath} (not found)`);
      return;
    }
    try {
      const buffer = fs.readFileSync(filePath);
      const filename = path.basename(filePath);
      const ext = path.extname(filePath).toLowerCase();
      const mime =
        ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg'
        : ext === '.png' ? 'image/png'
        : ext === '.gif' ? 'image/gif'
        : ext === '.webp' ? 'image/webp'
        : ext === '.pdf' ? 'application/pdf'
        : ext === '.mp3' ? 'audio/mpeg'
        : ext === '.ogg' || ext === '.opus' ? 'audio/ogg'
        : ext === '.wav' ? 'audio/wav'
        : ext === '.m4a' ? 'audio/mp4'
        : 'application/octet-stream';
      const isImage = mime.startsWith('image/');
      const isAudio = mime.startsWith('audio/');
      const url: string = await client.uploadContent(buffer, mime, filename);
      const msgtype = isImage ? 'm.image' : isAudio ? 'm.audio' : 'm.file';
      const content: Record<string, unknown> = {
        msgtype,
        body: caption ?? filename,
        url,
        info: { mimetype: mime, size: buffer.length },
      };
      if (msgtype === 'm.file') (content as { filename?: string }).filename = filename;
      await client.sendMessage(roomId, content);
      // If a caption was supplied AND we just sent a non-text artifact, also
      // send the caption as a separate text message so it's visible across
      // clients that don't render the body alongside the artifact.
      if (caption && !isImage && msgtype !== 'm.file') {
        await sendMessage(roomId, caption);
      }
    } catch (err) {
      logger.error({ err, filePath }, 'matrix file send failed');
      await sendMessage(roomId, `Failed to send file: ${path.basename(filePath)}`);
    }
  };

  /** Fire a single typing-indicator pulse. */
  const sendTyping = async (roomId: string): Promise<void> => {
    try {
      await client.setTyping(roomId, true, MATRIX_TYPING_TIMEOUT_MS);
    } catch {
      // Best-effort only.
    }
  };

  /** Stop typing — call from finally blocks to clean up the indicator. */
  const stopTyping = async (roomId: string): Promise<void> => {
    try { await client.setTyping(roomId, false, 0); } catch { /* ok */ }
  };

  /**
   * Convert mxc://server/mediaId to a downloadable HTTP URL on our homeserver,
   * then download the bytes. matrix-bot-sdk has `downloadContent` but the
   * exact return shape varies between versions; use a manual fetch for
   * deterministic behaviour.
   */
  const downloadMxcToTempFile = async (
    mxcUrl: string,
    contentType: string | undefined,
  ): Promise<string> => {
    const resp = await client.downloadContent(mxcUrl);
    // matrix-bot-sdk 0.7+ returns { data: Buffer, contentType: string }.
    const buf: Buffer = Buffer.isBuffer(resp) ? resp : (resp as { data: Buffer }).data;
    const ext = fileExtFromMime(contentType, '.bin');
    return writeTempFile(buf, ext);
  };

  /** Read m.room.encryption state — true if the room is E2EE. Cached. */
  const isRoomEncrypted = async (roomId: string): Promise<boolean> => {
    const cached = roomEncryptionCache.get(roomId);
    if (cached !== undefined) return cached;
    let encrypted = false;
    try {
      const ev = await client.getRoomStateEvent(roomId, 'm.room.encryption', '');
      encrypted = !!(ev && (ev as { algorithm?: string }).algorithm);
    } catch {
      encrypted = false;
    }
    roomEncryptionCache.set(roomId, encrypted);
    return encrypted;
  };

  /**
   * Core message handler — ported from signal-bot.ts handleTextMessage().
   * `chatId` is the Matrix roomId. Authorization is checked at the outer
   * event listener; by the time we land here, the sender is allowlisted.
   */
  async function handleTextMessage(
    roomId: string,
    message: string,
    forceVoiceReply = false,
  ): Promise<void> {
    const chatId = roomId;

    // Emergency kill works even when locked.
    if (checkKillPhrase(message)) {
      audit({ agentId: AGENT_ID, chatId, action: 'kill', detail: 'Emergency kill via Matrix', blocked: false });
      await sendMessage(chatId, 'EMERGENCY KILL activated. All agents stopping.');
      executeEmergencyKill();
      return;
    }

    // PIN lock gate.
    if (isLocked()) {
      if (unlock(message)) {
        audit({ agentId: AGENT_ID, chatId, action: 'unlock', detail: 'PIN accepted', blocked: false });
        await sendMessage(chatId, 'Unlocked. Session active.');
      } else {
        audit({ agentId: AGENT_ID, chatId, action: 'blocked', detail: 'Session locked, wrong PIN', blocked: true });
        await sendMessage(chatId, 'Session locked. Send your PIN to unlock.');
      }
      return;
    }

    touchActivity();
    audit({ agentId: AGENT_ID, chatId, action: 'message', detail: message.slice(0, 200), blocked: false });
    emitChatEvent({ type: 'user_message', chatId, content: message, source: 'matrix' });

    // Delegation (@agent or /delegate) — same parser as bot.ts/signal-bot.ts.
    const delegation = parseDelegation(message);
    if (delegation) {
      setProcessing(chatId, true);
      void sendTyping(chatId);
      try {
        const result = await delegateToAgent(
          delegation.agentId,
          delegation.prompt,
          chatId,
          AGENT_ID,
          async (progressMsg) => {
            emitChatEvent({ type: 'progress', chatId, description: progressMsg });
            await sendMessage(chatId, progressMsg);
          },
        );
        const text = result.text?.trim() || 'Agent completed with no output.';
        const header = `[${result.agentId} — ${Math.round(result.durationMs / 1000)}s]`;
        saveConversationTurn(chatId, delegation.prompt, text, undefined, delegation.agentId);
        emitChatEvent({ type: 'assistant_message', chatId, content: text, source: 'matrix' });
        await sendMessage(chatId, `${header}\n\n${text}`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error({ err, agentId: delegation.agentId }, 'Delegation failed');
        await sendMessage(chatId, `Delegation to ${delegation.agentId} failed: ${msg}`);
      } finally {
        setProcessing(chatId, false);
        await stopTyping(chatId);
      }
      return;
    }

    // Main agent path — build memory context, pick model, run, reply.
    const sessionId = getSession(chatId, AGENT_ID);
    const { contextText: memCtx, surfacedMemoryIds, surfacedMemorySummaries } =
      await buildMemoryContext(chatId, message, AGENT_ID);

    const parts: string[] = [];
    if (agentSystemPrompt && !sessionId) {
      parts.push(`[Agent role — follow these instructions]\n${agentSystemPrompt}\n[End agent role]`);
    }
    if (memCtx) parts.push(memCtx);

    const recentTasks = getRecentTaskOutputs(AGENT_ID, 30);
    if (recentTasks.length > 0) {
      const taskLines = recentTasks.map((t) => {
        const ago = Math.round((Date.now() / 1000 - t.last_run) / 60);
        return `[Scheduled task ran ${ago}m ago]\nTask: ${t.prompt}\nOutput:\n${t.last_result}`;
      });
      parts.push(`[Recent scheduled task context]\n${taskLines.join('\n\n')}\n[End task context]`);
    }

    if (shouldNudgeMemory(chatId, AGENT_ID)) parts.push(MEMORY_NUDGE_TEXT);
    parts.push(message);

    const userModel = chatModelOverride.get(chatId) ?? agentDefaultModel;
    const effectiveModel = (SMART_ROUTING_ENABLED && !userModel && classifyMessageComplexity(message) === 'simple')
      ? SMART_ROUTING_CHEAP_MODEL
      : (userModel ?? agentDefaultModel ?? 'claude-opus-4-6');

    void sendTyping(chatId);
    const typingInterval = setInterval(() => void sendTyping(chatId), MATRIX_TYPING_REFRESH_MS);
    setProcessing(chatId, true);

    try {
      const onProgress = (event: AgentProgressEvent): void => {
        emitChatEvent({ type: 'progress', chatId, description: event.description });
        // Like signal-bot, only surface task boundaries — tool_active would flood.
        if (event.type === 'task_started') void sendMessage(chatId, `🔄 ${event.description}`);
        if (event.type === 'task_completed') void sendMessage(chatId, `✓ ${event.description}`);
      };

      const abortCtrl = new AbortController();
      setActiveAbort(chatId, abortCtrl);
      const timeoutId = setTimeout(() => {
        logger.warn({ chatId, timeoutMs: AGENT_TIMEOUT_MS }, 'Agent query timed out (Matrix)');
        abortCtrl.abort();
      }, AGENT_TIMEOUT_MS);

      const fullMessage = parts.join('\n\n');

      // Per-room workdir: if rooms.yaml maps this roomId to a cwd, the agent
      // runs there (giving it access to that customer's repo). Otherwise the
      // agent uses its default cwd — behaviour for unmapped rooms is unchanged.
      const roomCfg = roomConfigs.get(chatId);
      const roomCwd = roomCfg?.cwd;
      if (roomCwd) {
        logger.debug({ chatId, roomCwd, label: roomCfg?.label }, 'Using per-room cwd');
      }

      const result = await runAgentWithRetry(
        fullMessage,
        sessionId,
        () => void sendTyping(chatId),
        onProgress,
        effectiveModel,
        abortCtrl,
        /* onStreamText */ undefined,
        async (attempt, error) => {
          await sendMessage(chatId, `${error.recovery.userMessage} (retry ${attempt}/2)`);
        },
        MODEL_FALLBACK_CHAIN.length > 0 ? MODEL_FALLBACK_CHAIN : undefined,
        agentMcpAllowlist,
        roomCwd,
      );

      clearTimeout(timeoutId);
      clearInterval(typingInterval);
      setActiveAbort(chatId, null);

      if (result.aborted) {
        setProcessing(chatId, false);
        const msg = result.text === null
          ? `Timed out after ${Math.round(AGENT_TIMEOUT_MS / 1000)}s. Raise AGENT_TIMEOUT_MS in your .env (default 1800000 = 30 min) and restart, or break the task into smaller steps.`
          : 'Stopped.';
        emitChatEvent({ type: 'assistant_message', chatId, content: msg, source: 'matrix' });
        await sendMessage(chatId, msg);
        await stopTyping(chatId);
        return;
      }

      if (result.newSessionId) setSession(chatId, result.newSessionId, AGENT_ID);

      let rawResponse = result.text?.trim() || 'Done.';

      if (EXFILTRATION_GUARD_ENABLED) {
        const protectedValues = PROTECTED_ENV_VARS
          .map((key) => process.env[key])
          .filter((v): v is string => !!v && v.length > 8);
        const matches = scanForSecrets(rawResponse, protectedValues);
        if (matches.length > 0) {
          rawResponse = redactSecrets(rawResponse, matches);
          logger.warn({ matchCount: matches.length }, 'Exfiltration guard: redacted secrets (Matrix)');
        }
      }

      const { text: responseText, files: fileMarkers } = extractFileMarkers(rawResponse);
      const costFooter = buildCostFooter(SHOW_COST_FOOTER, result.usage, effectiveModel);

      saveConversationTurn(chatId, message, rawResponse, result.newSessionId ?? sessionId, AGENT_ID);
      if (surfacedMemoryIds.length > 0) {
        void evaluateMemoryRelevance(surfacedMemoryIds, surfacedMemorySummaries, message, rawResponse).catch(() => {});
      }
      emitChatEvent({ type: 'assistant_message', chatId, content: rawResponse, source: 'matrix' });

      for (const file of fileMarkers) {
        await sendFile(chatId, file.filePath, file.caption);
      }

      const textWithFooter = responseText ? responseText + costFooter : '';
      const mode = voiceMode.get(chatId);
      const shouldSpeakBack = voiceCaps.tts && (
        mode === 'on' || (mode !== 'off' && forceVoiceReply)
      );

      if (textWithFooter) {
        if (shouldSpeakBack && responseText) {
          // TTS path: synth the response (without cost footer) and ship as
          // an m.audio attachment. Fall back to text if TTS fails.
          let audioPath: string | null = null;
          try {
            const audioBuffer = await synthesizeSpeech(responseText);
            audioPath = writeTempFile(audioBuffer, '.mp3');
            await sendFile(chatId, audioPath);
          } catch (ttsErr) {
            logger.error({ err: ttsErr }, 'TTS failed, falling back to text');
            await sendMessage(chatId, textWithFooter);
          } finally {
            if (audioPath) tryUnlink(audioPath);
          }
        } else {
          await sendMessage(chatId, textWithFooter);
        }
      }
    } catch (err) {
      clearInterval(typingInterval);
      setActiveAbort(chatId, null);
      const errMsg = err instanceof AgentError
        ? err.recovery.userMessage
        : err instanceof Error ? err.message : String(err);
      logger.error({ err }, 'Agent run failed (Matrix)');
      await sendMessage(chatId, `Error: ${errMsg}`);
    } finally {
      setProcessing(chatId, false);
      await stopTyping(chatId);
    }
  }

  /** Dispatch a bare /command. Returns true if handled. */
  async function handleCommand(roomId: string, text: string): Promise<boolean> {
    const chatId = roomId;
    const match = text.match(/^\/(\w+)(?:\s+(.*))?$/s);
    if (!match) return false;
    const cmd = match[1].toLowerCase();
    const arg = (match[2] ?? '').trim();

    switch (cmd) {
      case 'start':
        await sendMessage(chatId, `ClaudeClaw online via Matrix. Agent: ${AGENT_ID}.\n\nSend /help for commands.`);
        return true;

      case 'help':
        await sendMessage(chatId,
          'ClaudeClaw — Commands (Matrix)\n\n' +
          '/newchat — Start a new Claude session\n' +
          '/forget — Clear session\n' +
          '/memory — View recent memories\n' +
          '/pin <id> — Pin a memory\n' +
          '/unpin <id> — Unpin a memory\n' +
          '/voice on|off|auto — Voice replies: always / never / mirror input\n' +
          '/model <opus|sonnet|haiku> — Switch model\n' +
          '/cwd [<path>|unset] — Show or set this room\'s working directory\n' +
          '/agents — List available agents\n' +
          '/delegate <agent> <prompt> — Delegate to an agent\n' +
          '/dashboard — Get dashboard link\n' +
          '/lock — Lock session (PIN required to unlock)\n' +
          '/status — Security status\n' +
          '/stop — Stop current processing\n\n' +
          'Send a voice note for speech-to-text; /voice on for audio replies.\n' +
          'Everything else goes straight to Claude.');
        return true;

      case 'newchat':
      case 'forget':
        clearSession(chatId, AGENT_ID);
        await sendMessage(chatId, 'Session cleared. Next message starts fresh.');
        return true;

      case 'memory': {
        const memories = getRecentMemories(chatId, 10);
        if (memories.length === 0) {
          await sendMessage(chatId, 'No recent memories.');
        } else {
          const lines = memories.map((m) => `#${m.id} [${m.importance.toFixed(1)}] ${m.summary.slice(0, 150)}`);
          await sendMessage(chatId, `Recent memories:\n\n${lines.join('\n')}`);
        }
        return true;
      }

      case 'pin': {
        const id = parseInt(arg, 10);
        if (!id) { await sendMessage(chatId, 'Usage: /pin <memory_id>'); return true; }
        pinMemory(id);
        await sendMessage(chatId, `Memory #${id} pinned.`);
        return true;
      }

      case 'unpin': {
        const id = parseInt(arg, 10);
        if (!id) { await sendMessage(chatId, 'Usage: /unpin <memory_id>'); return true; }
        unpinMemory(id);
        await sendMessage(chatId, `Memory #${id} unpinned.`);
        return true;
      }

      case 'voice': {
        if (!voiceCaps.tts) {
          await sendMessage(chatId,
            'Voice replies not available. Configure one of:\n' +
            '  ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID\n' +
            '  GRADIUM_API_KEY + GRADIUM_VOICE_ID\n' +
            '  KOKORO_URL (local)\n' +
            '…or leave all unset to fall back to macOS `say` (Mac only).');
          return true;
        }
        const sub = arg.toLowerCase();
        if (sub === 'on') {
          voiceMode.set(chatId, 'on');
          await sendMessage(chatId, 'Voice replies enabled. All replies will be spoken. Send /voice off to disable or /voice auto for default mirroring.');
        } else if (sub === 'off') {
          voiceMode.set(chatId, 'off');
          await sendMessage(chatId, 'Voice replies disabled. All replies (including for voice notes) will be text.');
        } else if (sub === 'auto' || sub === 'reset' || sub === 'default') {
          voiceMode.delete(chatId);
          await sendMessage(chatId, 'Voice replies set to auto (mirror incoming modality).');
        } else {
          const state = voiceMode.get(chatId) ?? 'auto';
          await sendMessage(chatId, `Voice replies: ${state}\nUsage: /voice on | /voice off | /voice auto`);
        }
        return true;
      }

      case 'model': {
        const key = arg.toLowerCase();
        if (!key) {
          const current = chatModelOverride.get(chatId) ?? agentDefaultModel ?? 'claude-opus-4-6';
          await sendMessage(chatId, `Current model: ${current}\n\nUsage: /model <opus|sonnet|haiku>`);
          return true;
        }
        const target = AVAILABLE_MODELS[key];
        if (!target) { await sendMessage(chatId, 'Unknown model. Use opus, sonnet, or haiku.'); return true; }
        chatModelOverride.set(chatId, target);
        await sendMessage(chatId, `Model switched to ${target}.`);
        return true;
      }

      case 'cwd': {
        // No arg → show current cwd for this room.
        if (!arg) {
          const cfg = roomConfigs.get(chatId);
          if (cfg) {
            const labelStr = cfg.label ? ` (${cfg.label})` : '';
            await sendMessage(chatId,
              `Current container cwd for this room: \`${cfg.cwd}\`${labelStr}\n\n` +
              'Set with `/cwd <absolute-path>` or remove with `/cwd unset`.');
          } else {
            await sendMessage(chatId,
              'No override — using default agent cwd.\n\n' +
              'Set with `/cwd <absolute-path>` (path must exist inside the bot container).');
          }
          return true;
        }

        // Clear / unset.
        if (arg === 'unset' || arg === 'clear') {
          if (!roomConfigs.has(chatId)) {
            await sendMessage(chatId, 'No override to clear — already using default agent cwd.');
            return true;
          }
          await withCwdWriteLock(() => {
            // Re-read on disk in case external edits happened, then remove.
            const fresh = loadRoomConfigs();
            fresh.delete(chatId);
            roomConfigs.delete(chatId);
            // Merge in any other in-memory entries fresh missed (shouldn't
            // happen, but stay safe): in-memory is authoritative.
            for (const [rid, cfg] of roomConfigs.entries()) fresh.set(rid, cfg);
            saveRoomConfigs(fresh);
          });
          await sendMessage(chatId,
            'Workdir override cleared. This room now uses the default agent cwd.\n\n' +
            'Note: rooms.yaml comments were rewritten on save.');
          return true;
        }

        // Set.
        const targetPath = arg;
        if (!path.isAbsolute(targetPath)) {
          await sendMessage(chatId,
            'Path must be absolute (start with `/`).\n\n' +
            'Usage: `/cwd /home/customers/<name>` — paths refer to the bot CONTAINER, not your host.');
          return true;
        }
        if (!fs.existsSync(targetPath)) {
          await sendMessage(chatId,
            `Path not found inside the bot container: \`${targetPath}\`. Mount it via docker-compose first ` +
            '(e.g. add it under `/home/customers/` on the host and restart the bot).');
          return true;
        }
        try {
          await withCwdWriteLock(() => {
            const fresh = loadRoomConfigs();
            const existing = fresh.get(chatId) ?? roomConfigs.get(chatId);
            const label = existing?.label ?? path.basename(targetPath);
            const cfg: RoomConfig = { cwd: targetPath, label };
            fresh.set(chatId, cfg);
            roomConfigs.set(chatId, cfg);
            // Make sure any other in-memory entries the loader might miss
            // (e.g. if the file was deleted) survive the round-trip.
            for (const [rid, rcfg] of roomConfigs.entries()) {
              if (!fresh.has(rid)) fresh.set(rid, rcfg);
            }
            saveRoomConfigs(fresh);
          });
        } catch (err) {
          logger.error({ err, roomId: chatId, targetPath }, '/cwd save failed');
          const msg = err instanceof Error ? err.message : String(err);
          await sendMessage(chatId, `Failed to save rooms.yaml: ${msg}`);
          return true;
        }
        await sendMessage(chatId,
          `Workdir for this room set to \`${targetPath}\` (container path). Effective on the next message.\n\n` +
          'Saved. Note: rooms.yaml comments were rewritten.');
        return true;
      }

      case 'dashboard': {
        const token = DASHBOARD_TOKEN;
        if (!token) { await sendMessage(chatId, 'Dashboard not configured (DASHBOARD_TOKEN missing).'); return true; }
        const base = DASHBOARD_URL || `http://localhost:${DASHBOARD_PORT}`;
        const url = `${base}/?token=${token}&chatId=${encodeURIComponent(chatId)}`;
        await sendMessage(chatId, `Dashboard:\n${url}`);
        return true;
      }

      case 'agents': {
        const agentsDir = path.join(PROJECT_ROOT, 'agents');
        let ids: string[] = [];
        try {
          ids = fs.readdirSync(agentsDir).filter((d) => fs.statSync(path.join(agentsDir, d)).isDirectory());
        } catch { /* no agents dir */ }
        await sendMessage(chatId, ids.length ? `Agents: ${ids.join(', ')}` : 'No additional agents configured.');
        return true;
      }

      case 'delegate': {
        const rest = arg;
        if (!rest) { await sendMessage(chatId, 'Usage: /delegate <agent> <prompt>'); return true; }
        await handleTextMessage(roomId, `/delegate ${rest}`);
        return true;
      }

      case 'lock':
        lock();
        await sendMessage(chatId, 'Session locked. Send your PIN to unlock.');
        return true;

      case 'status': {
        const s = getSecurityStatus();
        const lines = [
          `PIN lock: ${s.pinEnabled ? (s.locked ? 'locked' : 'unlocked') : 'disabled'}`,
          `Kill phrase: ${s.killPhraseEnabled ? 'enabled' : 'disabled'}`,
          `Idle lock: ${s.idleLockMinutes > 0 ? `${s.idleLockMinutes} min` : 'disabled'}`,
        ];
        await sendMessage(chatId, lines.join('\n'));
        return true;
      }

      case 'stop': {
        const { abortActiveQuery } = await import('./state.js');
        abortActiveQuery(chatId);
        await sendMessage(chatId, 'Stopping.');
        return true;
      }

      default:
        return false;
    }
  }

  /**
   * Allowlist-gated invite handler. Replaces matrix-bot-sdk's
   * AutojoinRoomsMixin with a strict variant.
   */
  const onInvite = async (roomId: string, inviteEvent: { sender?: string }): Promise<void> => {
    const inviter = inviteEvent?.sender ?? '';
    if (!isAuthorised(inviter)) {
      logger.warn({ roomId, inviter }, 'Rejecting invite from unauthorized user');
      try { await client.leaveRoom(roomId); } catch (err) { logger.warn({ err, roomId }, 'leave after rejected invite failed'); }
      return;
    }
    try {
      await client.joinRoom(roomId);
      logger.info({ roomId, inviter }, 'Joined room on invite');
    } catch (err) {
      logger.error({ err, roomId }, 'joinRoom failed');
    }
  };

  /** room.message dispatcher. Filters by sender + msgtype, then routes. */
  const onMessage = async (roomId: string, event: MatrixMessageEvent): Promise<void> => {
    if (!event || !event.content) return;
    if (event.sender === MATRIX_USER_ID) return; // own echo
    if (!isAuthorised(event.sender)) {
      logger.warn({ sender: event.sender, roomId }, 'Dropped unauthorized Matrix message');
      return;
    }

    // Skip edits-of-edits (m.replace relations) — only react to fresh messages.
    if (event.content['m.relates_to']?.rel_type === 'm.replace') return;

    // Encrypted rooms aren't supported in this MVP. The cache lets us
    // skip the /state round-trip after the first message; warn once.
    const wasCached = roomEncryptionCache.has(roomId);
    if (await isRoomEncrypted(roomId)) {
      if (!wasCached) {
        try {
          await sendMessage(roomId, 'This room is end-to-end encrypted. The bot does not yet support E2EE — disable encryption or move to a non-encrypted room.');
        } catch { /* room may itself be unreachable in plaintext */ }
        logger.warn({ roomId }, 'Skipping E2EE room');
      }
      return;
    }

    const msgtype = event.content.msgtype;
    const body = (event.content.body ?? '').toString();

    // ── Voice / audio note ──────────────────────────────────────────────
    if (msgtype === 'm.audio') {
      const mxcUrl = event.content.url ?? event.content.file?.url;
      if (!mxcUrl) {
        await sendMessage(roomId, 'Got an audio message but no media URL. Try resending.');
        return;
      }
      if (!voiceCaps.stt) {
        await sendMessage(roomId,
          'Voice transcription not configured. Set GROQ_API_KEY for cloud STT or WHISPER_CPP_PATH/WHISPER_MODEL_PATH for local STT.',
        );
        return;
      }
      messageQueue.enqueue(roomId, async () => {
        let audioPath: string | null = null;
        try {
          audioPath = await downloadMxcToTempFile(mxcUrl, event.content?.info?.mimetype);
        } catch (err) {
          logger.error({ err, mxcUrl }, 'Voice download failed (Matrix)');
          await sendMessage(roomId, 'Could not download the audio. Try resending.');
          return;
        }
        let transcript: string;
        try {
          transcript = await transcribeAudio(audioPath);
        } catch (err) {
          logger.error({ err }, 'Voice transcription failed (Matrix)');
          await sendMessage(roomId, 'Voice transcription failed. Try again or send text.');
          tryUnlink(audioPath);
          return;
        }
        tryUnlink(audioPath);
        if (!transcript.trim()) {
          await sendMessage(roomId, 'Could not understand the audio. Try again.');
          return;
        }
        logger.info({ roomId, len: transcript.length }, 'Matrix voice transcribed');
        emitChatEvent({ type: 'user_message', chatId: roomId, content: `[voice] ${transcript}`, source: 'matrix' });
        await handleTextMessage(roomId, transcript, /* forceVoiceReply */ true);
      });
      return;
    }

    // ── Image / document — forward as text-with-context ─────────────────
    if (msgtype === 'm.image' || msgtype === 'm.file' || msgtype === 'm.video') {
      // Treat the media as an attachment: caption (body) becomes the prompt.
      // If no caption, ack the media without invoking the agent.
      const caption = body.trim();
      if (!caption) {
        await sendMessage(roomId, 'Got the media. Add a caption (or send a follow-up message) and I’ll work with it.');
        return;
      }
      messageQueue.enqueue(roomId, () => handleTextMessage(roomId, caption));
      return;
    }

    // ── Text ────────────────────────────────────────────────────────────
    if (msgtype !== 'm.text' && msgtype !== 'm.notice' && msgtype !== 'm.emote') return;

    const text = body.trim();
    if (!text) return;

    // Commands first — they bypass the message queue so /stop can interrupt.
    if (text.startsWith('/')) {
      const handled = await handleCommand(roomId, text);
      if (handled) return;
    }

    messageQueue.enqueue(roomId, () => handleTextMessage(roomId, text));
  };

  // matrix-bot-sdk's start() returns immediately after registering the sync
  // promise chain; there's a window with no pending I/O during which Node
  // would otherwise drain the event loop and exit. Hold a heartbeat timer
  // so the process stays alive regardless of scheduler/dashboard state.
  let keepAliveTimer: NodeJS.Timeout | null = null;

  return {
    async start(): Promise<void> {
      client.on('room.invite', (roomId: string, inviteState: unknown) => {
        void onInvite(roomId, (inviteState ?? {}) as { sender?: string }).catch((err) =>
          logger.error({ err, roomId }, 'onInvite threw'),
        );
      });
      client.on('room.message', (roomId: string, event: MatrixMessageEvent) => {
        void onMessage(roomId, event).catch((err) =>
          logger.error({ err, roomId }, 'onMessage threw'),
        );
      });

      await client.start();
      keepAliveTimer = setInterval(() => { /* hold event loop */ }, 60_000);
      logger.info(
        { homeserver: MATRIX_HOMESERVER_URL, userId: MATRIX_USER_ID, allowed: MATRIX_AUTHORIZED_USERS.length },
        'Matrix bot connected to homeserver',
      );
    },
    async stop(): Promise<void> {
      if (keepAliveTimer) { clearInterval(keepAliveTimer); keepAliveTimer = null; }
      try {
        client.stop();
      } catch (err) {
        logger.warn({ err }, 'matrix client stop threw');
      }
    },
    async sendTo(roomId: string, text: string): Promise<void> {
      await sendMessage(roomId, text);
    },
  };
}

/** Export for the scheduler — same shape as bot.ts's splitMessage. */
export { splitMessage };
