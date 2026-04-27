import fs from 'fs';
import os from 'os';
import path from 'path';

import { MatrixClient, SimpleFsStorageProvider } from 'matrix-bot-sdk';

import { runAgentWithRetry, AgentProgressEvent } from './agent.js';
import { AgentError } from './errors.js';
import {
  AGENT_ID,
  AGENT_TIMEOUT_MS,
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

// Per-chat voice-reply preference. Three states:
//   'on'    → always reply as TTS, even for typed prompts
//   'off'   → never reply as TTS, even for voice notes (hard-mute)
//   unset   → default: mirror the incoming modality
const voiceMode = new Map<string, 'on' | 'off'>();

// Track rooms we've decided NOT to process (encrypted rooms etc.) so we
// stop nagging on every message.
const skippedRooms = new Set<string>();

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

  /** Send plain-text message, splitting if too long. */
  const sendMessage = async (roomId: string, text: string): Promise<void> => {
    for (const part of splitMessage(text)) {
      try {
        await client.sendMessage(roomId, { msgtype: 'm.text', body: part });
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

  /** Read m.room.encryption state — true if the room is E2EE. */
  const isRoomEncrypted = async (roomId: string): Promise<boolean> => {
    try {
      const ev = await client.getRoomStateEvent(roomId, 'm.room.encryption', '');
      return !!(ev && (ev as { algorithm?: string }).algorithm);
    } catch {
      return false;
    }
  };

  /**
   * Core message handler — ported from signal-bot.ts handleTextMessage().
   * `chatId` here is the Matrix roomId. Authorization is checked at the
   * outer event listener; by the time we land here, the sender is known
   * to be allowlisted.
   */
  async function handleTextMessage(
    roomId: string,
    sender: string,
    eventId: string,
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
      : (userModel ?? 'claude-opus-4-6');

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
      const caps = voiceCapabilities();
      const mode = voiceMode.get(chatId);
      const shouldSpeakBack = caps.tts && (
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
  async function handleCommand(roomId: string, sender: string, eventId: string, text: string): Promise<boolean> {
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
        const caps = voiceCapabilities();
        if (!caps.tts) {
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
        await handleTextMessage(roomId, sender, eventId, `/delegate ${rest}`);
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

    // Encrypted rooms aren't supported in this MVP. Warn once per room.
    if (await isRoomEncrypted(roomId)) {
      if (!skippedRooms.has(roomId)) {
        skippedRooms.add(roomId);
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
      const caps = voiceCapabilities();
      if (!caps.stt) {
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
        await handleTextMessage(roomId, event.sender, event.event_id, transcript, /* forceVoiceReply */ true);
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
      messageQueue.enqueue(roomId, () => handleTextMessage(roomId, event.sender, event.event_id, caption));
      return;
    }

    // ── Text ────────────────────────────────────────────────────────────
    if (msgtype !== 'm.text' && msgtype !== 'm.notice' && msgtype !== 'm.emote') return;

    const text = body.trim();
    if (!text) return;

    // Commands first — they bypass the message queue so /stop can interrupt.
    if (text.startsWith('/')) {
      const handled = await handleCommand(roomId, event.sender, event.event_id, text);
      if (handled) return;
    }

    messageQueue.enqueue(roomId, () => handleTextMessage(roomId, event.sender, event.event_id, text));
  };

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
      logger.info(
        { homeserver: MATRIX_HOMESERVER_URL, userId: MATRIX_USER_ID, allowed: MATRIX_AUTHORIZED_USERS.length },
        'Matrix bot connected to homeserver',
      );
    },
    async stop(): Promise<void> {
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
