/**
 * API Route Handlers
 *
 * Implements OpenAI-compatible endpoints for Clawdbot integration.
 * Routes session-keyed requests through SessionPoolRouter; falls back to
 * ClaudeSubprocess for headerless or non-pooled requests.
 */

import type { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import { existsSync, readdirSync } from "fs";
import {
  ClaudeSubprocess,
  stageImages,
  cleanupImages,
} from "../subprocess/manager.js";
import {
  SessionPoolRouter,
  type ExecuteResult,
} from "../subprocess/router.js";
import { openaiToCli, openaiToCliDelta } from "../adapter/openai-to-cli.js";
import {
  cliResultToOpenai,
  createDoneChunk,
} from "../adapter/cli-to-openai.js";
import { getSession, setSession, clearSession } from "../subprocess/session-store.js";
import { getModelCatalog } from "../models/catalog.js";
import type { OpenAIChatRequest, OpenAIToolCall } from "../types/openai.js";
import type { ClaudeCliAssistant, ClaudeCliResult, ClaudeCliStreamEvent } from "../types/claude-cli.js";

// ---------------------------------------------------------------------------
// Module-level router reference (set by standalone.ts at startup)
// ---------------------------------------------------------------------------

let poolRouter: SessionPoolRouter | null = null;

export function setPoolRouter(router: SessionPoolRouter): void {
  poolRouter = router;
}

export function getPoolRouter(): SessionPoolRouter | null {
  return poolRouter;
}

/**
 * Check whether Claude CLI credentials exist at all. On a fresh container
 * with an empty /data volume, every request would otherwise run into the
 * CLI's onboarding/login failure - we can answer that upfront instead.
 */
function hasClaudeCredentials(): boolean {
  const dir = process.env.CLAUDE_CONFIG_DIR || `${process.env.HOME}/.claude`;
  try {
    if (!existsSync(dir)) return false;
    return readdirSync(dir).some((f) => f.endsWith(".json"));
  } catch {
    return false;
  }
}

/**
 * User-facing guidance when the Claude CLI's OAuth session has expired.
 * Returned as a normal assistant message instead of a raw 500 so the user
 * sees actionable steps directly in their chat client.
 */
const AUTH_EXPIRED_MESSAGE = [
  "Die Claude-Anmeldung auf dem Server ist abgelaufen – Anfragen sind derzeit nicht moeglich.",
  "",
  "Neuanmeldung direkt ueber die Admin-API (kein Server-Zugriff noetig):",
  "1. `POST /admin/relogin/start` (mit Admin-Key) – die Antwort enthaelt eine Login-URL.",
  "2. URL im Browser oeffnen und die Anmeldung bestaetigen.",
  "3. Den angezeigten Code per `POST /admin/relogin/complete` zurueckschicken.",
  "4. Danach funktioniert dieser Chat sofort wieder – die neuen Tokens liegen persistent im Volume.",
  "",
  "Details: siehe DOCKER.md, Abschnitt 'Re-Login ohne Container-Zugriff'.",
].join("\n");

/**
 * Parse a <tool_call>{...}</tool_call> envelope from Claude's text output.
 * Claude Code has no client-executed tool protocol in --print mode, so tool
 * requests arrive as text; we convert them into OpenAI tool_calls.
 */
function parseToolCall(text: string): { name: string; arguments: string } | null {
  const m = text.match(/<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/);
  if (!m) return null;
  try {
    const parsed = JSON.parse(m[1]);
    if (typeof parsed.name !== "string") return null;
    return {
      name: parsed.name,
      arguments:
        typeof parsed.arguments === "string"
          ? parsed.arguments
          : JSON.stringify(parsed.arguments ?? {}),
    };
  } catch {
    return null;
  }
}

let toolCallSeq = 0;
function nextToolCallId(): string {
  return `call_${Date.now().toString(36)}${(toolCallSeq++).toString(36)}`;
}

/** Build an OpenAI-style assistant message body (non-streaming) */
function authExpiredResponse(requestId: string, model: string) {
  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: AUTH_EXPIRED_MESSAGE },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

interface SessionContext {
  sessionKey: string | undefined;
  resume: boolean;
  messageCount: number;
}

/**
 * Resolve CLI input for a request, resuming a persisted Claude CLI session
 * when we have one for this `request.user` key instead of replaying the
 * full message history on every turn.
 */
function resolveCliInput(body: OpenAIChatRequest): {
  cliInput: ReturnType<typeof openaiToCli>;
  sessionKey: string | undefined;
  resume: boolean;
} {
  // Session resume requires a stable per-client identifier. Without `user`
  // we have no way to distinguish callers, so skip resume entirely rather
  // than fall back to a shared key that would cross-contaminate unrelated
  // conversations.
  const sessionKey = body.user;
  const existing = sessionKey ? getSession(sessionKey) : undefined;

  if (existing) {
    const cliInput = openaiToCliDelta(body, existing.messageCount);
    cliInput.sessionId = existing.claudeSessionId;
    // Effort is per-request, not per-session: honor it on resumed turns too
    cliInput.effort = openaiToCli(body).effort;
    return { cliInput, sessionKey, resume: true };
  }

  const cliInput = openaiToCli(body);
  if (sessionKey) {
    cliInput.sessionId = uuidv4(); // pin a known ID so we can --resume it later
  }
  return { cliInput, sessionKey, resume: false };
}

/**
 * Handle POST /v1/chat/completions
 *
 * Main endpoint for chat requests, supports both streaming and non-streaming
 */
export async function handleChatCompletions(
  req: Request,
  res: Response
): Promise<void> {
  const requestId = uuidv4().replace(/-/g, "").slice(0, 24);
  const body = req.body as OpenAIChatRequest;
  const stream = body.stream === true;
  const startTime = Date.now();
  const earlySessionKey = (req.headers["x-openclaw-session-key"] as string | undefined) || (body as any).sessionId;
  console.log(JSON.stringify({
    ts: new Date().toISOString(),
    event: "request_received",
    requestId,
    sessionKey: earlySessionKey || "(none)",
    model: body.model || "(none)",
    stream,
    messageCount: body.messages?.length ?? 0,
    contentLength: req.headers["content-length"] || "(none)",
  }));

  try {
    // Validate request
    if (
      !body.messages ||
      !Array.isArray(body.messages) ||
      body.messages.length === 0
    ) {
      res.status(400).json({
        error: {
          message: "messages is required and must be a non-empty array",
          type: "invalid_request_error",
          code: "invalid_messages",
        },
      });
      return;
    }

    // Fresh container without any login: answer upfront instead of
    // letting the request run into the CLI's onboarding failure
    if (!hasClaudeCredentials()) {
      console.error("[Auth] No Claude credentials found - prompting admin to run the relogin flow");
      const guidance = authExpiredResponse(requestId, "claude-sonnet-4");
      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.flushHeaders();
        const chunk = {
          id: guidance.id,
          object: "chat.completion.chunk",
          created: guidance.created,
          model: guidance.model,
          choices: [
            { index: 0, delta: { role: "assistant", content: guidance.choices[0].message.content }, finish_reason: null },
            { index: 0, delta: {}, finish_reason: "stop" },
          ],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      } else {
        res.json(guidance);
      }
      return;
    }

    // --- Pool routing (session-keyed requests through warm processes) ---
    const poolKey =
      (req.headers["x-openclaw-session-key"] as string | undefined) ||
      (body.user as string | undefined);
    const agentId = req.headers["x-openclaw-agent-id"] as string | undefined;

    if (poolKey && poolRouter) {
      const pooledInput = openaiToCli(body);
      const result = poolRouter.execute(
        pooledInput.prompt,
        pooledInput.latestPrompt,
        pooledInput.model,
        poolKey,
        body.messages.length
      );

      if (result) {
        // Pooled route
        const { emitter, routeType, pid, queueDepth } = result;
        console.log(JSON.stringify({
          ts: new Date().toISOString(),
          event: "request_routed",
          requestId,
          sessionKey: poolKey,
          model: pooledInput.model,
          routeType,
          pid,
          queueDepth,
          elapsedMs: Date.now() - startTime,
        }));

        if (stream) {
          await handlePooledStreaming(
            req,
            res,
            emitter,
            requestId,
            startTime,
            poolKey,
            agentId,
            pooledInput.model,
            routeType,
            pid,
            queueDepth
          );
        } else {
          await handlePooledNonStreaming(
            res,
            emitter,
            requestId,
            startTime,
            poolKey,
            agentId,
            pooledInput.model,
            routeType,
            pid,
            queueDepth
          );
        }
        return;
      }
      // result === null → fall through to ClaudeSubprocess
    }

    // --- Fallback: ClaudeSubprocess (no session key, unpooled model, or at capacity) ---
    // Convert to CLI input format, resuming a persisted session when we have one
    const { cliInput, sessionKey, resume } = resolveCliInput(body);
    const subprocess = new ClaudeSubprocess();
    const sessionCtx: SessionContext = { sessionKey, resume, messageCount: body.messages.length };

    // Stage attached images (OpenAI image_url blocks) as temp files and point
    // the prompt at them - Claude Code reads them via its Read tool
    let stagedImages: string[] = [];
    if (cliInput.images && cliInput.images.length > 0) {
      stagedImages = await stageImages(cliInput.images);
      delete cliInput.images; // don't hold base64 in memory longer than needed
      if (stagedImages.length > 0) {
        const listing = stagedImages.map((p, i) => `${i + 1}. ${p}`).join("\n");
        cliInput.prompt =
          `[Attached images - use your Read tool on each file to view them]\n${listing}\n\n` +
          cliInput.prompt;
      }
    }

    try {
      if (stream) {
        await handleStreamingResponse(req, res, subprocess, cliInput, requestId, sessionCtx);
      } else {
        await handleNonStreamingResponse(res, subprocess, cliInput, requestId, sessionCtx);
      }
    } finally {
      if (stagedImages.length > 0) {
        await cleanupImages(stagedImages);
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[handleChatCompletions] Error:", message);

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message,
          type: "server_error",
          code: null,
        },
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Pooled streaming response
// ---------------------------------------------------------------------------

async function handlePooledStreaming(
  _req: Request,
  res: Response,
  emitter: ExecuteResult["emitter"],
  requestId: string,
  startTime: number,
  sessionKey: string,
  agentId: string | undefined,
  model: string,
  routeType: string,
  pid: number | null,
  queueDepth: number
): Promise<void> {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);
  res.flushHeaders();
  res.write(":ok\n\n");

  return new Promise<void>((resolve) => {
    let isFirst = true;
    let lastModel = "claude-sonnet-4";
    let isComplete = false;
    let hasEmittedText = false;



    const onTextBlockStart = () => {
      if (hasEmittedText && !res.writableEnded) {
        const sepChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [
            { index: 0, delta: { content: "\n\n" }, finish_reason: null },
          ],
        };
        res.write(`data: ${JSON.stringify(sepChunk)}\n\n`);
      }
    };

    const onContentDelta = (event: ClaudeCliStreamEvent) => {
      const delta = event.event.delta;
      const text = (delta?.type === "text_delta" && delta.text) || "";
      if (text && !res.writableEnded) {
        const chunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [
            {
              index: 0,
              delta: {
                role: isFirst ? ("assistant" as const) : undefined,
                content: text,
              },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        isFirst = false;
        hasEmittedText = true;
      }
    };

    const onAssistant = (message: ClaudeCliAssistant) => {
      lastModel = message.message.model;
    };

    const onResult = (result: ClaudeCliResult) => {
      isComplete = true;
      const latencyMs = Date.now() - startTime;
      console.log(
        JSON.stringify({
          ts: new Date().toISOString(),
          event: "request",
          sessionKey,
          agentId,
          model,
          pid,
          latencyMs,
          queueDepth,
          routeType,
          cacheHit: routeType,
          requestCount: result.num_turns,
        })
      );

      if (!res.writableEnded) {
        const doneChunk = createDoneChunk(requestId, lastModel);
        if (result.usage) {
          doneChunk.usage = {
            prompt_tokens: result.usage.input_tokens || 0,
            completion_tokens: result.usage.output_tokens || 0,
            total_tokens:
              (result.usage.input_tokens || 0) +
              (result.usage.output_tokens || 0),
          };
        }
        res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      }
      resolve();
    };

    const onError = (error: Error) => {
      isComplete = true;
      emitter.removeListener("text_block_start", onTextBlockStart);
      emitter.removeListener("content_delta", onContentDelta);
      emitter.removeListener("assistant", onAssistant);
      emitter.removeListener("result", onResult);
      const latencyMs = Date.now() - startTime;
      const errWithStatus = error as Error & {
        statusCode?: number;
        retryAfter?: number;
      };

      console.error(
        JSON.stringify({
          ts: new Date().toISOString(),
          event: "request_error",
          sessionKey,
          model,
          pid,
          latencyMs,
          routeType,
          error: error.message,
        })
      );

      if (!res.headersSent) {
        const status = errWithStatus.statusCode || 500;
        if (status === 429) {
          res.setHeader("Retry-After", String(errWithStatus.retryAfter || 5));
        }
        res.status(status).json({
          error: {
            message: error.message,
            type: status === 429 ? "rate_limit_error" : "server_error",
            code: null,
          },
        });
      } else if (!res.writableEnded) {
        res.write(
          `data: ${JSON.stringify({
            error: {
              message: error.message,
              type: "server_error",
              code: null,
            },
          })}\n\n`
        );
        res.end();
      }
      resolve();
    };

    // Client disconnect: remove request-specific listeners and release the
    // CLI process so it doesn't stay "busy" with a response nobody consumes.
    res.on("close", () => {
      if (!isComplete) {
        emitter.removeListener("text_block_start", onTextBlockStart);
        emitter.removeListener("content_delta", onContentDelta);
        emitter.removeListener("assistant", onAssistant);
        emitter.removeListener("result", onResult);
        emitter.removeListener("error", onError);

        // Tell the router to kill+respawn the process — it's mid-response
        // with buffered output and no consumer.
        const router = getPoolRouter();
        if (router && sessionKey) {
          router.cancelRequest(sessionKey);
        }
      }
      resolve();
    });

    emitter.on("text_block_start", onTextBlockStart);
    emitter.on("content_delta", onContentDelta);
    emitter.on("assistant", onAssistant);
    emitter.on("result", onResult);
    emitter.on("error", onError);
  });
}

// ---------------------------------------------------------------------------
// Pooled non-streaming response
// ---------------------------------------------------------------------------

async function handlePooledNonStreaming(
  res: Response,
  emitter: ExecuteResult["emitter"],
  requestId: string,
  startTime: number,
  sessionKey: string,
  agentId: string | undefined,
  model: string,
  routeType: string,
  pid: number | null,
  queueDepth: number
): Promise<void> {
  return new Promise((resolve) => {
    emitter.on("result", (result: ClaudeCliResult) => {
      const latencyMs = Date.now() - startTime;
      console.log(
        JSON.stringify({
          ts: new Date().toISOString(),
          event: "request",
          sessionKey,
          agentId,
          model,
          pid,
          latencyMs,
          queueDepth,
          routeType,
          cacheHit: routeType,
          requestCount: result.num_turns,
        })
      );
      res.json(cliResultToOpenai(result, requestId));
      resolve();
    });

    emitter.on("error", (error: Error) => {
      const latencyMs = Date.now() - startTime;
      const errWithStatus = error as Error & {
        statusCode?: number;
        retryAfter?: number;
      };

      console.error(
        JSON.stringify({
          ts: new Date().toISOString(),
          event: "request_error",
          sessionKey,
          model,
          pid,
          latencyMs,
          routeType,
          error: error.message,
        })
      );

      if (!res.headersSent) {
        const status = errWithStatus.statusCode || 500;
        if (status === 429) {
          res.setHeader("Retry-After", String(errWithStatus.retryAfter || 5));
        }
        res.status(status).json({
          error: {
            message: error.message,
            type: status === 429 ? "rate_limit_error" : "server_error",
            code: null,
          },
        });
      }
      resolve();
    });
  });
}

// ---------------------------------------------------------------------------
// Fallback: ClaudeSubprocess streaming (existing behavior, unchanged)
// ---------------------------------------------------------------------------

async function handleStreamingResponse(
  req: Request,
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  sessionCtx: SessionContext
): Promise<void> {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);
  res.flushHeaders();
  res.write(":ok\n\n");

  return new Promise<void>((resolve, reject) => {
    let isFirst = true;
    let lastModel = "claude-sonnet-4";
    let isComplete = false;
    let hasEmittedText = false;
    let toolCallIndex = 0;
    let inToolBlock = false;
    let accumulatedText = "";
    // When the client provides tools, Claude asks for them via a <tool_call>
    // text envelope - buffer text so we can convert it into real tool_calls
    const expectsClientTools = !!cliInput.hasClientTools;

    res.on("close", () => {
      if (!isComplete) {
        subprocess.kill();
      }
      resolve();
    });

    subprocess.on("text_block_start", () => {
      if (hasEmittedText && !res.writableEnded) {
        const sepChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [
            {
              index: 0,
              delta: { content: "\n\n" },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(sepChunk)}\n\n`);
      }
    });

    // Extended thinking: forward as OpenAI-style reasoning chunks
    // (OpenWebUI renders delta.reasoning as a collapsible "thinking" section).
    // Note: subscription-billed thinking is often redacted/encrypted server-side
    // (empty deltas with only estimated_tokens) - those are skipped here.
    subprocess.on("thinking_delta", (event: ClaudeCliStreamEvent) => {
      const delta = event.event.delta;
      const text = (delta?.type === "thinking_delta" && delta.thinking) || "";
      if (text && !res.writableEnded) {
        const chunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: { reasoning: text },
            finish_reason: null,
          }],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
    });

    // Handle streaming content deltas
    subprocess.on("content_delta", (event: ClaudeCliStreamEvent) => {
      const delta = event.event.delta;
      let text = (delta?.type === "text_delta" && delta.text) || "";
      if (expectsClientTools) {
        // Buffer instead of streaming: the text may be a <tool_call> envelope
        // that must be converted into tool_calls, never shown to the user
        accumulatedText += text;
        return;
      }
      // CLI surfaces auth failures as plain text on stdout in some failure
      // modes - swallow them here; the error/close handlers emit the guidance
      // message exactly once.
      if (text && subprocess.hasAuthError()) {
        return;
      }
      if (text && !res.writableEnded) {
        const chunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [
            {
              index: 0,
              delta: {
                role: isFirst ? ("assistant" as const) : undefined,
                content: text,
              },
              finish_reason: null,
            },
          ],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        isFirst = false;
        hasEmittedText = true;
      }
    });

    subprocess.on("assistant", (message: ClaudeCliAssistant) => {
      lastModel = message.message.model;
    });

    subprocess.on("result", (result: ClaudeCliResult) => {
      isComplete = true;
      if (sessionCtx.sessionKey && cliInput.sessionId) {
        setSession(sessionCtx.sessionKey, cliInput.sessionId, sessionCtx.messageCount);
      }
      // Client-tool request? The streamed text is a <tool_call> envelope -
      // replace it with proper OpenAI tool_calls chunks
      if (expectsClientTools) {
        const tc = parseToolCall(accumulatedText || result.result || "");
        if (tc && !res.writableEnded) {
          const callId = nextToolCallId();
          const chunk = {
            id: `chatcmpl-${requestId}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: lastModel,
            choices: [{
              index: 0,
              delta: {
                role: "assistant",
                tool_calls: [{
                  index: 0,
                  id: callId,
                  type: "function",
                  function: { name: tc.name, arguments: tc.arguments },
                }],
              },
              finish_reason: "tool_calls",
            }],
          };
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          res.write("data: [DONE]\n\n");
          res.end();
          resolve();
          return;
        }
      }
      // No tool call detected - flush the buffered text as regular content
      if (expectsClientTools && accumulatedText && !res.writableEnded) {
        const textChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: { role: "assistant", content: accumulatedText },
            finish_reason: null,
          }],
        };
        res.write(`data: ${JSON.stringify(textChunk)}\n\n`);
        hasEmittedText = true;
        isFirst = false;
      }
      if (!res.writableEnded) {
        const doneChunk = createDoneChunk(requestId, lastModel);
        if (result.usage) {
          doneChunk.usage = {
            prompt_tokens: result.usage.input_tokens || 0,
            completion_tokens: result.usage.output_tokens || 0,
            total_tokens:
              (result.usage.input_tokens || 0) + (result.usage.output_tokens || 0),
            // Prompt caching is automatic in Claude Code - surface the metrics
            ...(result.usage.cache_read_input_tokens
              ? { cache_read_input_tokens: result.usage.cache_read_input_tokens }
              : {}),
            ...(result.usage.cache_creation_input_tokens
              ? { cache_creation_input_tokens: result.usage.cache_creation_input_tokens }
              : {}),
          };
        }
        res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      }
      resolve();
    });

    subprocess.on("error", (error: Error) => {
      console.error("[Streaming] Error:", error.message);
      // Resume may have failed (e.g. stale/missing session) — drop it so the
      // next turn self-heals with a fresh full-history session
      if (sessionCtx.resume && sessionCtx.sessionKey) {
        clearSession(sessionCtx.sessionKey);
      }
      if (subprocess.hasAuthError()) {
        console.error("[Auth] Claude CLI authentication expired - user notified in chat");
        if (!res.writableEnded) {
          const chunk = {
            id: `chatcmpl-${requestId}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: lastModel,
            choices: [{ index: 0, delta: { role: "assistant", content: AUTH_EXPIRED_MESSAGE }, finish_reason: null }],
          };
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          res.write("data: [DONE]\n\n");
          res.end();
        }
        resolve();
        return;
      }
      if (!res.writableEnded) {
        res.write(
          `data: ${JSON.stringify({
            error: {
              message: error.message,
              type: "server_error",
              code: null,
            },
          })}\n\n`
        );
        res.end();
      }
      resolve();
    });

    subprocess.on("close", (code: number | null) => {
      // Subprocess exited - ensure response is closed
      if (code !== 0 && !isComplete) {
        if (sessionCtx.resume && sessionCtx.sessionKey) {
          clearSession(sessionCtx.sessionKey);
        }
        if (!res.writableEnded) {
          // Abnormal exit without result - send error
          res.write(`data: ${JSON.stringify({
            error: { message: `Process exited with code ${code}`, type: "server_error", code: null },
          })}\n\n`);
        }
      }
      if (!res.writableEnded) {
        res.write("data: [DONE]\n\n");
        res.end();
      }
      resolve();
    });

    // Start the subprocess
    subprocess.start(cliInput.prompt, {
      model: cliInput.model,
      sessionId: cliInput.sessionId,
      resume: sessionCtx.resume,
      effort: cliInput.effort,
      disableBuiltinTools: cliInput.hasClientTools,
    }).catch((err) => {
      console.error("[Streaming] Subprocess start error:", err);
      reject(err);
    });
  });
}

// ---------------------------------------------------------------------------
// Fallback: ClaudeSubprocess non-streaming (existing behavior, unchanged)
// ---------------------------------------------------------------------------

async function handleNonStreamingResponse(
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  sessionCtx: SessionContext
): Promise<void> {
  return new Promise((resolve) => {
    let finalResult: ClaudeCliResult | null = null;

    subprocess.on("result", (result: ClaudeCliResult) => {
      finalResult = result;
    });

    subprocess.on("error", (error: Error) => {
      console.error("[NonStreaming] Error:", error.message);
      if (sessionCtx.resume && sessionCtx.sessionKey) {
        clearSession(sessionCtx.sessionKey);
      }
      if (subprocess.hasAuthError()) {
        console.error("[Auth] Claude CLI authentication expired - user notified in chat");
        res.json(authExpiredResponse(requestId, "claude-sonnet-4"));
        resolve();
        return;
      }
      res.status(500).json({
        error: {
          message: error.message,
          type: "server_error",
          code: null,
        },
      });
      resolve();
    });

    subprocess.on("close", (code: number | null) => {
      if (finalResult) {
        if (sessionCtx.sessionKey && cliInput.sessionId) {
          setSession(sessionCtx.sessionKey, cliInput.sessionId, sessionCtx.messageCount);
        }
        // Client-tool request? Convert <tool_call> text into OpenAI tool_calls
        if (cliInput.hasClientTools) {
          const tc = parseToolCall(finalResult.result || "");
          if (tc) {
            res.json({
              id: `chatcmpl-${requestId}`,
              object: "chat.completion",
              created: Math.floor(Date.now() / 1000),
              model: "claude-sonnet-4",
              choices: [{
                index: 0,
                message: {
                  role: "assistant",
                  content: null,
                  tool_calls: [{
                    id: nextToolCallId(),
                    type: "function",
                    function: { name: tc.name, arguments: tc.arguments },
                  }],
                },
                finish_reason: "tool_calls",
              }],
              usage: {
                prompt_tokens: finalResult.usage?.input_tokens || 0,
                completion_tokens: finalResult.usage?.output_tokens || 0,
                total_tokens:
                  (finalResult.usage?.input_tokens || 0) +
                  (finalResult.usage?.output_tokens || 0),
              },
            });
            resolve();
            return;
          }
        }
        res.json(cliResultToOpenai(finalResult, requestId));
      } else {
        if (sessionCtx.resume && sessionCtx.sessionKey) {
          clearSession(sessionCtx.sessionKey);
        }
        if (!res.headersSent) {
          if (subprocess.hasAuthError()) {
            console.error("[Auth] Claude CLI authentication expired - user notified in chat");
            res.json(authExpiredResponse(requestId, "claude-sonnet-4"));
          } else {
            res.status(500).json({
              error: {
                message: `Claude CLI exited with code ${code} without response`,
                type: "server_error",
                code: null,
              },
            });
          }
        }
      }
      resolve();
    });

    subprocess
      .start(cliInput.prompt, {
        model: cliInput.model,
        sessionId: cliInput.sessionId,
        resume: sessionCtx.resume,
        effort: cliInput.effort,
        disableBuiltinTools: cliInput.hasClientTools,
      })
      .catch((error) => {
        res.status(500).json({
          error: {
            message: error.message,
            type: "server_error",
            code: null,
          },
        });
        resolve();
      });
  });
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

/**
 * Returns the model list discovered from the CLI's bundled catalog
 * (aliases + full model IDs) — see src/models/catalog.ts
 */
export async function handleModels(_req: Request, res: Response): Promise<void> {
  const now = Math.floor(Date.now() / 1000);
  const { aliases, models } = await getModelCatalog();
  const modelIds = [...aliases, ...models.map((m) => m.id)];
  res.json({
    object: "list",
    data: modelIds.map((id) => ({
      id,
      object: "model",
      owned_by: "anthropic",
      created: now,
    })),
  });
}

// ---------------------------------------------------------------------------
// GET /health — includes pool stats when available
// ---------------------------------------------------------------------------

export function handleHealth(_req: Request, res: Response): void {
  const base: Record<string, unknown> = {
    status: "ok",
    provider: "claude-code-cli",
    timestamp: new Date().toISOString(),
  };

  if (poolRouter) {
    base.pool = poolRouter.stats();
  }

  res.json(base);
}
