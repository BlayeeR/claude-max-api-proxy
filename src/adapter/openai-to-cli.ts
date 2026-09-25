/**
 * Converts OpenAI chat request format to Claude CLI input
 */

import type { OpenAIChatRequest, OpenAIContentBlock } from "../types/openai.js";

/**
 * Model string passed to `claude --model`. The CLI accepts family aliases
 * ("opus", "sonnet", "haiku", "fable", ...) and full model IDs
 * ("claude-fable-5", ...) — anything else it rejects itself.
 */
export type ClaudeModel = string;

export interface CliInput {
  prompt: string;
  model: ClaudeModel;
  sessionId?: string;
}

/** Model families the CLI resolves to their latest release */
const FAMILIES = ["opus", "sonnet", "haiku", "fable"];

// Matches short-form family names like "claude-sonnet-4" or
// "claude-opus-4-6" (but not exact ids like "claude-3-5-haiku")
const SHORT_FAMILY_RE = new RegExp(
  `^claude-(${FAMILIES.join("|")})(?:-\\d[\\w-]*)?$`
);

/**
 * Extract the model string to pass to `claude --model`.
 *
 * No hardcoded version list: full model IDs pass through for the CLI to
 * resolve or reject, and short-form family names resolve to the family
 * alias so they always track the latest release.
 */
export function extractModel(model: string): ClaudeModel {
  // Strip provider prefixes like `claude-code-cli/` and `claude-max/`
  const stripped = model.replace(/^(?:claude-code-cli|claude-max)\//, "");

  if (FAMILIES.includes(stripped) || stripped === "best") {
    return stripped;
  }

  // "claude-sonnet-4", "claude-opus-4-6" → family alias (latest)
  const shortFamily = stripped.match(SHORT_FAMILY_RE);
  if (shortFamily) {
    return shortFamily[1];
  }

  // Full model IDs ("claude-fable-5-1", "claude-3-5-haiku", ...) pass
  // through — the CLI resolves them, so new releases work without
  // proxy changes. Unknown IDs fail in the CLI, which is correct.
  if (/^claude-[a-z0-9-]+$/.test(stripped)) {
    return stripped;
  }

  // Default for unrecognizable input (Claude Max subscription)
  return "opus";
}

/**
 * Extract text from a content field that may be a string or array of content blocks.
 * OpenAI API allows content as either:
 *   - A plain string: "Hello"
 *   - An array of content blocks: [{"type": "text", "text": "Hello"}]
 */
function extractText(content: string | OpenAIContentBlock[]): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((block) => block.type === "text" || block.type === "input_text")
      .map((block) => block.text)
      .join("\n");
  }
  return String(content || "");
}

/**
 * Strip OpenClaw-specific tooling sections from system prompts.
 * These reference tools (exec, process, web_search, etc.) that don't exist
 * in the Claude Code CLI environment, causing the model to get confused.
 * We remove: ## Tooling, ## Tool Call Style, ## OpenClaw CLI Quick Reference,
 * ## OpenClaw Self-Update
 */
function stripOpenClawTooling(text: string): string {
  const sectionsToStrip = [
    "## Tooling",
    "## Tool Call Style",
    "## OpenClaw CLI Quick Reference",
    "## OpenClaw Self-Update",
  ];
  let result = text;
  for (const section of sectionsToStrip) {
    // Match from section header to the next ## header (or end of string)
    const pattern = new RegExp(
      section.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") +
        "\\n[\\s\\S]*?(?=\\n## |$)",
      "g"
    );
    result = result.replace(pattern, "");
  }
  // Clean up excessive blank lines left behind
  result = result.replace(/\n{3,}/g, "\n\n");
  return result.trim();
}

/**
 * Convert OpenAI messages array to a single prompt string for Claude CLI
 *
 * Claude Code CLI in --print mode expects a single prompt, not a conversation.
 * We format the messages into a readable format that preserves context.
 */
export function messagesToPrompt(
  messages: OpenAIChatRequest["messages"]
): string {
  const parts: string[] = [];

  for (const msg of messages) {
    const text = extractText(msg.content);
    switch (msg.role) {
      case "system":
        // System messages become context instructions
        // Strip OpenClaw tooling sections that conflict with Claude Code's native tools
        parts.push(`<system>\n${stripOpenClawTooling(text)}\n</system>\n`);
        break;

      case "user":
        // User messages are the main prompt
        parts.push(text);
        break;

      case "assistant":
        // Previous assistant responses for context
        parts.push(`<previous_response>\n${text}\n</previous_response>\n`);
        break;
    }
  }

  return parts.join("\n").trim();
}

/**
 * Convert OpenAI chat request to CLI input format
 */
export function openaiToCli(request: OpenAIChatRequest): CliInput {
  return {
    prompt: messagesToPrompt(request.messages),
    model: extractModel(request.model),
    sessionId: request.user, // Use OpenAI's user field for session mapping
  };
}

/**
 * Build CLI input for a request that will --resume an existing Claude CLI
 * session. Since the CLI already remembers everything up to `sinceIndex`
 * (it generated the assistant turns itself), we only need to forward the
 * messages appended since then — not the full history again.
 */
export function openaiToCliDelta(
  request: OpenAIChatRequest,
  sinceIndex: number
): CliInput {
  const newMessages = request.messages
    .slice(sinceIndex)
    .filter((m) => m.role !== "assistant");

  return {
    // Fallback to full history if nothing new was found (shouldn't happen,
    // but never send an empty prompt to the CLI)
    prompt: messagesToPrompt(newMessages.length ? newMessages : request.messages),
    model: extractModel(request.model),
    sessionId: request.user,
  };
}
