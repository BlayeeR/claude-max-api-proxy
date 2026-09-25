/**
 * Dynamic model catalog, discovered from the Claude CLI's own bundled catalog.
 *
 * The CLI ships a baked-in model list — the `{id:"claude-...",family:"...",
 * display_name:"..."}` literals inside its (minified) bundle, which is a plain
 * JS file for npm installs and an embedded bundle for native builds. Scanning
 * it at first use means new models appear as soon as the CLI is updated — no
 * proxy code changes needed. If the bundle can't be read (e.g. a Windows shim
 * rather than a real file), we fall back to the evergreen family aliases,
 * which the CLI always resolves to the latest model of each family.
 */

import { readFile } from "fs/promises";
import { existsSync, realpathSync, statSync } from "fs";
import path from "path";
import { resolveClaudeBin } from "../subprocess/manager.js";

export interface ModelEntry {
  id: string;
  family: string;
  displayName: string;
}

export interface ModelCatalog {
  /** CLI model aliases: family names plus specials like "best" */
  aliases: string[];
  /** Full model IDs discovered from the CLI bundle (empty if scan failed) */
  models: ModelEntry[];
}

/**
 * Evergreen CLI aliases. Family names are stable across releases; "best"
 * and "opusplan" are special routing modes. Families discovered in the
 * scanned catalog are added to this list automatically.
 */
const STATIC_ALIASES = [
  "sonnet",
  "opus",
  "haiku",
  "fable",
  "best",
  "opusplan",
];

const MODEL_ENTRY_RE =
  /\{id:"(claude-[a-z0-9-]+)",family:"([a-z0-9-]+)",display_name:"((?:[^"\\]|\\.)*)"/g;

let cached: ModelCatalog | null = null;
let scanPromise: Promise<ModelCatalog> | null = null;

/**
 * Get the model catalog. Scans the CLI bundle on first call (concurrent
 * callers share one scan) and caches the result for the process lifetime.
 */
export function getModelCatalog(): Promise<ModelCatalog> {
  if (cached) return Promise.resolve(cached);
  if (!scanPromise) {
    scanPromise = scanCliBundle()
      .catch((err) => {
        const reason = err instanceof Error ? err.message : String(err);
        console.error(
          "[Models] CLI bundle scan failed, using alias fallback:",
          reason
        );
        return fallbackCatalog();
      })
      .then((catalog) => {
        cached = catalog;
        return catalog;
      });
  }
  return scanPromise;
}

function fallbackCatalog(): ModelCatalog {
  return { aliases: [...STATIC_ALIASES], models: [] };
}

async function scanCliBundle(): Promise<ModelCatalog> {
  const filePath = resolveClaudeFilePath();
  if (!filePath) {
    throw new Error("no readable claude CLI file found on PATH");
  }

  // Latin1 decoding avoids UTF-8 multibyte issues over what is
  // effectively ASCII JS source (works for compiled binaries too)
  let raw = await readFile(filePath, "latin1");
  let extracted = extractCatalogEntries(raw);

  if (extracted.models.length === 0) {
    // The PATH hit may be an npm shim script — try the cli.js bundle it
    // launches, which contains the actual model catalog
    const cliJs = npmCliJsPath(filePath);
    if (cliJs) {
      raw = await readFile(cliJs, "latin1");
      extracted = extractCatalogEntries(raw);
    }
  }

  if (extracted.models.length === 0) {
    // Bundle read fine but contained no model entries — format changed.
    // Treat as scan failure so consumers get the alias fallback.
    throw new Error("no model catalog entries found in CLI bundle");
  }

  // Aliases: static set plus any new families found in the catalog
  const aliases = [...STATIC_ALIASES];
  for (const family of extracted.families) {
    if (!aliases.includes(family)) aliases.push(family);
  }

  console.log(
    `[Models] Discovered ${extracted.models.length} models from CLI bundle ` +
      `(${extracted.families.size} families: ${[...extracted.families].join(", ")})`
  );

  return { aliases, models: extracted.models };
}

function extractCatalogEntries(raw: string): {
  models: ModelEntry[];
  families: Set<string>;
} {
  const models: ModelEntry[] = [];
  const seen = new Set<string>();
  const families = new Set<string>();

  let match: RegExpExecArray | null;
  MODEL_ENTRY_RE.lastIndex = 0;
  while ((match = MODEL_ENTRY_RE.exec(raw)) !== null) {
    const [, id, family, displayName] = match;
    if (seen.has(id)) continue;
    seen.add(id);
    families.add(family);
    models.push({ id, family, displayName });
  }

  return { models, families };
}

/**
 * Resolve a readable path to the CLI bundle for scanning.
 *
 * `resolveClaudeBin` returns what to *spawn*: CLAUDE_BIN overrides,
 * resolved Windows exes, or the bare `claude` command name on PATH. For
 * scanning we need a real file path — search PATH ourselves and follow
 * symlinks (native installs symlink `claude` into ~/.local/bin).
 */
function resolveClaudeFilePath(): string | null {
  const { bin } = resolveClaudeBin();

  // CLAUDE_BIN override or manager-resolved Windows exe: use directly
  if (bin !== "claude") {
    return existsSync(bin) ? bin : null;
  }

  const exeNames =
    process.platform === "win32" ? ["claude.exe", "claude"] : ["claude"];
  const pathDirs = (process.env.PATH || "")
    .split(path.delimiter)
    .filter(Boolean);

  for (const dir of pathDirs) {
    for (const name of exeNames) {
      const candidate = path.join(dir, name);
      try {
        if (!statSync(candidate).isFile()) continue;
        // Follow symlinks to the real bundle file
        return realpathSync(candidate);
      } catch {
        // Not present or not readable — keep searching
      }
    }
  }
  return null;
}

/**
 * If `filePath` is an npm-generated shim, return the path of the cli.js
 * bundle it launches (relative to the shim's node_modules directory).
 */
function npmCliJsPath(filePath: string): string | null {
  const pkg = path.join(
    path.dirname(realpathSync(filePath)),
    "node_modules",
    "@anthropic-ai",
    "claude-code",
    "cli.js"
  );
  return existsSync(pkg) ? pkg : null;
}
