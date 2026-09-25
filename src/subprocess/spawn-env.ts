/**
 * Environment for spawning the Claude CLI.
 *
 * The CLI refuses --dangerously-skip-permissions when running as root
 * ("cannot be used with root/sudo privileges for security reasons") unless
 * IS_SANDBOX is set - the CLI's own escape hatch for container deployments.
 * Without this, every pooled/fallback process exits code 1 instantly when
 * the container runs as root.
 */

/** True when this process runs as root (uid 0) on a POSIX system */
export function isRunningAsRoot(): boolean {
  return typeof process.getuid === "function" && process.getuid() === 0;
}

/**
 * Build the child env for a Claude CLI spawn: inherits the parent
 * environment (minus CLAUDECODE) and sets IS_SANDBOX=1 when running as
 * root so --dangerously-skip-permissions is permitted.
 */
export function cliSpawnEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = Object.fromEntries(
    Object.entries(process.env).filter(([k]) => k !== "CLAUDECODE")
  );
  if (isRunningAsRoot() && !env.IS_SANDBOX) {
    env.IS_SANDBOX = "1";
  }
  return env;
}
