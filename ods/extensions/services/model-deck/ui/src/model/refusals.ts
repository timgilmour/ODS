/**
 * Which park refusals the Force button can actually override.
 *
 * `?force=true` skips the conversation/host-agent busy guards ONLY
 * (app/routers/control.py `_hipfire_park`: "?force=true skips the
 * conversation-guard, never the route guard"). Two park guards raise
 * regardless of force, so arming Force on their 409s offers a button that
 * refuses identically on click:
 *
 *  - the park allowlist — app/engines/docker_ctl.py `_guard`:
 *    `container {name!r} is not in the park allowlist`
 *  - the default-route guard — app/engines/hipfire.py `park()`, raised
 *    before the force check even runs:
 *    `litellm's default route currently targets hipfire`
 *
 * Matching is on those producers' exact phrases (cite-the-backend-line
 * rule). Unknown refusal text fails OPEN — Force stays offered, so a new
 * backend guard degrades to today's behavior (click 409s again), never to
 * a silently hidden override.
 */
export function forceParkCanOverride(detail: string): boolean {
  return (
    !detail.includes("is not in the park allowlist") &&
    !detail.includes("litellm's default route currently targets")
  );
}
