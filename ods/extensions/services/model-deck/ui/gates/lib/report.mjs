/** Render gate results. Failures sort first: a gate with forty green lines
 * and one red one is exactly the report someone skims. */
function ordered(rows) {
  return [...rows.filter((r) => !r.ok), ...rows.filter((r) => r.ok)];
}

export function renderMarkdown(rows, meta) {
  const passed = rows.filter((r) => r.ok).length;
  const failed = rows.length - passed;
  const lines = [
    `# deck-gate — ${meta.tier} tier`,
    "",
    `- started: ${meta.startedIso}`,
    `- target: ${meta.target}`,
    `- result: **${passed} passed / ${failed} failed**`,
    "",
  ];
  for (const r of ordered(rows)) {
    lines.push(`- ${r.ok ? "PASS" : "FAIL"} ${r.name}${r.detail ? " — " + r.detail : ""}`);
  }
  return lines.join("\n") + "\n";
}

export function renderJson(rows, meta) {
  const passed = rows.filter((r) => r.ok).length;
  return (
    JSON.stringify(
      { ...meta, passed, failed: rows.length - passed, rows: ordered(rows) },
      null,
      2,
    ) + "\n"
  );
}
