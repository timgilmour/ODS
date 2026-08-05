import { useEffect, useMemo, useState } from "react";
import { getEvents, type EventEntry } from "../api";
import { labels, messages } from "../model/messages";
import Panel from "../ui/Panel";
import Toolbar, { type Filter } from "../ui/Toolbar";

const SEVERITY: Record<string, string> = {
  refused: "ui-pill-bad",
  error: "ui-pill-bad",
  reconciled: "ui-pill-good",
  load: "ui-pill-off",
  unload: "ui-pill-off",
  pull: "ui-pill-off",
};

/** The full event view. Replaces the strip that used to sit under the board:
 * eighteen rows of repeated noise on the board buried the one line that
 * mattered, which is what search and severity filtering are here to fix. */
export default function EventsView({ refreshTrigger }: { refreshTrigger: number }) {
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [kinds, setKinds] = useState<string[]>([]);

  useEffect(() => {
    getEvents(200).then(
      (evs) => {
        setEvents(evs);
        setError(null);
      },
      (err) => setError(err instanceof Error ? err.message : String(err)),
    );
  }, [refreshTrigger]);

  const filters: Filter[] = useMemo(() => {
    const seen = [...new Set(events.map((e) => e.kind))].sort();
    return seen.map((k) => ({ id: k, label: k, active: kinds.includes(k) }));
  }, [events, kinds]);

  const rows = useMemo(() => {
    const needle = search.toLowerCase();
    return [...events]
      .reverse()
      .filter((e) => kinds.length === 0 || kinds.includes(e.kind))
      .filter(
        (e) =>
          needle === "" ||
          e.kind.toLowerCase().includes(needle) ||
          JSON.stringify(e.detail).toLowerCase().includes(needle),
      );
  }, [events, kinds, search]);

  return (
    <Panel title={labels.events}>
      <Toolbar
        search={search}
        onSearch={setSearch}
        placeholder={labels.filterEvents}
        filters={filters}
        onToggleFilter={(id) =>
          setKinds((cur) => (cur.includes(id) ? cur.filter((k) => k !== id) : [...cur, id]))
        }
      />
      {error && <div className="banner-error">{error}</div>}
      <div className="event-table">
        {rows.length === 0 ? (
          <div>{messages.noEvents().title}</div>
        ) : (
          rows.map((e, i) => (
            <div className="event-row" key={`${e.ts}-${i}`}>
              <span className="event-ts">{e.ts}</span>
              <span className={`ui-pill ${SEVERITY[e.kind] ?? "ui-pill-off"}`}>{e.kind}</span>
              <span className="event-detail">{JSON.stringify(e.detail)}</span>
            </div>
          ))
        )}
      </div>
    </Panel>
  );
}
