import { useEffect, useState } from "react";
import { getEvents, type EventEntry } from "../api";

interface EventLogProps {
  /** Bumped by App on every 3s poll tick (and after any mutating action),
   * so this component re-fetches its own last-50 window on the same cadence
   * without App having to own event state itself. */
  refreshTrigger: number;
}

export default function EventLog({ refreshTrigger }: EventLogProps) {
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvents(50)
      .then((evs) => {
        setEvents(evs);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [refreshTrigger]);

  const newestFirst = [...events].reverse();

  return (
    <div className="panel">
      <h2>Events</h2>
      {error && <div className="banner-error"><span>{error}</span></div>}
      <div className="event-log-body">
        {newestFirst.length === 0 ? (
          <div>no events yet</div>
        ) : (
          newestFirst.map((e, i) => (
            <div className="event-line" key={i}>
              {e.ts} {e.kind} {JSON.stringify(e.detail)}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
