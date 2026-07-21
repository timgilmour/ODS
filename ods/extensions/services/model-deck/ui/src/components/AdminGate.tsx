import { useState } from "react";
import { TOKEN_KEY } from "../api";

interface AdminGateProps {
  token: string;
  onTokenChange: (token: string) => void;
}

/** Header token control: password-type input + Set/Clear, persisted to
 * localStorage here (the one place that writes deck-token); api.ts reads
 * it back at call time. */
export default function AdminGate({ token, onTokenChange }: AdminGateProps) {
  const [draft, setDraft] = useState("");

  function handleSet() {
    if (!draft) return;
    localStorage.setItem(TOKEN_KEY, draft);
    onTokenChange(draft);
    setDraft("");
  }

  function handleClear() {
    localStorage.removeItem(TOKEN_KEY);
    onTokenChange("");
    setDraft("");
  }

  return (
    <div className="admin-gate">
      <input
        type="password"
        aria-label="admin token"
        placeholder="admin token"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSet();
        }}
      />
      <button onClick={handleSet} disabled={!draft}>
        Set
      </button>
      <button onClick={handleClear} disabled={!token}>
        Clear
      </button>
      <span className="admin-gate-status">{token ? "admin" : "read-only"}</span>
    </div>
  );
}
