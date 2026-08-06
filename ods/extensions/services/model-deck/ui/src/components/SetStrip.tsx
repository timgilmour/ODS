import { useEffect, useState } from "react";
import { getSets, slugify, PREVIOUS_SLUG, type ConfigSet } from "../api";
import { messages } from "../model/messages";
import Banner from "../ui/Banner";
import ApplyModal from "./ApplyModal";

interface SetStripProps {
  onModalOpenChange: (open: boolean) => void;
  onChanged: () => void;
}

interface ActiveSet {
  slug: string;
  cfgset: ConfigSet;
}

/** Horizontal row of saved config-set buttons. Click -> ApplyModal (preview
 * -> confirm -> apply -> report). Close refetches everything (both this
 * component's own set list and the parent's world state, via onChanged). */
export default function SetStrip({ onModalOpenChange, onChanged }: SetStripProps) {
  const [sets, setSets] = useState<ConfigSet[]>([]);
  const [previous, setPrevious] = useState<ConfigSet | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [active, setActive] = useState<ActiveSet | null>(null);

  function refreshList() {
    getSets()
      .then((r) => {
        setSets(r.sets);
        setPrevious(r.previous);
        setListError(null);
      })
      .catch((err) => setListError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(refreshList, []);

  function openPreview(slug: string, cfgset: ConfigSet) {
    setActive({ slug, cfgset });
    onModalOpenChange(true);
  }

  function closeModal() {
    setActive(null);
    onModalOpenChange(false);
    refreshList();
    onChanged();
  }

  return (
    <div className="panel">
      <h2>Config sets</h2>
      {listError && <Banner message={messages.stateRefreshFailed(listError)} />}
      <div className="set-strip">
        {sets.map((s) => (
          <button key={s.name} onClick={() => openPreview(slugify(s.name), s)}>
            {s.name}
          </button>
        ))}
        {previous && (
          <button
            className="set-btn-previous"
            onClick={() => openPreview(PREVIOUS_SLUG, previous)}
          >
            {previous.name}
          </button>
        )}
        {sets.length === 0 && !previous && <span>no saved sets</span>}
      </div>

      {active && <ApplyModal slug={active.slug} cfgset={active.cfgset} onClose={closeModal} />}
    </div>
  );
}
