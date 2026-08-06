import { useState } from "react";
import { slugify, type ConfigSet } from "../api";
import { labels, messages } from "../model/messages";
import Banner from "../ui/Banner";
import Panel from "../ui/Panel";

/** The visible saved-sets list the original builder hid in a dropdown —
 * save/load/duplicate/delete were always in the API (/api/sets); this panel
 * simply surfaces them (build design, phase 2). Delete is two-click, armed
 * per row and disarmed by any other row's arming or a list refresh.
 *
 * Every button here is rendered INSIDE SetBuilder's `<fieldset disabled>`,
 * so an overwrite confirmation pending upstream locks these rows too — see
 * CRITICAL 1 in SetBuilder. Nothing in this file re-decides that; it just
 * must not be lifted out of the fieldset. */
export default function SavedSets({
  sets,
  listError,
  onLoad,
  onDuplicate,
  onDelete,
}: {
  sets: ConfigSet[];
  listError: string | null;
  onLoad: (slug: string) => void;
  onDuplicate: (slug: string) => void;
  onDelete: (slug: string) => void;
}) {
  const [armedSlug, setArmedSlug] = useState<string | null>(null);

  return (
    <Panel title={labels.savedSets} className="saved-sets">
      {listError && <Banner message={messages.stateRefreshFailed(listError)} />}
      {sets.length === 0 && !listError && (
        <div className="helper-text">{labels.noSavedSets}</div>
      )}
      <ul className="saved-sets-list">
        {sets.map((s) => {
          const slug = slugify(s.name);
          const armed = armedSlug === slug;
          return (
            <li key={slug} className="saved-set-row">
              <div className="saved-set-name" title={s.notes || s.name}>
                {s.name}
              </div>
              <div className="saved-set-actions">
                <button type="button" onClick={() => onLoad(slug)}>
                  {labels.loadSet}
                </button>
                <button type="button" onClick={() => onDuplicate(slug)}>
                  {labels.duplicateSet}
                </button>
                {armed ? (
                  <>
                    <button
                      type="button"
                      className="primary"
                      onClick={() => {
                        setArmedSlug(null);
                        onDelete(slug);
                      }}
                    >
                      {labels.reallyDelete}
                    </button>
                    <button type="button" onClick={() => setArmedSlug(null)}>
                      {labels.cancel}
                    </button>
                  </>
                ) : (
                  <button type="button" onClick={() => setArmedSlug(slug)}>
                    {labels.deleteSet}
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}
