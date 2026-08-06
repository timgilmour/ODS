import { useMemo, useState } from "react";
import { bytesToGB, truncateMiddle, type ModelFile } from "../api";
import { labels } from "../model/messages";
import Panel from "../ui/Panel";
import Toolbar from "../ui/Toolbar";

interface ModelLibraryProps {
  models: ModelFile[];
  onPlace: (file: string) => void;
  /** Physical GPU index of the lemonade/comfyui drop target this model
   * lands on — derived by SetBuilder from the world snapshot's placement
   * map, not hardcoded, so this label stays correct on any layout. */
  targetGpu: number;
}

/** Left panel of the Set Builder: every model the registry scan found.
 * Each row is HTML5-draggable (drag onto the lemonade/comfyui column in
 * SetBuilder), and also carries a click "Place" button — drag is never the
 * only path to an action, so the flow stays usable by keyboard/touch.
 *
 * The search box filters the rendered list only. It holds no other state and
 * never touches the draft, so a filtered library and a full one place the
 * same model; a registry with several dozen GGUFs is simply scannable. */
export default function ModelLibrary({ models, onPlace, targetGpu }: ModelLibraryProps) {
  const [search, setSearch] = useState("");
  const visible = useMemo(() => {
    const needle = search.toLowerCase();
    return models.filter((m) => m.file.toLowerCase().includes(needle));
  }, [models, search]);

  return (
    <Panel title={labels.modelLibrary} className="model-library">
      <Toolbar
        search={search}
        onSearch={setSearch}
        placeholder={labels.searchModels}
        filters={[]}
        onToggleFilter={() => {}}
      />
      {visible.length === 0 ? (
        <div className="helper-text">{labels.noModels}</div>
      ) : (
        <ul className="model-library-list">
          {visible.map((m) => (
            <li
              key={m.file}
              className="model-library-row"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData("text/plain", m.file);
                e.dataTransfer.effectAllowed = "copy";
              }}
            >
              <span className="model-library-name" title={m.file}>
                {truncateMiddle(m.file)}
              </span>
              <span className="model-library-footprint">{bytesToGB(m.footprint)} GB</span>
              <button
                type="button"
                onClick={() => onPlace(m.file)}
                aria-label={`place ${m.file} on GPU ${targetGpu}`}
              >
                {labels.place}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
