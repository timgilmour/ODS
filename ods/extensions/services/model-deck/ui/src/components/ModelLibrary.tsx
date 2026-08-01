import { bytesToGB, truncateMiddle, type ModelFile } from "../api";

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
 * only path to an action, so the flow stays usable by keyboard/touch. */
export default function ModelLibrary({ models, onPlace, targetGpu }: ModelLibraryProps) {
  return (
    <div className="panel model-library">
      <h2>Model library</h2>
      {models.length === 0 ? (
        <div>no models found</div>
      ) : (
        <ul className="model-library-list">
          {models.map((m) => (
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
              <button onClick={() => onPlace(m.file)} aria-label={`place ${m.file} on GPU ${targetGpu}`}>
                Place
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
