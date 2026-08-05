export interface Filter {
  id: string;
  label: string;
  active: boolean;
}

export default function Toolbar({
  search,
  onSearch,
  placeholder,
  filters,
  onToggleFilter,
}: {
  search: string;
  onSearch: (value: string) => void;
  placeholder: string;
  filters: Filter[];
  onToggleFilter: (id: string) => void;
}) {
  return (
    <div className="ui-toolbar">
      <input
        className="ui-toolbar-search"
        value={search}
        placeholder={placeholder}
        aria-label={placeholder}
        onChange={(e) => onSearch(e.target.value)}
      />
      {filters.map((f) => (
        <button
          key={f.id}
          type="button"
          className={`ui-filter-chip ${f.active ? "ui-filter-active" : ""}`}
          onClick={() => onToggleFilter(f.id)}
        >
          {f.label}
        </button>
      ))}
    </div>
  );
}
