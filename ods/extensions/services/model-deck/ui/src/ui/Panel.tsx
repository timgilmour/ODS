import type { ReactNode } from "react";

/** A titled surface. `draft` switches the border to dashed — the signal that
 * nothing inside is deployed (Set Builder, phase 2). */
export default function Panel({
  title,
  draft = false,
  actions,
  className,
  children,
}: {
  title?: ReactNode;
  draft?: boolean;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  const classes = ["ui-panel", draft ? "ui-panel-draft" : null, className]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={classes}>
      {(title || actions) && (
        <div className="ui-panel-head">
          {title && <h2 className="ui-panel-title">{title}</h2>}
          {actions && <div className="ui-panel-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}
