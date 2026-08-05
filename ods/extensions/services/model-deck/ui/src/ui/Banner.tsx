import { labels, type Message } from "../model/messages";

/** An inline notice. Tone comes from the message, never from the call site,
 * so one condition cannot be styled two ways in two screens. */
export default function Banner({
  message,
  onAction,
  onDismiss,
}: {
  message: Message;
  onAction?: () => void;
  onDismiss?: () => void;
}) {
  return (
    // Tone drives the live-region politeness for the same reason it drives
    // colour: a failure should interrupt, an informational notice should not.
    <div
      className={`ui-banner ui-banner-${message.tone}`}
      role={message.tone === "danger" ? "alert" : "status"}
    >
      <span className="ui-banner-title">{message.title}</span>
      {message.body && <span className="ui-banner-body">{message.body}</span>}
      {(message.action || onDismiss) && <span className="ui-banner-spacer" />}
      {message.action && onAction && (
        <button type="button" onClick={onAction}>
          {message.action.label}
        </button>
      )}
      {onDismiss && (
        <button type="button" onClick={onDismiss} aria-label={labels.dismiss}>
          ×
        </button>
      )}
    </div>
  );
}
