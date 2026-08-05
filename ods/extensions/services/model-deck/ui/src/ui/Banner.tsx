import type { Message } from "../model/messages";

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
    <div className={`ui-banner ui-banner-${message.tone}`} role="status">
      <span className="ui-banner-title">{message.title}</span>
      {message.body && <span className="ui-banner-body">{message.body}</span>}
      {(message.action || onDismiss) && <span className="ui-banner-spacer" />}
      {message.action && onAction && (
        <button onClick={onAction}>{message.action.label}</button>
      )}
      {onDismiss && (
        <button onClick={onDismiss} aria-label="dismiss">
          ×
        </button>
      )}
    </div>
  );
}
