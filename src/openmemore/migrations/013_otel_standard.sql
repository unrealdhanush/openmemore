-- OpenTelemetry capture follows the GenAI semantic conventions: every model
-- call carries its messages, so nothing has to be held between events. The tables 012
-- added for one client's own event format go.
DROP TABLE otel_responses;
DROP TABLE otel_prompts;
