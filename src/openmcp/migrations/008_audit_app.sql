-- The application on the other end of each call (MCP clientInfo.name, e.g. what VS Code
-- or Claude Desktop reports), next to the client name the server was started with. They
-- can disagree: VS Code can start servers configured for Claude Desktop, which then run
-- with Claude Desktop's name and scopes.
ALTER TABLE audit_log ADD COLUMN app TEXT;
CREATE INDEX audit_log_client_app ON audit_log (client, id) WHERE app IS NOT NULL;
