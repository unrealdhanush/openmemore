-- Recall exchanges: turns where an assistant only looked up earlier conversations ("what
-- did I ask Copilot?"). Stored like any other turn, but search ranks them below the
-- conversations they point to; otherwise every recall becomes the best match for the
-- next one, crowding out the original.
ALTER TABLE messages ADD COLUMN recall INTEGER NOT NULL DEFAULT 0;
