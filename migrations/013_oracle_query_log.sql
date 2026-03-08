-- Migration 013: Oracle 2.0 query logging table
-- Applied after: 012_narrative_graph.sql

CREATE TABLE IF NOT EXISTS oracle_query_log (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    query TEXT NOT NULL,
    intent VARCHAR(32),
    complexity VARCHAR(16),
    tools_used TEXT[],
    execution_time FLOAT,
    success BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_oracle_log_created_at ON oracle_query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_oracle_log_session ON oracle_query_log(session_id);
CREATE INDEX IF NOT EXISTS idx_oracle_log_intent ON oracle_query_log(intent);
