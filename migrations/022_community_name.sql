-- Migration 022: Add community_name column to storylines
-- LLM-generated 2-4 word macro-theme label, populated by compute_communities.py

ALTER TABLE storylines
    ADD COLUMN IF NOT EXISTS community_name TEXT DEFAULT NULL;

COMMENT ON COLUMN storylines.community_name IS
    'LLM-generated 2-4 word macro-theme label for this community, refreshed daily by compute_communities.py';
