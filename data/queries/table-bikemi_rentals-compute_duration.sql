ALTER TABLE bikemi_source_data
    ADD COLUMN IF NOT EXISTS durata_noleggio interval GENERATED ALWAYS AS (DATE_TRUNC('second', (data_restituzione - data_prelievo))) STORED;
