CREATE MATERIALIZED VIEW IF NOT EXISTS clstrs_dly_rntls_bfr_2019 AS
(
SELECT drb2019.data_partenza,
       bcs.cluster_nil || ' - ' || bcs.cluster AS cluster,
       SUM(drb2019.noleggi_giornalieri)        AS noleggi_giornalieri
FROM dly_rntls_bfr_2019 drb2019
         JOIN bikemi_stalls_clusters bcs on drb2019.numero_stazione = bcs.numero_stazione
GROUP BY drb2019.data_partenza, bcs.cluster, bcs.cluster_nil
ORDER BY bcs.cluster, drb2019.data_partenza
    );