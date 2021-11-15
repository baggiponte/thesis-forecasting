CREATE MATERIALIZED VIEW IF NOT EXISTS clusters_daily_rentals AS
(
SELECT drb2019.data_partenza,
       bcs.cluster_nil || ' - ' || bcs.cluster AS cluster,
       SUM(drb2019.noleggi_giornalieri)        AS noleggi_giornalieri
FROM daily_rentals_before_2019 drb2019
         JOIN bikemi_clustered_stalls bcs on drb2019.numero_stazione = bcs.numero_stazione
GROUP BY drb2019.data_partenza, bcs.cluster, bcs.cluster_nil
ORDER BY bcs.cluster, drb2019.data_partenza
    );