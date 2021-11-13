CREATE MATERIALIZED VIEW IF NOT EXISTS clusters_hourly_rentals AS
(
SELECT hrb2019.data_partenza,
       bcs.cluster,
       SUM(hrb2019.noleggi_per_ora) AS noleggi_per_ora,
       bcs.cluster_id_nil,
       bcs.cluster_nil
FROM hourly_rentals_before_2019 hrb2019
         inner join bikemi_clustered_stalls bcs on hrb2019.numero_stazione = bcs.numero_stazione
GROUP BY hrb2019.data_partenza, bcs.cluster, bcs.cluster_id_nil, bcs.cluster_nil
ORDER BY bcs.cluster, hrb2019.data_partenza
    );