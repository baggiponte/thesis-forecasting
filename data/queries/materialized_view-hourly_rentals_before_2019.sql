CREATE MATERIALIZED VIEW IF NOT EXISTS bikemi_rentals.hrly_rntls_bfr_2019 AS
(
WITH cross_table AS (
    SELECT d.date_time AS data_partenza,
           s.nome      AS stazione_partenza,
           s.numero_stazione
    FROM (
             SELECT *
             FROM bikemi_rentals.bikemi_stations
             WHERE anno < 2019
         ) s
             CROSS JOIN (
        SELECT generate_series(
                       timestamp '2015-06-01',
                       timestamp '2018-06-01',
                       interval '1 hour'
                   )::timestamp
    ) d(date_time)
    WHERE EXTRACT(
                  'hour'
                  FROM d.date_time
              ) BETWEEN 7 and 24
    ORDER BY nome,
             date_time
)
SELECT c.data_partenza,
       c.stazione_partenza,
       c.numero_stazione,
       COUNT(b.*)::smallint AS noleggi_per_ora
FROM cross_table c
         LEFT JOIN bikemi_rentals.rntls_bfr_2019 b ON b.numero_stazione_prelievo = c.numero_stazione
    AND DATE_TRUNC('hour', b.data_prelievo)::timestamp = c.data_partenza
GROUP BY c.data_partenza,
         c.stazione_partenza,
         c.numero_stazione
ORDER BY stazione_partenza,
         data_partenza
    );