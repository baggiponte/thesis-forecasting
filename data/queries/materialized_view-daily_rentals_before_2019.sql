CREATE MATERIALIZED VIEW IF NOT EXISTS bikemi_rentals.dly_rntls_bfr_2019 AS
(
WITH cross_table AS (
    SELECT d.date AS data_partenza,
           s.nome AS stazione_partenza,
           s.numero_stazione
    FROM (
             SELECT *
             FROM bikemi_rentals.bikemi_stalls
             WHERE anno < 2019
         ) s
             CROSS JOIN (
        SELECT generate_series(
                       timestamp '2015-06-01',
                       timestamp '2018-06-01',
                       interval '1 day'
                   )::date
    ) d(date)
    ORDER BY nome,
             date
)
SELECT c.data_partenza,
       c.stazione_partenza,
       c.numero_stazione,
       COUNT(b.*)::smallint AS noleggi_giornalieri
FROM cross_table c
         LEFT JOIN bikemi_rentals.rntls_bfr_2019 b ON b.numero_stazione_prelievo = c.numero_stazione
    AND DATE_TRUNC('day', b.data_prelievo)::date = c.data_partenza
GROUP BY c.data_partenza,
         c.stazione_partenza,
         c.numero_stazione
ORDER BY stazione_partenza,
         data_partenza
    )