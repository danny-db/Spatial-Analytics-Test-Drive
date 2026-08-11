-- Gold layer — Victoria-filtered, business-ready transmission assets.
--
-- The transmission network is filtered to Victoria using the VIC state
-- multipolygon. H3 is used in preference to st_intersects()/st_intersection()
-- alone because H3 containment joins parallelise, whereas ST functions run on a
-- single worker. Migrated from the former 1_dataset/2_Silver_Gold_Layers notebook.

-- VIC Transmission Substations.
-- Point-in-polygon via H3: index both substations (points) and the VIC boundary
-- (multipolygon) to resolution 10 and equi-join on the H3 cell.
CREATE OR REFRESH MATERIALIZED VIEW gold_vic_transmission_substation AS
WITH tn_sub_h3 AS (
    SELECT
        name AS substation_name,
        featuretyp AS feature_type,
        class AS tn_class,
        revised AS revised_ts,
        voltagekv,
        locality,
        x_coordina AS x_coord,
        y_coordina AS y_coord,
        geom_4326,
        h3_pointash3(geom_4326, 10) AS tn_substation_cell
    FROM silver_transmission_substation
    WHERE state = 'Victoria'
),
state_h3 AS (
    SELECT
        state,
        explode(h3_polyfillash3(geom_4326, 10)) AS state_cell
    FROM geo.digital_boundary.silver_au_states_and_territories
    WHERE state = 'Victoria'
)
SELECT
    s.state,
    substation_name,
    feature_type,
    tn_class,
    revised_ts,
    voltagekv,
    locality,
    x_coord,
    y_coord,
    geom_4326
FROM tn_sub_h3 t
JOIN state_h3 s
    ON t.tn_substation_cell = s.state_cell;

-- VIC Transmission Lines: containment + intersection.
-- Tessellate each line to per-cell chips, keep chips whose H3 cell is covered by
-- the VIC boundary, then clip each line to the polygon via st_intersection and
-- reassemble with st_union_agg.
CREATE OR REFRESH MATERIALIZED VIEW gold_vic_transmission_line AS
WITH silver_vic_cover_r10 AS (
    -- H3 cover cells for the VIC polygon.
    SELECT
        state,
        explode(h3_coverash3(geom_4326, 10)) AS cell
    FROM geo.digital_boundary.silver_au_states_and_territories
    WHERE state = 'Victoria'
),
silver_vic_line_chips_r10 AS (
    -- Tessellate transmission lines to per-cell chips (chip = WKB of line ∩ cell).
    SELECT
        name AS line_name,
        cellid AS cell,
        core,
        chip
    FROM (
        SELECT
            name,
            -- h3_tessellateaswkb returns an array of structs; INLINE expands it to
            -- one row per cell, exposing cellid / core / chip columns.
            INLINE(h3_tessellateaswkb(geom_4326, 10))
        FROM silver_transmission_line
        WHERE state = 'Victoria'
    )
),
vic_cover AS (
    SELECT DISTINCT
        p.line_name,
        p.chip,
        r.state
    FROM silver_vic_line_chips_r10 p
    JOIN silver_vic_cover_r10 r
        ON p.cell = r.cell
),
vic_cover_count AS (
    SELECT
        c.state,
        c.line_name,
        COUNT(c.chip) AS number_segments
    FROM vic_cover c
    GROUP BY c.state, c.line_name
    -- TODO: Added this to reduce computation scope since ST functions are very computationally expensive
    HAVING number_segments <= 10
)
SELECT
    c_count.state,
    c_count.line_name,
    -- Store the reconstructed linestring as WKT.
    st_astext(
        st_union_agg(
            st_intersection(
                st_geomfromwkb(c.chip),
                st_geomfromtext(s.geom_4326)
            )
        )
    ) AS geom_4326
FROM vic_cover c
INNER JOIN vic_cover_count c_count ON c.line_name = c_count.line_name
LEFT JOIN geo.digital_boundary.silver_au_states_and_territories s ON s.state = c_count.state
WHERE st_intersects(
    st_geomfromwkb(c.chip),      -- transmission line fragment in this cell
    st_geomfromtext(s.geom_4326) -- VIC state geometry
)
GROUP BY c_count.state, c_count.line_name;
