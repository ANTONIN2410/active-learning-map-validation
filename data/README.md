# Data

`lucas_points.parquet`: 10,583 LUCAS in situ points in four French regions (Occitanie,
Brittany, Grand Est, Massif Central), one row per point.

| Column | Content |
|---|---|
| `point_id`, `region`, `block_id` | LUCAS point identifier, region, 20 km spatial block |
| `split` | `dev` or `test`: whole spatial blocks assigned by hashing, `test` is only used for the final benchmark |
| `x_l93`, `y_l93` | Coordinates (Lambert-93) |
| `lucas_class` | Land cover observed in the field, harmonised to 7 classes (1 built-up, 2 cropland, 3 herbaceous, 4 trees, 5 shrubs, 6 water, 7 bare soil) |
| `wc`, `dw`, `oso`, `clc` | Class of each map at the point, same legend (2021 products) |
| `err_wc`, ... | 1 if the map disagrees with LUCAS |
| `n_classes` ... `dw_entropy`, `oso_valid_obs` | Indicators available at every pixel |
| `emb_00` ... `emb_63` | AlphaEarth annual satellite embedding (2021) at the point |

## Sources and licences

- LUCAS survey, Eurostat: harmonised 2006-2018 database (d'Andrimont et al., 2020) and 2022 microdata. Reuse authorised with acknowledgement.
- ESA WorldCover 10 m 2021 v200 (CC BY 4.0), Google Dynamic World (CC BY 4.0), Theia OSO 2021 (Etalab Open Licence), Copernicus CLC+ Backbone 2021 (Copernicus data policy).
- AlphaEarth Foundations satellite embeddings, Google Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (CC BY 4.0).
