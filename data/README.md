# Data

`lucas_points.parquet`: 10,583 LUCAS 2022 points observed in the field, in four French study areas,
one row per point.

| Column | Content |
|---|---|
| `point_id`, `region`, `block_id` | LUCAS point identifier, study area, 20 km spatial block |
| `split` | `dev` or `test`: whole spatial blocks assigned by hashing, `test` is only used for the final benchmark |
| `x_l93`, `y_l93` | Coordinates (Lambert-93) |
| `lucas_class` | Land cover observed in the field, harmonised to 7 classes (1 built-up, 2 cropland, 3 herbaceous, 4 trees, 5 shrubs, 6 water, 7 bare soil) |
| `wc`, `dw`, `oso`, `clc` | Class of each map at the point, same legend (2021 products) |
| `err_wc`, ... | 1 if the map disagrees with LUCAS |
| `n_classes` ... `dw_entropy`, `oso_valid_obs` | Indicators available at every pixel |
| `emb_00` ... `emb_63` | AlphaEarth annual satellite embedding (2021) at the point |

## Selection of the points

* **Survey**: LUCAS 2022 microdata (Eurostat). The field campaign ran from 2022 to mid-2023: 98% of the
  points were surveyed in 2022, the rest in early 2023. The median gap with the 2021 maps is about one
  year.
* **Field observations only**: about half of the LUCAS 2022 points in France were photo-interpreted in
  the office rather than visited. Those points are excluded: a photo-interpreted label and a satellite
  map both come from imagery, so their errors would be correlated exactly where the maps are hard to
  check. The remaining points were observed on site (from the point itself or from a distance).
* **Assessable at 10 m**: narrow linear features (LUCAS class A22, mostly rural roads) are excluded,
  because a road narrower than a pixel measures the minimum mapping unit rather than a classification
  error.

## Study areas

The four areas are the study extents of the project, named after the region they mainly cover; they are
not administrative regions. No point belongs to two areas, but Occitanie and Massif Central meet along a
boundary that crosses two départements (Cantal, Haute-Loire) and two 20 km blocks (both in the
development half).

| Area | Points | Main NUTS 3 units (départements) |
|---|---|---|
| Occitanie | 2,254 | FRJ (Occitanie), FRI1-FRI2 (south of Nouvelle-Aquitaine), FRK12-FRK13 edges |
| Brittany | 2,169 | FRH (Brittany), FRG01-FRG03 (Pays de la Loire), FRD1 (Manche, Normandy) |
| Grand Est | 3,978 | FRF (Grand Est), FRC1-FRC2 (Bourgogne-Franche-Comté), FRE21, FR102 edges |
| Massif Central | 2,182 | FRK1 (Auvergne), FRK25-FRK26, FRC13 |

## Sources and licences

- LUCAS 2022 survey, Eurostat (microdata). Reuse authorised with acknowledgement.
- ESA WorldCover 10 m 2021 v200 (CC BY 4.0), Google Dynamic World (CC BY 4.0), Theia OSO 2021 (Etalab Open Licence), Copernicus CLC+ Backbone 2021 (Copernicus data policy).
- AlphaEarth Foundations satellite embeddings, Google Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (CC BY 4.0).
