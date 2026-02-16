# TerraMatch (QGIS Plugin)

TerraMatch builds binary suitability rasters from known successful training points and mixed predictors (rasters and polygon vectors) for any suitability workflow.

## What It Does

- Uses clicked points or a point layer as training data
- Supports any number of predictor layers
- Supports rasters and polygon vectors
- For vector predictors:
  - Numeric fields: `Within`, `Or Higher`, `Or Lower`, `Match`
  - Non-numeric fields: `Match`
- Applies strict all-layer pass logic
- Treats NoData as unsuitable
- Writes detailed run reports to file

## Processing Toolbox

TerraMatch appears in:

- `SORUS > TerraMatch > TerraMatch`

## Compatibility

- QGIS 3.34+

## License

GPL-2.0-or-later
