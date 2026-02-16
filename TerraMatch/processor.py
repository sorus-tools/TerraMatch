import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from osgeo import gdal, ogr, osr
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsGeometry,
    QgsMapLayerType,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsSpatialIndex,
    QgsWkbTypes,
)

from .run_report import TerraMatchRunReport


@dataclass
class GridSpec:
    crs: QgsCoordinateReferenceSystem
    extent: QgsRectangle
    width: int
    height: int
    x_res: float
    y_res: float
    geotransform: Tuple[float, float, float, float, float, float]


@dataclass
class PredictorCriterion:
    layer_id: str
    layer_name: str
    layer_type: str
    field_name: Optional[str]
    buffer_pct: float
    mode: str  # numeric | categorical
    threshold_mode: str = "within"  # within | or_higher | or_lower | match
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    categories: Optional[Set[Any]] = None
    numeric_match: bool = False


class TerraMatchProcessor:
    NO_DATA_SENTINEL = -3.4028235e38
    MAX_PIXEL_COUNT = 120_000_000

    def run(self, params: Dict[str, Any], report: TerraMatchRunReport) -> str:
        gdal.UseExceptions()

        predictors = self._resolve_predictors(params["predictors"], report)
        if not predictors:
            raise RuntimeError("No valid predictors were resolved.")

        report.section("Predictors")
        for pred in predictors:
            layer = pred["layer"]
            report.info(
                f"{layer.name()} | type={pred['layer_type']} | "
                f"field={pred.get('field_name') or '-'} | buffer={pred['buffer_pct']}% | "
                f"mode={pred.get('match_mode', 'within')}"
            )

        training_points, training_crs = self._collect_training_points(params, report)
        if not training_points:
            raise RuntimeError("No training points found.")
        report.section("Training")
        report.info(
            f"training_mode={params['training_mode']} | points={len(training_points)} | "
            f"crs={training_crs.authid() if training_crs else '(none)'}"
        )

        grid = self._determine_grid(
            predictors,
            report,
        )
        report.section("Grid")
        report.info(
            f"crs={grid.crs.authid() or '(custom)'} | size={grid.width}x{grid.height} | "
            f"pixel={grid.x_res} x {grid.y_res}"
        )
        report.info(
            "extent="
            f"{grid.extent.xMinimum():.6f},{grid.extent.yMinimum():.6f},"
            f"{grid.extent.xMaximum():.6f},{grid.extent.yMaximum():.6f}"
        )

        criteria: List[PredictorCriterion] = []
        report.section("Derived Criteria")
        for predictor in predictors:
            criterion = self._derive_criterion_for_predictor(
                predictor, training_points, training_crs, report
            )
            criteria.append(criterion)
            if criterion.mode == "numeric":
                report.info(
                    f"{criterion.layer_name}: [{criterion.min_val}, {criterion.max_val}] "
                    f"(buffer={criterion.buffer_pct}%, mode={criterion.threshold_mode})"
                )
            else:
                vals = sorted([str(v) for v in (criterion.categories or set())])
                if criterion.numeric_match:
                    report.info(f"{criterion.layer_name}: numeric_match_values={vals}")
                else:
                    report.info(f"{criterion.layer_name}: categories={vals}")

        temp_workspace = tempfile.mkdtemp(prefix="terramatch_work_")
        report.info(f"workspace={temp_workspace}")
        keep_workspace = False
        try:
            result_mask = np.ones((grid.height, grid.width), dtype=bool)

            report.section("Evaluation")
            for predictor, criterion in zip(predictors, criteria):
                layer_mask = self._evaluate_predictor_to_mask(
                    predictor=predictor,
                    criterion=criterion,
                    grid=grid,
                    workspace=temp_workspace,
                    report=report,
                )
                layer_pass = int(np.count_nonzero(layer_mask))
                layer_total = int(layer_mask.size)
                report.info(
                    f"{criterion.layer_name}: passing cells={layer_pass}/{layer_total} "
                    f"({(layer_pass / layer_total) * 100:.2f}%)"
                )
                result_mask &= layer_mask

            output_path = self._resolve_output_path(params.get("output_dir"))
            self._write_output_raster(output_path, result_mask, grid)

            pass_count = int(np.count_nonzero(result_mask))
            total_count = int(result_mask.size)
            report.section("Output")
            report.info(f"raster={output_path}")
            report.info(
                f"suitable_cells={pass_count}/{total_count} ({(pass_count / total_count) * 100:.2f}%)"
            )
            return output_path
        except Exception:
            keep_workspace = True
            raise
        finally:
            if keep_workspace:
                report.warn(f"workspace retained for debugging: {temp_workspace}")
            else:
                try:
                    shutil.rmtree(temp_workspace)
                except Exception:
                    report.warn(f"failed to remove workspace: {temp_workspace}")

    def _resolve_predictors(
        self, raw_predictors: Sequence[Dict[str, Any]], report: TerraMatchRunReport
    ) -> List[Dict[str, Any]]:
        resolved = []
        for raw in raw_predictors:
            layer_id = raw.get("layer_id")
            layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
            if layer is None:
                report.warn(f"Skipping missing layer id: {layer_id}")
                continue

            layer_type = raw.get("layer_type")
            if layer_type == "raster":
                if layer.type() != QgsMapLayerType.RasterLayer:
                    report.warn(f"Skipping non-raster layer in raster slot: {layer.name()}")
                    continue
            elif layer_type == "vector":
                if layer.type() != QgsMapLayerType.VectorLayer:
                    report.warn(f"Skipping non-vector layer in vector slot: {layer.name()}")
                    continue
                if layer.geometryType() != QgsWkbTypes.PolygonGeometry:
                    report.warn(f"Skipping non-polygon vector layer: {layer.name()}")
                    continue
            else:
                report.warn(f"Skipping predictor with unknown type: {layer.name()}")
                continue

            resolved.append(
                {
                    "layer": layer,
                    "layer_id": layer.id(),
                    "layer_type": layer_type,
                    "field_name": raw.get("field_name"),
                    "buffer_pct": float(raw.get("buffer_pct", 10.0)),
                    "match_mode": self._normalize_match_mode(raw.get("match_mode", "within")),
                }
            )
        return resolved

    def _normalize_match_mode(self, raw_mode: Any) -> str:
        mode = str(raw_mode or "within").strip().lower().replace(" ", "_")
        if mode in {"within", "range"}:
            return "within"
        if mode in {"or_higher", "higher"}:
            return "or_higher"
        if mode in {"or_lower", "lower"}:
            return "or_lower"
        if mode in {"match", "matches"}:
            return "match"
        return "within"

    def _collect_training_points(
        self, params: Dict[str, Any], report: TerraMatchRunReport
    ) -> Tuple[List[QgsPointXY], QgsCoordinateReferenceSystem]:
        mode = params.get("training_mode")
        if mode == "clicked":
            clicked_points = params.get("clicked_points", [])
            clicked_crs = params.get("clicked_crs")
            if clicked_crs is None:
                clicked_crs = QgsProject.instance().crs()
            return list(clicked_points), clicked_crs

        point_layer_id = params.get("point_layer_id")
        point_layer = (
            QgsProject.instance().mapLayer(point_layer_id) if point_layer_id else None
        )
        if point_layer is None:
            raise RuntimeError("Point layer was not found for training.")

        points: List[QgsPointXY] = []
        for feat in point_layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if QgsWkbTypes.isMultiType(geom.wkbType()):
                points.extend(list(geom.asMultiPoint()))
            else:
                points.append(geom.asPoint())

        if not points:
            report.warn(f"Point layer '{point_layer.name()}' had no valid point geometries.")
        return points, point_layer.crs()

    def _determine_grid(
        self,
        predictors: Sequence[Dict[str, Any]],
        report: TerraMatchRunReport,
    ) -> GridSpec:
        raster_candidates = []
        for pred in predictors:
            if pred["layer_type"] != "raster":
                continue
            layer = pred["layer"]
            x_res, y_res = self._raster_pixel_size(layer)
            if x_res is None or y_res is None:
                continue
            raster_candidates.append((x_res * y_res, x_res, y_res, layer))

        if raster_candidates:
            raster_candidates.sort(key=lambda tup: tup[0])
            _, x_res, y_res, ref_layer = raster_candidates[0]
            output_crs = ref_layer.crs()
            report.info(f"reference_raster={ref_layer.name()} (highest resolution)")
        else:
            output_crs = predictors[0]["layer"].crs()
            x_res = y_res = None

        max_area = -1.0
        largest_extent = None
        for pred in predictors:
            layer = pred["layer"]
            extent = self._transform_extent(layer.extent(), layer.crs(), output_crs)
            area = max(0.0, extent.width()) * max(0.0, extent.height())
            if area > max_area:
                max_area = area
                largest_extent = extent

        if largest_extent is None or largest_extent.isEmpty():
            raise RuntimeError("Could not compute analysis extent from predictors.")

        if not raster_candidates:
            auto_size = self._estimate_vector_only_pixel_size(
                predictors=predictors,
                output_crs=output_crs,
                largest_extent=largest_extent,
                report=report,
            )
            x_res = y_res = auto_size
            report.info(
                f"no raster predictors; auto-selected vector-only pixel size={auto_size}"
            )

        if x_res is None or y_res is None:
            raise RuntimeError("Grid resolution could not be determined.")

        width = int(math.ceil(largest_extent.width() / x_res))
        height = int(math.ceil(largest_extent.height() / y_res))
        if width <= 0 or height <= 0:
            raise RuntimeError("Computed output raster dimensions were invalid.")
        if width * height > self.MAX_PIXEL_COUNT:
            raise RuntimeError(
                f"Output grid is too large ({width}x{height}). "
                "Increase pixel size or reduce predictor extent."
            )

        xmin = largest_extent.xMinimum()
        ymax = largest_extent.yMaximum()
        xmax = xmin + width * x_res
        ymin = ymax - height * y_res
        aligned_extent = QgsRectangle(xmin, ymin, xmax, ymax)
        geotransform = (xmin, x_res, 0.0, ymax, 0.0, -y_res)
        return GridSpec(
            crs=output_crs,
            extent=aligned_extent,
            width=width,
            height=height,
            x_res=x_res,
            y_res=y_res,
            geotransform=geotransform,
        )

    def _estimate_vector_only_pixel_size(
        self,
        predictors: Sequence[Dict[str, Any]],
        output_crs: QgsCoordinateReferenceSystem,
        largest_extent: QgsRectangle,
        report: TerraMatchRunReport,
    ) -> float:
        area = max(0.0, largest_extent.width()) * max(0.0, largest_extent.height())
        if area <= 0:
            return 30.0

        safety_pixel_budget = max(1, int(self.MAX_PIXEL_COUNT * 0.90))
        min_pixel_for_budget = math.sqrt(area / float(safety_pixel_budget))
        target_pixel = math.sqrt(area / 5_000_000.0)

        sampled_short_sides: List[float] = []
        sample_limit_per_layer = 300
        for pred in predictors:
            if pred["layer_type"] != "vector":
                continue
            layer = pred["layer"]
            transform = None
            if layer.crs() != output_crs:
                transform = QgsCoordinateTransform(
                    layer.crs(), output_crs, QgsProject.instance().transformContext()
                )

            sampled = 0
            for feat in layer.getFeatures():
                if sampled >= sample_limit_per_layer:
                    break
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    continue
                bbox = geom.boundingBox()
                if transform is not None:
                    try:
                        bbox = transform.transformBoundingBox(bbox)
                    except Exception:
                        continue
                short_side = min(abs(float(bbox.width())), abs(float(bbox.height())))
                if not math.isfinite(short_side) or short_side <= 0:
                    continue
                sampled_short_sides.append(short_side)
                sampled += 1

        if sampled_short_sides:
            sampled_short_sides.sort()
            idx = int((len(sampled_short_sides) - 1) * 0.30)
            feature_scale = sampled_short_sides[idx]
            detail_pixel = feature_scale / 4.0
            pixel_size = max(min_pixel_for_budget, detail_pixel)
            report.info(
                "vector auto-scale: "
                f"samples={len(sampled_short_sides)} | p30_short_side={feature_scale}"
            )
        else:
            pixel_size = max(min_pixel_for_budget, target_pixel)
            report.warn(
                "vector auto-scale: unable to sample feature geometry scale; "
                "using extent-based estimate."
            )

        shortest_extent_side = min(
            abs(float(largest_extent.width())),
            abs(float(largest_extent.height())),
        )
        if shortest_extent_side > 0:
            max_coarse_pixel = shortest_extent_side / 64.0
            pixel_size = min(pixel_size, max_coarse_pixel)
            pixel_size = max(pixel_size, min_pixel_for_budget)

        if not math.isfinite(pixel_size) or pixel_size <= 0:
            pixel_size = max(min_pixel_for_budget, target_pixel, 30.0)

        return float(pixel_size)

    def _raster_pixel_size(self, layer) -> Tuple[Optional[float], Optional[float]]:
        source = self._resolve_raster_source(layer)
        ds = gdal.Open(source)
        if ds is None:
            return None, None
        gt = ds.GetGeoTransform(can_return_null=True)
        ds = None
        if not gt:
            return None, None
        x_res = abs(float(gt[1]))
        y_res = abs(float(gt[5]))
        if not math.isfinite(x_res) or not math.isfinite(y_res):
            return None, None
        if x_res <= 0 or y_res <= 0:
            return None, None
        return x_res, y_res

    def _transform_extent(
        self,
        extent: QgsRectangle,
        src_crs: QgsCoordinateReferenceSystem,
        dst_crs: QgsCoordinateReferenceSystem,
    ) -> QgsRectangle:
        if src_crs == dst_crs:
            return QgsRectangle(extent)
        transform = QgsCoordinateTransform(
            src_crs, dst_crs, QgsProject.instance().transformContext()
        )
        return transform.transformBoundingBox(extent)

    def _derive_criterion_for_predictor(
        self,
        predictor: Dict[str, Any],
        training_points: Sequence[QgsPointXY],
        training_crs: QgsCoordinateReferenceSystem,
        report: TerraMatchRunReport,
    ) -> PredictorCriterion:
        layer = predictor["layer"]
        layer_type = predictor["layer_type"]
        field_name = predictor.get("field_name")
        buffer_pct = predictor.get("buffer_pct", 10.0)
        threshold_mode = self._normalize_match_mode(predictor.get("match_mode", "within"))

        if layer_type == "raster":
            if threshold_mode == "match":
                report.warn(
                    f"{layer.name()}: raster predictor cannot use mode 'match'; "
                    "falling back to 'within'."
                )
                threshold_mode = "within"
            values = self._sample_raster_at_points(layer, training_points, training_crs)
            if not values:
                raise RuntimeError(
                    f"No valid training values sampled from raster '{layer.name()}'."
                )
            min_val, max_val = self._buffered_range(values, buffer_pct)
            return PredictorCriterion(
                layer_id=layer.id(),
                layer_name=layer.name(),
                layer_type="raster",
                field_name=None,
                buffer_pct=buffer_pct,
                mode="numeric",
                threshold_mode=threshold_mode,
                min_val=min_val,
                max_val=max_val,
            )

        if not field_name:
            raise RuntimeError(f"Vector layer '{layer.name()}' missing selected field.")

        field = layer.fields().field(field_name)
        if field is None:
            raise RuntimeError(
                f"Field '{field_name}' does not exist on layer '{layer.name()}'."
            )

        raw_values = self._sample_vector_field_at_points(
            layer, field_name, training_points, training_crs, report
        )
        if not raw_values:
            raise RuntimeError(
                f"No training values extracted from vector layer '{layer.name()}' "
                f"for field '{field_name}'."
            )

        if self._is_numeric_field(field):
            numeric_values = [
                float(v)
                for v in raw_values
                if self._is_finite_number(v)
            ]
            if not numeric_values:
                raise RuntimeError(
                    f"Vector field '{field_name}' on '{layer.name()}' had no numeric values "
                    "at training points."
                )
            if threshold_mode == "match":
                return PredictorCriterion(
                    layer_id=layer.id(),
                    layer_name=layer.name(),
                    layer_type="vector",
                    field_name=field_name,
                    buffer_pct=buffer_pct,
                    mode="categorical",
                    threshold_mode="match",
                    categories=set(numeric_values),
                    numeric_match=True,
                )
            min_val, max_val = self._buffered_range(numeric_values, buffer_pct)
            return PredictorCriterion(
                layer_id=layer.id(),
                layer_name=layer.name(),
                layer_type="vector",
                field_name=field_name,
                buffer_pct=buffer_pct,
                mode="numeric",
                threshold_mode=threshold_mode,
                min_val=min_val,
                max_val=max_val,
            )

        categories = set(raw_values)
        if threshold_mode != "match":
            report.warn(
                f"{layer.name()}: non-numeric field '{field_name}' ignores mode "
                f"'{threshold_mode}' and uses 'match' categorical matching."
            )
        return PredictorCriterion(
            layer_id=layer.id(),
            layer_name=layer.name(),
            layer_type="vector",
            field_name=field_name,
            buffer_pct=buffer_pct,
                mode="categorical",
                threshold_mode="match",
                categories=categories,
                numeric_match=False,
            )

    def _sample_raster_at_points(
        self,
        layer,
        points: Sequence[QgsPointXY],
        points_crs: QgsCoordinateReferenceSystem,
    ) -> List[float]:
        if not points:
            return []
        layer_points = self._transform_points(points, points_crs, layer.crs())
        provider = layer.dataProvider()
        no_data = provider.sourceNoDataValue(1)

        sampled = []
        for pt in layer_points:
            sample = provider.sample(pt, 1)
            value, ok = None, False
            if isinstance(sample, tuple):
                value = sample[0]
                ok = bool(sample[1]) if len(sample) > 1 else value is not None
            else:
                value = sample
                ok = value is not None
            if not ok:
                continue
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            if no_data is not None and math.isclose(value, float(no_data), rel_tol=0, abs_tol=1e-12):
                continue
            sampled.append(value)
        return sampled

    def _sample_vector_field_at_points(
        self,
        layer,
        field_name: str,
        points: Sequence[QgsPointXY],
        points_crs: QgsCoordinateReferenceSystem,
        report: TerraMatchRunReport,
    ) -> List[Any]:
        if not points:
            return []
        layer_points = self._transform_points(points, points_crs, layer.crs())

        features: List[QgsFeature] = list(layer.getFeatures())
        if not features:
            return []
        feature_map = {feat.id(): feat for feat in features}
        spatial_index = QgsSpatialIndex(features)

        extracted = []
        misses = 0
        for pt in layer_points:
            pt_geom = QgsGeometry.fromPointXY(pt)
            search_rect = QgsRectangle(pt.x(), pt.y(), pt.x(), pt.y())
            candidate_ids = spatial_index.intersects(search_rect)
            found_val = None

            for fid in candidate_ids:
                feat = feature_map.get(fid)
                if feat is None:
                    continue
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    continue
                if geom.contains(pt_geom) or geom.intersects(pt_geom):
                    found_val = feat[field_name]
                    break

            if found_val is None:
                misses += 1
                continue
            extracted.append(found_val)

        if misses > 0:
            report.warn(
                f"{layer.name()}: {misses} training point(s) were outside polygon features."
            )
        return extracted

    def _evaluate_predictor_to_mask(
        self,
        predictor: Dict[str, Any],
        criterion: PredictorCriterion,
        grid: GridSpec,
        workspace: str,
        report: TerraMatchRunReport,
    ) -> np.ndarray:
        layer = predictor["layer"]
        if predictor["layer_type"] == "raster":
            arr = self._warp_raster_to_grid(layer, grid)
            if criterion.mode != "numeric":
                raise RuntimeError("Raster criterion mode must be numeric.")
            return self._numeric_mask(
                arr,
                criterion.min_val,
                criterion.max_val,
                criterion.threshold_mode,
            )

        vector_path, value_field = self._export_vector_for_rasterization(
            layer=layer,
            criterion=criterion,
            target_crs=grid.crs,
            workspace=workspace,
            report=report,
        )
        arr = self._rasterize_vector_to_grid(vector_path, value_field, criterion, grid)
        if criterion.mode == "numeric":
            return self._numeric_mask(
                arr,
                criterion.min_val,
                criterion.max_val,
                criterion.threshold_mode,
            )
        return arr == 1.0

    def _numeric_mask(
        self,
        arr: np.ndarray,
        min_val: Optional[float],
        max_val: Optional[float],
        threshold_mode: str,
    ):
        valid = np.isfinite(arr)
        mode = self._normalize_match_mode(threshold_mode)
        if mode == "match":
            mode = "within"
        if mode == "or_higher":
            if min_val is not None:
                valid &= arr >= min_val
        elif mode == "or_lower":
            if max_val is not None:
                valid &= arr <= max_val
        else:
            if min_val is not None:
                valid &= arr >= min_val
            if max_val is not None:
                valid &= arr <= max_val
        return valid

    def _warp_raster_to_grid(self, layer, grid: GridSpec) -> np.ndarray:
        source = self._resolve_raster_source(layer)
        src_ds = gdal.Open(source)
        if src_ds is None:
            raise RuntimeError(f"Unable to open raster source: {source}")

        src_band = src_ds.GetRasterBand(1)
        src_no_data = src_band.GetNoDataValue()

        kwargs = {
            "format": "MEM",
            "width": grid.width,
            "height": grid.height,
            "dstSRS": grid.crs.toWkt(),
            "outputBounds": (
                grid.extent.xMinimum(),
                grid.extent.yMinimum(),
                grid.extent.xMaximum(),
                grid.extent.yMaximum(),
            ),
            "resampleAlg": gdal.GRA_NearestNeighbour,
            "dstNodata": self.NO_DATA_SENTINEL,
            "multithread": True,
        }
        if src_no_data is not None:
            kwargs["srcNodata"] = float(src_no_data)

        warped = gdal.Warp("", src_ds, options=gdal.WarpOptions(**kwargs))
        src_ds = None
        if warped is None:
            raise RuntimeError(f"Failed to warp raster layer '{layer.name()}'.")

        band = warped.GetRasterBand(1)
        arr = band.ReadAsArray().astype(float)
        band_no_data = band.GetNoDataValue()
        if band_no_data is not None:
            arr[arr == float(band_no_data)] = np.nan
        warped = None
        return arr

    def _resolve_raster_source(self, layer) -> str:
        provider = layer.dataProvider()
        candidates = [provider.dataSourceUri(), layer.source()]
        for candidate in candidates:
            if not candidate:
                continue
            path = candidate.split("|")[0]
            ds = gdal.Open(path)
            if ds is not None:
                ds = None
                return path
        raise RuntimeError(f"Could not open raster source for layer '{layer.name()}'.")

    def _export_vector_for_rasterization(
        self,
        layer,
        criterion: PredictorCriterion,
        target_crs: QgsCoordinateReferenceSystem,
        workspace: str,
        report: TerraMatchRunReport,
    ) -> Tuple[str, str]:
        out_name = f"{self._safe_name(layer.name())}_{uuid.uuid4().hex[:10]}.gpkg"
        out_path = os.path.join(workspace, out_name)

        driver = ogr.GetDriverByName("GPKG")
        if driver is None:
            raise RuntimeError("GDAL GPKG driver is unavailable.")
        if os.path.exists(out_path):
            driver.DeleteDataSource(out_path)
        ds = driver.CreateDataSource(out_path)
        if ds is None:
            raise RuntimeError("Failed to create temporary GeoPackage for vector rasterization.")

        spatial_ref = self._spatial_ref_from_crs(target_crs)
        ogr_layer = ds.CreateLayer("predictor", srs=spatial_ref, geom_type=ogr.wkbUnknown)
        if ogr_layer is None:
            ds = None
            raise RuntimeError("Failed to create temporary vector layer for rasterization.")

        if criterion.mode == "numeric":
            value_field = "tm_val"
            ogr_field = ogr.FieldDefn(value_field, ogr.OFTReal)
        else:
            value_field = "tm_match"
            ogr_field = ogr.FieldDefn(value_field, ogr.OFTInteger)
        ogr_layer.CreateField(ogr_field)
        layer_defn = ogr_layer.GetLayerDefn()

        transform = None
        if layer.crs() != target_crs:
            transform = QgsCoordinateTransform(
                layer.crs(), target_crs, QgsProject.instance().transformContext()
            )

        written = 0
        skipped = 0
        category_values = criterion.categories or set()
        source_field = criterion.field_name

        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                skipped += 1
                continue

            out_geom = QgsGeometry(geom)
            if transform is not None:
                try:
                    out_geom.transform(transform)
                except Exception:
                    skipped += 1
                    continue

            attr_value = feat[source_field] if source_field else None

            if criterion.mode == "numeric":
                if not self._is_finite_number(attr_value):
                    skipped += 1
                    continue
                raster_val = float(attr_value)
            else:
                if criterion.numeric_match:
                    if not self._is_finite_number(attr_value):
                        skipped += 1
                        continue
                    match_value = float(attr_value)
                else:
                    match_value = attr_value
                raster_val = 1 if match_value in category_values else 0

            ogr_geom = ogr.CreateGeometryFromWkb(bytes(out_geom.asWkb()))
            if ogr_geom is None:
                skipped += 1
                continue

            ogr_feat = ogr.Feature(layer_defn)
            ogr_feat.SetGeometry(ogr_geom)
            ogr_feat.SetField(value_field, raster_val)
            if ogr_layer.CreateFeature(ogr_feat) != 0:
                skipped += 1
                ogr_feat = None
                continue

            written += 1
            ogr_feat = None

        ds = None
        report.info(f"{layer.name()}: vector features exported={written}, skipped={skipped}")
        if written == 0:
            raise RuntimeError(
                f"No features were exported for vector predictor '{layer.name()}'."
            )
        return out_path, value_field

    def _rasterize_vector_to_grid(
        self,
        vector_path: str,
        value_field: str,
        criterion: PredictorCriterion,
        grid: GridSpec,
    ) -> np.ndarray:
        if criterion.mode == "numeric":
            init_val = self.NO_DATA_SENTINEL
            nodata_val = self.NO_DATA_SENTINEL
            out_type = gdal.GDT_Float32
        else:
            init_val = 0
            nodata_val = 0
            out_type = gdal.GDT_Byte

        options = gdal.RasterizeOptions(
            format="MEM",
            outputBounds=(
                grid.extent.xMinimum(),
                grid.extent.yMinimum(),
                grid.extent.xMaximum(),
                grid.extent.yMaximum(),
            ),
            width=grid.width,
            height=grid.height,
            outputSRS=grid.crs.toWkt(),
            attribute=value_field,
            initValues=[init_val],
            noData=nodata_val,
            outputType=out_type,
            allTouched=False,
        )
        ds = gdal.Rasterize("", vector_path, options=options)
        if ds is None:
            raise RuntimeError(f"Failed to rasterize vector source: {vector_path}")
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(float)
        if criterion.mode == "numeric":
            arr[arr == self.NO_DATA_SENTINEL] = np.nan
        ds = None
        return arr

    def _buffered_range(self, values: Sequence[float], buffer_pct: float) -> Tuple[float, float]:
        vmin = float(min(values))
        vmax = float(max(values))
        spread = vmax - vmin
        if spread <= 0:
            ref = abs(vmin) if abs(vmin) > 0 else 1.0
            expand = ref * (buffer_pct / 100.0)
        else:
            expand = spread * (buffer_pct / 100.0)
        return vmin - expand, vmax + expand

    def _is_numeric_field(self, field) -> bool:
        if field is None:
            return False
        field_type = field.type()
        numeric_types = {
            getattr(QVariant, "Int", None),
            getattr(QVariant, "UInt", None),
            getattr(QVariant, "LongLong", None),
            getattr(QVariant, "ULongLong", None),
            getattr(QVariant, "Double", None),
        }
        return field_type in {t for t in numeric_types if t is not None}

    def _is_finite_number(self, value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    def _transform_points(
        self,
        points: Sequence[QgsPointXY],
        src_crs: QgsCoordinateReferenceSystem,
        dst_crs: QgsCoordinateReferenceSystem,
    ) -> List[QgsPointXY]:
        if src_crs == dst_crs:
            return [QgsPointXY(pt) for pt in points]
        transform = QgsCoordinateTransform(
            src_crs, dst_crs, QgsProject.instance().transformContext()
        )
        transformed = []
        for point in points:
            transformed.append(transform.transform(point))
        return transformed

    def _resolve_output_path(self, output_dir: Optional[str]) -> str:
        if output_dir:
            out_dir = output_dir
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = tempfile.gettempdir()
        stamp = uuid.uuid4().hex[:8]
        return os.path.join(out_dir, f"terramatch_suitability_{stamp}.tif")

    def _write_output_raster(self, output_path: str, mask: np.ndarray, grid: GridSpec):
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise RuntimeError("GDAL GeoTIFF driver is unavailable.")
        if os.path.exists(output_path):
            driver.Delete(output_path)

        ds = driver.Create(
            output_path,
            grid.width,
            grid.height,
            1,
            gdal.GDT_Byte,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        if ds is None:
            raise RuntimeError(f"Failed to create output raster: {output_path}")

        ds.SetGeoTransform(grid.geotransform)
        wkt = grid.crs.toWkt()
        if wkt:
            ds.SetProjection(wkt)
        out_arr = np.where(mask, 1, 0).astype(np.uint8)
        band = ds.GetRasterBand(1)
        band.WriteArray(out_arr)
        band.FlushCache()
        ds.FlushCache()
        ds = None

    def _spatial_ref_from_crs(self, crs: QgsCoordinateReferenceSystem):
        sref = osr.SpatialReference()
        wkt = crs.toWkt()
        if wkt:
            try:
                sref.ImportFromWkt(wkt)
                return sref
            except Exception:
                pass

        authid = crs.authid() if crs is not None else ""
        if authid.startswith("EPSG:"):
            try:
                epsg = int(authid.split(":", 1)[1])
                sref.ImportFromEPSG(epsg)
                return sref
            except Exception:
                pass
        return None

    def _safe_name(self, value: str) -> str:
        if not value:
            return "layer"
        chars = []
        for ch in value:
            if ch.isalnum() or ch in ("_", "-"):
                chars.append(ch)
            else:
                chars.append("_")
        return "".join(chars)
