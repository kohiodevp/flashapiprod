# api.py - Version finale avec support QGIS Server + gestion dynamique de projets
# ------------------------------------------------------------------
import os
import json
import zipfile
import shutil
import subprocess
import logging
import fcntl
import uuid
import time
import functools
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from threading import Lock, Thread

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import Transformer, CRS
import fiona
from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from pydantic import BaseModel, Field, validator, ValidationError
from flask_cors import CORS

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

DEFAULT_CRS = os.getenv("DEFAULT_CRS", "EPSG:32630")
DEFAULT_CRS_WGS84 = "EPSG:4326"
BASE_DIR = Path(os.getenv("BASE_DIR", "/data"))
PROJECTS_DIR = BASE_DIR / "projects"
DEFAULT_PROJECT = PROJECTS_DIR / os.getenv("DEFAULT_PROJECT", "default.qgs")

CATEGORIES = ["shapefiles", "csv", "geojson", "projects", "other", "tiles", "parcels", "documents"]
for d in CATEGORIES:
    (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("qgis-api")

# ------------------------------------------------------------------
# Redis
# ------------------------------------------------------------------
redis_client = None
try:
    import redis
    redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST", "localhost")
    if redis_url.startswith("redis://"):
        redis_client = redis.Redis.from_url(
            redis_url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5
        )
    else:
        redis_client = redis.Redis(
            host=redis_url, port=int(os.getenv("REDIS_PORT", 6379)), db=0,
            decode_responses=True, socket_timeout=5, socket_connect_timeout=5
        )
    redis_client.ping()
    log.info("✅ Redis connecté via %s", redis_url)
except Exception as e:
    log.warning("❌ Redis non disponible : %s", e)
    redis_client = None

# ------------------------------------------------------------------
# Gestion QGIS (dynamique)
# ------------------------------------------------------------------
project_sessions = {}
project_sessions_lock = Lock()
qgis_manager = None
SESSION_TIMEOUT = timedelta(minutes=30)

class QgisManager:
    def __init__(self):
        self._initialized = False
        self.qgs_app = None
        self.classes = {}
        self.init_errors = []
        self._lock = Lock()

    def initialize(self):
        with self._lock:
            if self._initialized:
                return True, None
            
            log.info("=== DÉBUT DE L'INITIALISATION DE QGIS ===")
            try:
                self._setup_qgis_environment()

                from qgis.core import (
                    Qgis, QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
                    QgsMapSettings, QgsMapRendererParallelJob, QgsProcessingFeedback,
                    QgsProcessingContext, QgsRectangle, QgsPalLayerSettings, QgsTextFormat,
                    QgsVectorLayerSimpleLabeling, QgsPrintLayout, QgsLayoutItemMap,
                    QgsLayoutItemLegend, QgsLayoutItemLabel, QgsLayoutExporter,
                    QgsCoordinateReferenceSystem, QgsMapLayer, QgsFeature, QgsGeometry,
                    QgsPointXY, QgsFields, QgsField, QgsVectorFileWriter, QgsWkbTypes,
                    QgsLayerTreeModel, QgsFeatureRequest, QgsLayerTreeLayer, QgsLayerTreeGroup,
                    QgsSingleSymbolRenderer, QgsSymbol, QgsFillSymbol, QgsLineSymbol,
                    QgsMarkerSymbol, QgsReadWriteContext, QgsLayoutItemPage, QgsUnitTypes,
                    QgsLayoutItemScaleBar, QgsLayoutItemPicture, QgsLayoutPoint, QgsLayoutSize,
                    QgsLayoutItemAttributeTable, QgsLayoutFrame, QgsMargins, QgsLayoutTable,
                    QgsLayoutItemRegistry, QgsLayerTree, QgsCoordinateTransform,
                    QgsLayoutItemMapGrid, QgsLayoutItemMapOverview, QgsLayoutMeasurement,
                    QgsRenderContext, QgsSimpleFillSymbolLayer, QgsDropShadowEffect,
                    QgsLegendRenderer, QgsLegendSettings, QgsLegendStyle, QSize, QRectF,
                    QgsLayoutItemShape, QgsLayoutItemHtml, QgsLayoutManager, QgsLayoutItem,
                    QgsMasterLayoutInterface, QgsExpression, QgsLayoutTableColumn,
                    QgsLinePatternFillSymbolLayer, QgsSimpleLineSymbolLayer, QgsLayoutItemMarker
                )
                from PyQt5.QtCore import QVariant, Qt, QSize as QtQSize, QBuffer, QByteArray, QIODevice
                from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap, QIcon, QPen, QBrush

                if not QgsApplication.instance():
                    self.qgs_app = QgsApplication([], False)
                    self.qgs_app.initQgis()
                    log.info("Application QGIS initialisée.")
                else:
                    self.qgs_app = QgsApplication.instance()
                    log.info("Instance QGIS existante trouvée.")
                
                try:
                    from qgis import processing
                    log.info("Module de traitement QGIS importé.")
                except ImportError:
                    log.warning("Module de traitement QGIS non disponible.")
                    processing = None

                self.classes = {
                    'Qgis': Qgis, 'QgsApplication': QgsApplication, 'QgsProject': QgsProject,
                    'QgsVectorLayer': QgsVectorLayer, 'QgsRasterLayer': QgsRasterLayer,
                    'QgsMapSettings': QgsMapSettings, 'QgsMapRendererParallelJob': QgsMapRendererParallelJob,
                    'QgsProcessingFeedback': QgsProcessingFeedback, 'QgsProcessingContext': QgsProcessingContext,
                    'QgsRectangle': QgsRectangle, 'QgsPalLayerSettings': QgsPalLayerSettings,
                    'QgsTextFormat': QgsTextFormat, 'QgsVectorLayerSimpleLabeling': QgsVectorLayerSimpleLabeling,
                    'QgsPrintLayout': QgsPrintLayout, 'QgsLayoutItemMap': QgsLayoutItemMap,
                    'QgsLayoutItemLegend': QgsLayoutItemLegend, 'QgsLayoutItemLabel': QgsLayoutItemLabel,
                    'QgsLayoutExporter': QgsLayoutExporter, 'QgsLayoutItemPicture': QgsLayoutItemPicture,
                    'QgsLayoutPoint': QgsLayoutPoint, 'QgsLayoutSize': QgsLayoutSize,
                    'QgsUnitTypes': QgsUnitTypes, 'QgsLayoutItemPage': QgsLayoutItemPage,
                    'QgsSingleSymbolRenderer': QgsSingleSymbolRenderer, 'QgsFillSymbol': QgsFillSymbol,
                    'QgsLineSymbol': QgsLineSymbol, 'QgsMarkerSymbol': QgsMarkerSymbol,
                    'QgsCoordinateReferenceSystem': QgsCoordinateReferenceSystem,
                    'QgsMapLayer': QgsMapLayer, 'QgsFeature': QgsFeature, 'QgsGeometry': QgsGeometry,
                    'QgsPointXY': QgsPointXY, 'QgsFields': QgsFields, 'QgsField': QgsField,
                    'QgsVectorFileWriter': QgsVectorFileWriter, 'QgsWkbTypes': QgsWkbTypes,
                    'QgsLayerTreeModel': QgsLayerTreeModel, 'QgsFeatureRequest': QgsFeatureRequest,
                    'QgsLayerTreeLayer': QgsLayerTreeLayer, 'QgsLayerTreeGroup': QgsLayerTreeGroup,
                    'QgsReadWriteContext': QgsReadWriteContext, 'QSize': QSize, 'QRectF': QRectF,
                    'QByteArray': QByteArray, 'QPainter': QPainter, 'QImage': QImage,
                    'QgsLayoutItemAttributeTable': QgsLayoutItemAttributeTable,
                    'QgsLayoutFrame': QgsLayoutFrame, 'QgsMargins': QgsMargins,
                    'QgsLayoutItemRegistry': QgsLayoutItemRegistry, 'QgsLayerTree': QgsLayerTree,
                    'QgsCoordinateTransform': QgsCoordinateTransform,
                    'QgsLayoutItemMapGrid': QgsLayoutItemMapGrid,
                    'QgsLayoutItemMapOverview': QgsLayoutItemMapOverview,
                    'QgsLayoutMeasurement': QgsLayoutMeasurement,
                    'QgsRenderContext': QgsRenderContext,
                    'QgsSimpleFillSymbolLayer': QgsSimpleFillSymbolLayer,
                    'QgsDropShadowEffect': QgsDropShadowEffect,
                    'QgsLegendRenderer': QgsLegendRenderer,
                    'QgsLegendSettings': QgsLegendSettings,
                    'QgsLegendStyle': QgsLegendStyle,
                    'QgsLayoutItemShape': QgsLayoutItemShape,
                    'QgsLayoutItemHtml': QgsLayoutItemHtml,
                    'QgsLayoutManager': QgsLayoutManager,
                    'QgsLayoutItem': QgsLayoutItem,
                    'QgsMasterLayoutInterface': QgsMasterLayoutInterface,
                    'QgsExpression': QgsExpression,
                    'QgsLayoutTableColumn': QgsLayoutTableColumn,
                    'QgsLinePatternFillSymbolLayer': QgsLinePatternFillSymbolLayer,
                    'QgsSimpleLineSymbolLayer': QgsSimpleLineSymbolLayer,
                    'QgsLayoutItemMarker': QgsLayoutItemMarker,
                    'processing': processing, 'QVariant': QVariant, 'QColor': QColor,
                    'QFont': QFont, 'Qt': Qt, 'QIcon': QIcon, 'QPen': QPen, 'QBrush': QBrush,
                    'QBuffer': QBuffer, 'QIODevice': QIODevice, 'QPixmap': QPixmap,
                    'QtQSize': QtQSize
                }
                
                self._initialized = True
                log.info("=== QGIS INITIALISÉ AVEC SUCCÈS ===")
                return True, None

            except Exception as e:
                error_msg = f"Erreur d'initialisation de QGIS : {e}"
                self.init_errors.append(error_msg)
                log.error(error_msg, exc_info=True)
                return False, error_msg

    def _setup_qgis_environment(self):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['QT_DEBUG_PLUGINS'] = '0'
        os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts/truetype'
        log.info("Environnement QGIS configuré pour le rendu hors écran.")

    def is_initialized(self):
        return self._initialized

    def get_classes(self):
        if not self._initialized:
            raise RuntimeError("QGIS n'est pas initialisé.")
        return self.classes

class ProjectSession:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.project = None
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def get_project(self, qgs_project_class):
        if self.project is None:
            self.project = qgs_project_class()
            self.project.setTitle(f"Projet de session - {self.session_id}")
        self.last_accessed = datetime.now()
        return self.project

    def is_expired(self):
        return datetime.now() - self.last_accessed > SESSION_TIMEOUT

def get_qgis_manager() -> QgisManager:
    global qgis_manager
    if qgis_manager is None:
        qgis_manager = QgisManager()
    return qgis_manager

def get_project_session(session_id: str = None) -> tuple[ProjectSession, str]:
    with project_sessions_lock:
        if session_id and session_id in project_sessions:
            session = project_sessions[session_id]
        else:
            new_session_id = session_id or str(uuid.uuid4())
            session = ProjectSession(new_session_id)
            project_sessions[new_session_id] = session
    return session, session.session_id

def cleanup_expired_sessions():
    while True:
        time.sleep(60)
        with project_sessions_lock:
            expired = [sid for sid, sess in project_sessions.items() if sess.is_expired()]
            for sid in expired:
                del project_sessions[sid]
                log.info(f"🧹 Session expirée nettoyée : {sid}")

# ------------------------------------------------------------------
# Modèles Pydantic
# ------------------------------------------------------------------
class LayerModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    source: str
    geom: str = Field("Polygon", pattern=r"^(Point|LineString|Polygon|MultiPolygon)$")
    lid: Optional[str] = None
    crs: str = Field(DEFAULT_CRS)

    class Config:
        extra = 'forbid'

class ProjectModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    crs_authid: str = Field(DEFAULT_CRS, pattern=r"^EPSG:\d+$")
    crs_proj4: str = Field("+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs")
    crs_wkt: str = ""
    srsid: int = Field(3452, gt=0)
    srid: int = Field(32630, gt=0)
    layers: List[LayerModel] = []
    description: Optional[str] = None

    class Config:
        extra = 'forbid'

class ParcelCreateModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    geometry: Dict[str, Any]
    commune: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    numero: str = Field(..., min_length=1)
    superficie: Optional[float] = Field(None, gt=0)
    proprietaire: Optional[str] = None
    usage: Optional[str] = None
    crs: str = Field(DEFAULT_CRS_WGS84)

    @validator('geometry')
    def validate_geometry(cls, v):
        if not isinstance(v, dict):
            raise ValueError('La géométrie doit être un objet GeoJSON')
        geom_type = v.get('type')
        coords = v.get('coordinates')
        if not geom_type or coords is None:
            raise ValueError('Champs "type" et "coordinates" requis')
        valid_types = ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon']
        if geom_type not in valid_types:
            raise ValueError(f'Type invalide. Acceptés: {", ".join(valid_types)}')

        def validate_point(c):
            if not (isinstance(c, (list, tuple)) and len(c) >= 2):
                raise ValueError(f'Point invalide: {c}')
            if not all(isinstance(i, (int, float)) for i in c[:2]):
                raise ValueError(f'Coordonnées non numériques: {c[:2]}')

        def validate_line_string(c):
            if not isinstance(c, (list, tuple)) or len(c) < 2:
                raise ValueError('LineString trop courte')
            for pt in c:
                validate_point(pt)

        def validate_polygon(c):
            if not isinstance(c, (list, tuple)) or len(c) == 0:
                raise ValueError('Polygon vide')
            for ring in c:
                if not isinstance(ring, (list, tuple)) or len(ring) < 4:
                    raise ValueError('Anneau de Polygon < 4 points')
                for pt in ring:
                    validate_point(pt)

        try:
            if geom_type == "Point":
                validate_point(coords)
            elif geom_type == "LineString":
                validate_line_string(coords)
            elif geom_type == "Polygon":
                validate_polygon(coords)
            elif geom_type == "MultiPoint":
                for pt in coords:
                    validate_point(pt)
            elif geom_type == "MultiLineString":
                for line in coords:
                    validate_line_string(line)
            elif geom_type == "MultiPolygon":
                for poly in coords:
                    validate_polygon(poly)
        except Exception as e:
            raise ValueError(f'Géométrie {geom_type} invalide: {e}')
        return v

    @validator('crs')
    def validate_crs(cls, v):
        if not v.startswith('EPSG:'):
            raise ValueError('CRS doit être au format EPSG:xxxx')
        try:
            CRS.from_epsg(int(v.split(':')[1]))
        except:
            raise ValueError(f'CRS invalide: {v}')
        return v

    class Config:
        extra = 'forbid'

class ParcelAnalysisModel(BaseModel):
    parcel_id: str
    analysis_type: str = Field(..., pattern="^(superficie|perimetre|distance|buffer|centroid)$")
    parameters: Optional[Dict[str, Any]] = None
    output_crs: str = Field(DEFAULT_CRS)

    @validator('parameters')
    def validate_parameters(cls, v, values):
        analysis_type = values.get('analysis_type')
        if analysis_type == 'buffer' and v:
            if 'distance' not in v:
                raise ValueError('Paramètre "distance" requis pour buffer')
            if not isinstance(v['distance'], (int, float)) or v['distance'] <= 0:
                raise ValueError('Distance doit être un nombre positif')
        return v

    class Config:
        extra = 'forbid'

class CoordinateTransformModel(BaseModel):
    coordinates: List[List[float]]
    from_crs: str = Field(DEFAULT_CRS_WGS84)
    to_crs: str = Field(DEFAULT_CRS)

    @validator('coordinates')
    def validate_coordinates(cls, v):
        if not v:
            raise ValueError('Liste de coordonnées vide')
        for coord in v:
            if len(coord) < 2:
                raise ValueError('Chaque coordonnée doit avoir ≥2 valeurs')
        return v

    @validator('from_crs', 'to_crs')
    def validate_crs(cls, v):
        if not v.startswith('EPSG:'):
            raise ValueError('CRS doit être au format EPSG:xxxx')
        try:
            CRS.from_epsg(int(v.split(':')[1]))
        except:
            raise ValueError(f'CRS invalide: {v}')
        return v

    class Config:
        extra = 'forbid'

# ------------------------------------------------------------------
# Services
# ------------------------------------------------------------------
class ParcelService:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir) / "parcels"
        self.base_dir.mkdir(exist_ok=True)
        self.default_crs = DEFAULT_CRS
        self.metadata_file = self.base_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Erreur sauvegarde métadonnées: {e}")

    def _aggregate_parcels(self):
        try:
            parcel_files = list(self.base_dir.glob("*.geojson"))
            if not parcel_files:
                empty_gdf = gpd.GeoDataFrame(
                    columns=["id", "name", "commune", "section", "numero", "superficie_m2", "geometry"],
                    crs=self.default_crs
                )
                empty_gdf.to_file(self.base_dir / "all_parcels.geojson", driver="GeoJSON")
                log.info("✅ Fichier all_parcels.geojson vide créé")
                return

            gdfs = []
            for f in parcel_files:
                try:
                    gdf = gpd.read_file(f)
                    if not gdf.empty:
                        gdfs.append(gdf)
                except Exception as e:
                    log.warning(f"Impossible de lire {f} pour agrégation: {e}")
                    continue

            if not gdfs:
                empty_gdf = gpd.GeoDataFrame(
                    columns=["id", "name", "commune", "section", "numero", "superficie_m2", "geometry"],
                    crs=self.default_crs
                )
                empty_gdf.to_file(self.base_dir / "all_parcels.geojson", driver="GeoJSON")
            else:
                merged = pd.concat(gdfs, ignore_index=True)
                merged_gdf = gpd.GeoDataFrame(merged, crs=self.default_crs)
                required_cols = ["id", "name", "commune", "section", "numero", "superficie_m2"]
                for col in required_cols:
                    if col not in merged_gdf.columns:
                        merged_gdf[col] = ""
                merged_gdf.to_file(self.base_dir / "all_parcels.geojson", driver="GeoJSON")

            log.info(f"✅ Agrégation réussie : {len(gdfs)} parcelles → all_parcels.geojson")
        except Exception as e:
            log.error(f"❌ Erreur agrégation: {e}", exc_info=True)
            raise

    def create_parcel(self, parcel_data: ParcelCreateModel) -> Dict[str, Any]:
        parcel_id = str(uuid.uuid4())
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        try:
            if parcel_data.crs != self.default_crs:
                log.info(f"Conversion de {parcel_data.crs} vers {self.default_crs}")
                geometry = self._transform_geometry(
                    parcel_data.geometry,
                    parcel_data.crs,
                    self.default_crs
                )
            else:
                geometry = shape(parcel_data.geometry)

            if not geometry.is_valid:
                geometry = geometry.buffer(0)
                if not geometry.is_valid:
                    raise ValueError("Géométrie invalide après correction")

            superficie_m2 = self._calculate_area_m2(geometry)
            perimetre_m = round(geometry.length, 2)
            centroid = geometry.centroid

            gdf = gpd.GeoDataFrame([{
                'id': parcel_id,
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'superficie_m2': superficie_m2,
                'superficie_ha': round(superficie_m2 / 10000, 4),
                'superficie_ares': round(superficie_m2 / 100, 2),
                'perimetre_m': perimetre_m,
                'proprietaire': parcel_data.proprietaire or '',
                'usage': parcel_data.usage or '',
                'crs': self.default_crs,
                'centroid_x': round(centroid.x, 2),
                'centroid_y': round(centroid.y, 2),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'geometry': geometry
            }], crs=self.default_crs)

            gdf.to_file(parcel_file, driver='GeoJSON')

            self.metadata[parcel_id] = {
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'file': str(parcel_file.name)
            }
            self._save_metadata()
            self._aggregate_parcels()

            log.info(f"✅ Parcelle {parcel_id} créée")
            return {
                'id': parcel_id,
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'superficie_m2': superficie_m2,
                'superficie_ha': round(superficie_m2 / 10000, 4),
                'superficie_ares': round(superficie_m2 / 100, 2),
                'perimetre_m': perimetre_m,
                'crs': self.default_crs,
                'centroid': {
                    'x': round(centroid.x, 2),
                    'y': round(centroid.y, 2)
                },
                'file': str(parcel_file)
            }

        except Exception as e:
            log.error(f"❌ Erreur création parcelle: {e}")
            if parcel_file.exists():
                parcel_file.unlink()
            raise

    def _transform_geometry(self, geometry: Dict, from_crs: str, to_crs: str):
        try:
            transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
            shapely_geom = shape(geometry)

            def transform_coords(x, y, z=None):
                x_new, y_new = transformer.transform(x, y)
                return (x_new, y_new)

            return transform(transform_coords, shapely_geom)
        except Exception as e:
            raise ValueError(f"Erreur transformation géométrique: {e}")

    def _calculate_area_m2(self, geometry) -> float:
        return round(geometry.area, 2)

    def get_parcel(self, parcel_id: str, output_crs: str = None) -> Optional[Dict]:
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            return None
        try:
            gdf = gpd.read_file(parcel_file)
            if gdf.empty:
                return None
            parcel_data = gdf.iloc[0].to_dict()
            if output_crs and output_crs != self.default_crs:
                gdf = gdf.to_crs(output_crs)
                transformed_geom = gdf.iloc[0].geometry
                parcel_data['geometry'] = mapping(transformed_geom)
                parcel_data['crs'] = output_crs
            else:
                parcel_data['geometry'] = mapping(gdf.geometry.iloc[0])
                parcel_data['crs'] = self.default_crs
            return parcel_data
        except Exception as e:
            log.error(f"Erreur lecture parcelle {parcel_id}: {e}")
            return None

    def list_parcels(self, commune: str = None) -> List[Dict]:
        parcels = []
        for parcel_file in self.base_dir.glob("*.geojson"):
            try:
                gdf = gpd.read_file(parcel_file)
                if gdf.empty:
                    continue
                parcel = gdf.iloc[0].to_dict()
                if commune and parcel.get('commune', '').lower() != commune.lower():
                    continue
                parcel.pop('geometry', None)
                parcels.append(parcel)
            except Exception as e:
                log.error(f"Erreur lecture {parcel_file}: {e}")
                continue
        return sorted(parcels, key=lambda x: x.get('created_at', ''), reverse=True)

    def delete_parcel(self, parcel_id: str) -> bool:
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            return False
        try:
            parcel_file.unlink()
            if parcel_id in self.metadata:
                del self.metadata[parcel_id]
                self._save_metadata()
            self._aggregate_parcels()
            log.info(f"✅ Parcelle {parcel_id} supprimée")
            return True
        except Exception as e:
            log.error(f"Erreur suppression parcelle {parcel_id}: {e}")
            return False

    def analyze_parcel(self, analysis_data: ParcelAnalysisModel) -> Dict[str, Any]:
        parcel = self.get_parcel(analysis_data.parcel_id, analysis_data.output_crs)
        if not parcel:
            raise ValueError(f"Parcelle {analysis_data.parcel_id} non trouvée")
        geometry = shape(parcel['geometry'])
        result = {
            'parcel_id': analysis_data.parcel_id,
            'analysis_type': analysis_data.analysis_type,
            'crs': parcel['crs']
        }
        if analysis_data.analysis_type == "superficie":
            result['data'] = self._analyze_area(geometry, parcel['crs'])
        elif analysis_data.analysis_type == "perimetre":
            result['data'] = self._analyze_perimeter(geometry, parcel['crs'])
        elif analysis_data.analysis_type == "buffer":
            result['data'] = self._analyze_buffer(geometry, analysis_data.parameters, parcel['crs'])
        elif analysis_data.analysis_type == "centroid":
            result['data'] = self._analyze_centroid(geometry, parcel['crs'])
        return result

    def _analyze_area(self, geometry, crs: str) -> Dict[str, Any]:
        area_m2 = geometry.area
        return {
            'superficie_m2': round(area_m2, 2),
            'superficie_ha': round(area_m2 / 10000, 4),
            'superficie_ares': round(area_m2 / 100, 2),
            'precision': 'exacte' if crs == DEFAULT_CRS else 'approximative'
        }

    def _analyze_perimeter(self, geometry, crs: str) -> Dict[str, Any]:
        perimeter = geometry.length
        return {
            'perimetre_m': round(perimeter, 2),
            'perimetre_km': round(perimeter / 1000, 3)
        }

    def _analyze_buffer(self, geometry, params: Dict, crs: str) -> Dict[str, Any]:
        distance = params.get('distance', 10)
        resolution = params.get('resolution', 16)
        buffer_geom = geometry.buffer(distance, resolution=resolution)
        buffer_area = buffer_geom.area
        return {
            'buffer_geometry': mapping(buffer_geom),
            'distance_buffer_m': distance,
            'superficie_buffer_m2': round(buffer_area, 2),
            'superficie_buffer_ha': round(buffer_area / 10000, 4),
            'crs': crs
        }

    def _analyze_centroid(self, geometry, crs: str) -> Dict[str, Any]:
        centroid = geometry.centroid
        return {
            'centroid': {
                'x': round(centroid.x, 6),
                'y': round(centroid.y, 6),
                'geometry': mapping(centroid)
            },
            'crs': crs
        }

class CoordinateService:
    def __init__(self):
        self.common_crs = {
            '32630': {'name': 'EPSG:32630 - UTM zone 30N', 'unit': 'mètres', 'type': 'projected'},
            '4326': {'name': 'EPSG:4326 - WGS84', 'unit': 'degrés', 'type': 'geographic'},
            '3857': {'name': 'EPSG:3857 - Web Mercator', 'unit': 'mètres', 'type': 'projected'},
            '2154': {'name': 'EPSG:2154 - RGF93/Lambert-93', 'unit': 'mètres', 'type': 'projected'}
        }

    def transform_coordinates(self, transform_data: CoordinateTransformModel) -> Dict[str, Any]:
        try:
            transformer = Transformer.from_crs(
                transform_data.from_crs,
                transform_data.to_crs,
                always_xy=True
            )
            transformed_coords = []
            for i, coord in enumerate(transform_data.coordinates):
                try:
                    x, y = coord[0], coord[1]
                    x_trans, y_trans = transformer.transform(x, y)
                    transformed_coords.append([round(x_trans, 6), round(y_trans, 6)])
                except Exception as e:
                    log.warning(f"Erreur transformation coordonnée {i}: {e}")
                    continue
            return {
                'coordinates': transformed_coords,
                'from_crs': transform_data.from_crs,
                'to_crs': transform_data.to_crs,
                'count': len(transformed_coords),
                'count_input': len(transform_data.coordinates)
            }
        except Exception as e:
            raise ValueError(f"Erreur transformation: {str(e)}")

    def get_crs_info(self, epsg_code: str) -> Dict[str, Any]:
        try:
            if epsg_code.startswith('EPSG:'):
                epsg_code = epsg_code.split(':')[1]
            crs = CRS.from_epsg(int(epsg_code))
            return {
                'epsg': f"EPSG:{epsg_code}",
                'name': crs.name,
                'type': 'projected' if crs.is_projected else 'geographic',
                'units': str(crs.axis_info[0].unit_name) if crs.axis_info else 'unknown',
                'area_of_use': crs.area_of_use.name if crs.area_of_use else 'Unknown',
                'proj4': crs.to_proj4(),
                'wkt': crs.to_wkt()
            }
        except Exception as e:
            raise ValueError(f"CRS non trouvé: EPSG:{epsg_code} - {str(e)}")

    def list_common_crs(self) -> Dict[str, Any]:
        return {
            'default': DEFAULT_CRS,
            'common_crs': self.common_crs,
            'description': 'Systèmes de coordonnées courants'
        }

# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
parcel_service = ParcelService(BASE_DIR)
coord_service = CoordinateService()

# ------------------------------------------------------------------
# Décorateurs
# ------------------------------------------------------------------
def handle_errors(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            log.error(f"Erreur validation: {e}")
            return jsonify({"error": "Erreur de validation", "details": e.errors()}), 400
        except ValueError as e:
            log.error(f"Erreur valeur: {e}")
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            log.error(f"Fichier non trouvé: {e}")
            return jsonify({"error": "Ressource non trouvée"}), 404
        except Exception as e:
            log.error(f"Erreur inattendue: {e}", exc_info=True)
            return jsonify({"error": "Erreur serveur", "message": str(e)}), 500
    return wrapper

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    redis_status = False
    if redis_client:
        try:
            redis_status = redis_client.ping()
        except:
            pass
    qgis_status = get_qgis_manager().is_initialized()
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.2.0",
        "crs_default": DEFAULT_CRS,
        "services": {
            "default_project": DEFAULT_PROJECT.exists(),
            "qgis_dynamic": qgis_status,
            "redis": redis_status,
            "base_dir": str(BASE_DIR),
            "parcels_count": len(list((BASE_DIR / 'parcels').glob('*.geojson')))
        }
    })

@app.route('/api/crs/info', methods=['GET'])
@handle_errors
def get_crs_info():
    epsg = request.args.get('epsg', '32630')
    info = coord_service.get_crs_info(epsg)
    return jsonify(info)

@app.route('/api/crs/list', methods=['GET'])
def list_crs():
    return jsonify(coord_service.list_common_crs())

@app.route('/api/crs/transform', methods=['POST'])
@handle_errors
def transform_coordinates():
    data = CoordinateTransformModel.parse_raw(request.data)
    result = coord_service.transform_coordinates(data)
    return jsonify(result)

@app.route('/api/parcels', methods=['POST'])
@handle_errors
def create_parcel():
    data = ParcelCreateModel.parse_raw(request.data)
    result = parcel_service.create_parcel(data)
    return jsonify(result), 201

@app.route('/api/parcels', methods=['GET'])
@handle_errors
def list_parcels():
    commune = request.args.get('commune')
    parcels = parcel_service.list_parcels(commune)
    return jsonify({'count': len(parcels), 'parcels': parcels})

@app.route('/api/parcels/<parcel_id>', methods=['GET'])
@handle_errors
def get_parcel(parcel_id):
    output_crs = request.args.get('crs', DEFAULT_CRS)
    parcel = parcel_service.get_parcel(parcel_id, output_crs)
    if not parcel:
        return jsonify({"error": "Parcelle non trouvée"}), 404
    return jsonify(parcel)

@app.route('/api/parcels/<parcel_id>', methods=['DELETE'])
@handle_errors
def delete_parcel(parcel_id):
    success = parcel_service.delete_parcel(parcel_id)
    if not success:
        return jsonify({"error": "Parcelle non trouvée"}), 404
    return jsonify({"message": "Parcelle supprimée", "id": parcel_id}), 200

@app.route('/api/parcels/<parcel_id>/analyze', methods=['POST'])
@handle_errors
def analyze_parcel(parcel_id):
    data = ParcelAnalysisModel.parse_raw(request.data)
    data.parcel_id = parcel_id
    result = parcel_service.analyze_parcel(data)
    return jsonify(result)

# ------------------------------------------------------------------
# OGC - QGIS Server avec support MAP dynamique
# ------------------------------------------------------------------
QGIS_BIN = "/usr/lib/cgi-bin/qgis_mapserv.fcgi"

@app.route('/api/ogc/<service>', methods=['GET'])
@handle_errors
def ogc_service(service):
    if service.upper() not in ['WMS', 'WFS', 'WCS']:
        return jsonify({"error": "Service non supporté"}), 400
    if not os.path.isfile(QGIS_BIN):
        log.warning("QGIS Server non installé → mock OGC")
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "mock": True,
            "hint": "QGIS Server absent"
        }), 200

    qs = request.query_string.decode()
    parsed_qs = request.args
    request_param = parsed_qs.get('REQUEST', '').upper()

    map_param = parsed_qs.get('MAP')
    if map_param:
        requested_path = (PROJECTS_DIR / map_param).resolve()
        if not str(requested_path).startswith(str(PROJECTS_DIR.resolve())):
            return jsonify({"error": "Projet hors dossier autorisé"}), 403
        if not requested_path.exists():
            return jsonify({"error": f"Projet non trouvé: {map_param}"}), 404
        project_file = str(requested_path)
    else:
        if not DEFAULT_PROJECT.exists():
            return jsonify({"error": f"Projet par défaut absent: {DEFAULT_PROJECT}"}), 500
        project_file = str(DEFAULT_PROJECT)

    if request_param == 'GETMAP':
        if 'CRS=' not in qs.upper() and 'SRS=' not in qs.upper():
            separator = '&' if '?' in qs else '?'
            qs += f"{separator}CRS={DEFAULT_CRS}"

    env = os.environ.copy()
    env.update({
        "QUERY_STRING": qs,
        "QGIS_PROJECT_FILE": project_file,
        "SERVICE": service.upper(),
        "QT_QPA_PLATFORM": "offscreen",
        "REQUEST_METHOD": "GET"
    })

    try:
        result = subprocess.run(
            [QGIS_BIN],
            env=env,
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            log.error(f"Erreur QGIS Server: {result.stderr.decode()}")
            return jsonify({"error": "Erreur service OGC"}), 500

        content_type = "text/xml"
        stdout = result.stdout
        if stdout.startswith(b"\x89PNG"):
            content_type = "image/png"
        elif stdout.startswith(b"%PDF"):
            content_type = "application/pdf"
        elif stdout.startswith(b"GIF89a"):
            content_type = "image/gif"
        elif b"<ServiceException" in stdout:
            content_type = "text/xml"

        return Response(stdout, content_type=content_type)
    except subprocess.TimeoutExpired:
        log.error("Timeout QGIS Server")
        return jsonify({"error": "Timeout du service OGC"}), 504
    except Exception as e:
        log.error(f"Erreur OGC service: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# Rapports PDF dynamiques
# ------------------------------------------------------------------
@app.route('/api/reports/parcel/<parcel_id>', methods=['GET'])
@handle_errors
def generate_parcel_report(parcel_id):
    qgis_mgr = get_qgis_manager()
    if not qgis_mgr.is_initialized():
        return jsonify({"error": "QGIS non initialisé"}), 500
    
    classes = qgis_mgr.get_classes()
    QgsProject = classes['QgsProject']
    QgsVectorLayer = classes['QgsVectorLayer']
    QgsPrintLayout = classes['QgsPrintLayout']
    QgsLayoutItemMap = classes['QgsLayoutItemMap']
    QgsLayoutItemLabel = classes['QgsLayoutItemLabel']
    QgsLayoutExporter = classes['QgsLayoutExporter']
    QgsRectangle = classes['QgsRectangle']
    QByteArray = classes['QByteArray']
    QBuffer = classes['QBuffer']
    QIODevice = classes['QIODevice']
    
    parcel = parcel_service.get_parcel(parcel_id, output_crs=DEFAULT_CRS)
    if not parcel:
        return jsonify({"error": "Parcelle non trouvée"}), 404
    
    geometry = shape(parcel['geometry'])
    bounds = geometry.bounds

    project = QgsProject.instance()
    project.clear()
    
    parcels_path = str(BASE_DIR / "parcels" / "all_parcels.geojson")
    vlayer = QgsVectorLayer(parcels_path, "parcels", "ogr")
    if not vlayer.isValid():
        return jsonify({"error": "Couche parcels invalide"}), 500
    project.addMapLayer(vlayer)
    
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("Rapport Parcelle")
    
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(20, 20, 180, 180)
    margin_x = (bounds[2] - bounds[0]) * 0.1
    margin_y = (bounds[3] - bounds[1]) * 0.1
    map_extent = QgsRectangle(
        bounds[0] - margin_x,
        bounds[1] - margin_y,
        bounds[2] + margin_x,
        bounds[3] + margin_y
    )
    map_item.setExtent(map_extent)
    map_item.setLayers([vlayer])
    layout.addLayoutItem(map_item)
    map_item.attemptMove(classes['QgsLayoutPoint'](10, 10, classes['QgsUnitTypes'].LayoutMillimeters))
    map_item.attemptResize(classes['QgsLayoutSize'](190, 190, classes['QgsUnitTypes'].LayoutMillimeters))
    
    title = QgsLayoutItemLabel(layout)
    title.setText(f"Rapport Parcelle - {parcel['name']}")
    title.setFont(classes['QFont']("Arial", 14, classes['QFont'].Bold))
    layout.addLayoutItem(title)
    title.attemptMove(classes['QgsLayoutPoint'](10, 5, classes['QgsUnitTypes'].LayoutMillimeters))
    title.attemptResize(classes['QgsLayoutSize'](190, 10, classes['QgsUnitTypes'].LayoutMillimeters))
    
    info_text = (
        f"Commune : {parcel['commune']}\n"
        f"Section : {parcel['section']}\n"
        f"Numéro : {parcel['numero']}\n"
        f"Superficie : {parcel['superficie_ha']} ha\n"
        f"Centroïde : ({parcel['centroid']['x']}, {parcel['centroid']['y']})"
    )
    info_label = QgsLayoutItemLabel(layout)
    info_label.setText(info_text)
    info_label.setFont(classes['QFont']("Arial", 10))
    layout.addLayoutItem(info_label)
    info_label.attemptMove(classes['QgsLayoutPoint'](10, 205, classes['QgsUnitTypes'].LayoutMillimeters))
    info_label.attemptResize(classes['QgsLayoutSize'](190, 30, classes['QgsUnitTypes'].LayoutMillimeters))
    
    exporter = QgsLayoutExporter(layout)
    pdf_data = QByteArray()
    pdf_buffer = QBuffer(pdf_data)
    pdf_buffer.open(QIODevice.WriteOnly)
    
    result = exporter.exportToPdf(pdf_buffer, classes['QgsLayoutExporter'].PdfExportSettings())
    if result != classes['QgsLayoutExporter'].Success:
        return jsonify({"error": "Échec de l'export PDF"}), 500
    
    pdf_bytes = bytes(pdf_data)
    return Response(pdf_bytes, mimetype='application/pdf', 
                    headers={"Content-Disposition": f"attachment;filename=parcel_{parcel_id}.pdf"})

# ------------------------------------------------------------------
# Admin
# ------------------------------------------------------------------
@app.route('/api/admin/stats', methods=['GET'])
@handle_errors
def admin_stats():
    parcels_dir = BASE_DIR / "parcels"
    stats = {
        "parcels": {
            "total": len(list(parcels_dir.glob("*.geojson"))),
            "storage_mb": sum(f.stat().st_size for f in parcels_dir.glob("*.geojson")) / (1024 * 1024)
        },
        "storage": {"categories": {}},
        "system": {
            "default_project": DEFAULT_PROJECT.exists(),
            "qgis_dynamic": get_qgis_manager().is_initialized(),
            "redis": redis_client.ping() if redis_client else False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    for category in CATEGORIES:
        cat_dir = BASE_DIR / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*"))
            stats["storage"]["categories"][category] = {
                "files": len(files),
                "storage_mb": round(sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024), 2)
            }
    return jsonify(stats)

@app.route('/api/admin/cache/clear', methods=['POST'])
@handle_errors
def clear_cache():
    if not redis_client:
        return jsonify({"error": "Redis non disponible"}), 503
    try:
        redis_client.flushdb()
        return jsonify({"message": "Cache nettoyé", "timestamp": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        log.error(f"Erreur nettoyage cache: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# Fichiers
# ------------------------------------------------------------------
@app.route('/api/files/upload', methods=['POST'])
@handle_errors
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400
    file = request.files['file']
    category = request.form.get('category', 'other')
    if not file.filename:
        return jsonify({"error": "Nom de fichier invalide"}), 400
    if category not in CATEGORIES:
        return jsonify({"error": f"Catégorie invalide. Acceptées: {', '.join(CATEGORIES)}"}), 400

    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    new_filename = f"{file_id}.{ext}"
    filepath = BASE_DIR / category / new_filename

    try:
        file.save(str(filepath))
        file_info = {
            "id": file_id,
            "filename": filename,
            "category": category,
            "size_bytes": filepath.stat().st_size,
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        if ext in ['shp', 'geojson', 'gpkg']:
            try:
                gdf = gpd.read_file(str(filepath))
                file_info['geo_info'] = {
                    'crs': str(gdf.crs) if gdf.crs else 'unknown',
                    'features': len(gdf),
                    'geometry_type': str(gdf.geometry.type.iloc[0]) if not gdf.empty else 'unknown',
                    'bounds': gdf.total_bounds.tolist() if not gdf.empty else None
                }
            except Exception as e:
                log.warning(f"Métadonnées géospatiales non lues: {e}")
        log.info(f"✅ Fichier uploadé: {filename} -> {new_filename}")
        return jsonify(file_info), 201
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        log.error(f"Erreur upload: {e}")
        raise

@app.route('/api/files/<category>', methods=['GET'])
@handle_errors
def list_files(category):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    cat_dir = BASE_DIR / category
    files = []
    for file_path in cat_dir.glob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
            })
    return jsonify({
        "category": category,
        "count": len(files),
        "files": sorted(files, key=lambda x: x['modified_at'], reverse=True)
    })

@app.route('/api/files/<category>/<filename>', methods=['GET'])
@handle_errors
def download_file(category, filename):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    safe_filename = secure_filename(filename)
    file_path = BASE_DIR / category / safe_filename
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "Fichier non trouvé"}), 404
    return send_from_directory(BASE_DIR / category, safe_filename, as_attachment=True)

@app.route('/api/files/<category>/<filename>', methods=['DELETE'])
@handle_errors
def delete_file(category, filename):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    safe_filename = secure_filename(filename)
    file_path = BASE_DIR / category / safe_filename
    if not file_path.exists():
        return jsonify({"error": "Fichier non trouvé"}), 404
    try:
        file_path.unlink()
        log.info(f"✅ Fichier supprimé: {safe_filename}")
        return jsonify({"message": "Fichier supprimé", "filename": safe_filename})
    except Exception as e:
        log.error(f"Erreur suppression fichier: {e}")
        raise

# ------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------
@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({
        "version": "2.2.0",
        "title": "API QGIS Server - EPSG:32630",
        "default_crs": DEFAULT_CRS,
        "endpoints": {
            "reports": {
                "path": "/api/reports/parcel/{id}",
                "method": "GET",
                "description": "Génère un rapport PDF pour une parcelle"
            },
            "ogc": {
                "path": "/api/ogc/{service}",
                "method": "GET",
                "description": "Services OGC (WMS/WFS/WCS)",
                "note": "Utilisez ?MAP=nom.qgs pour choisir un projet dans /data/projects/"
            }
        }
    })

# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé", "documentation": "/api/docs"}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "Fichier trop volumineux",
        "max_size_mb": app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    }), 413

# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------
if __name__ == '__main__':
    is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
    
    # Initialisation de QGIS
    qgis_mgr = get_qgis_manager()
    success, error = qgis_mgr.initialize()
    if not success:
        log.error(f"❌ Échec initialisation QGIS : {error}")
        sys.exit(1)
    
    # Démarrer le nettoyage des sessions
    cleanup_thread = Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    
    log.info("=" * 60)
    log.info("🚀 Démarrage API QGIS Server")
    log.info(f"📁 Données: {BASE_DIR}")
    log.info(f"🗺️  Projet par défaut: {DEFAULT_PROJECT}")
    log.info(f"🔌 Redis: {'✅' if redis_client else '❌'}")
    log.info(f"🧠 QGIS dynamique: {'✅ Initialisé' if qgis_mgr.is_initialized() else '❌'}")
    log.info("=" * 60)
    if not DEFAULT_PROJECT.exists():
        log.warning(f"⚠️  Projet par défaut absent: {DEFAULT_PROJECT}")

    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 10000)),
        debug=False if is_gunicorn else os.getenv('DEBUG', 'False').lower() == 'true',
        threaded=True
    )