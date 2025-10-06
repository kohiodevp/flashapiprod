# api.py - Version optimisée pour Render avec QGIS Server
# Performances maximales pour applications mobiles géolocalisées
# ================================================================
import os
import json
import zipfile
import gzip
import uuid
import time
import functools
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from threading import Lock, Thread
from io import BytesIO
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape, mapping, box
from shapely.ops import transform
from pyproj import Transformer, CRS
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
from pydantic import BaseModel, Field, validator, ValidationError
from flask_cors import CORS
from flask_compress import Compress
# ================================================================
# Configuration optimisée pour Render
# ================================================================
app = Flask(__name__)
Compress(app)  # Compression automatique des réponses
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Session-ID"],
        "expose_headers": ["X-Session-ID", "X-RateLimit-Remaining"]
    }
})
app.config.update(
    MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,  # Performance
    SEND_FILE_MAX_AGE_DEFAULT=3600,  # Cache 1h pour fichiers statiques
)
# Variables d'environnement
DEFAULT_CRS = os.getenv("DEFAULT_CRS", "EPSG:32630")
DEFAULT_CRS_WGS84 = "EPSG:4326"
BASE_DIR = Path(os.getenv("BASE_DIR", "/opt/render/project/src/data"))
PROJECTS_DIR = BASE_DIR / "projects"
DEFAULT_PROJECT = PROJECTS_DIR / os.getenv("DEFAULT_PROJECT", "default.qgs")
CACHE_DIR = BASE_DIR / "cache"
TILES_DIR = BASE_DIR / "tiles"
# Création des dossiers
CATEGORIES = ["shapefiles", "csv", "geojson", "projects", "other", "tiles", "parcels", "documents"]
for d in CATEGORIES + ["cache"]:
    (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)
# Logging optimisé
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / 'api.log')
    ]
)
log = logging.getLogger("qgis-api")
# ================================================================
# Redis avec fallback sur cache fichier
# ================================================================
redis_client = None
file_cache = {}
cache_lock = Lock()
try:
    import redis
    redis_url = os.getenv("REDIS_URL", os.getenv("REDIS_HOST", ""))
    if redis_url:
        if redis_url.startswith("redis://"):
            redis_client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=3)
        else:
            redis_client = redis.Redis(
                host=redis_url, 
                port=int(os.getenv("REDIS_PORT", 6379)), 
                db=0,
                decode_responses=True, 
                socket_timeout=3
            )
        redis_client.ping()
        log.info("✅ Redis connecté")
except Exception as e:
    log.warning(f"⚠️ Redis indisponible, utilisation cache fichier: {e}")
    redis_client = None
def cache_get(key: str) -> Optional[str]:
    """Récupération cache avec fallback"""
    if redis_client:
        try:
            val = redis_client.get(key)
            # Si la valeur est un objet bytes (ce qui peut arriver), le décode
            if isinstance(val, bytes):
                try:
                    val = val.decode('utf-8')
                except UnicodeDecodeError:
                    val = val.decode('latin1') # Fallback pour les données binaires comme les PDFs
            return val
        except:
            pass
    with cache_lock:
        return file_cache.get(key)

def cache_set(key: str, value: Any, expire: int = 3600):
    """Enregistrement cache avec fallback"""
    # Convertir la valeur en chaîne pour Redis
    str_value = value
    if not isinstance(value, str):
        # Pour les données binaires (comme les PDFs), on les encode en latin1 pour les stocker comme chaîne
        # et on les décode en latin1 pour les récupérer.
        # Pour les dictionnaires/listes, on les sérialise en JSON.
        if isinstance(value, bytes):
            str_value = value.decode('latin1')
        else:
            str_value = json.dumps(value)
    if redis_client:
        try:
            redis_client.setex(key, expire, str_value)
            return
        except:
            pass
    with cache_lock:
        file_cache[key] = str_value

# ================================================================
# Gestionnaire QGIS optimisé
# ================================================================
qgis_manager = None
project_sessions = {}
project_sessions_lock = Lock()
SESSION_TIMEOUT = timedelta(minutes=30)
class QgisManager:
    def __init__(self):
        self._initialized = False
        self.qgs_app = None
        self.classes = {}
        self._lock = Lock()
    def initialize(self):
        with self._lock:
            if self._initialized:
                return True, None
            log.info("🔧 Initialisation QGIS...")
            try:
                self._setup_qgis_environment()
                from qgis.core import (
                    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
                    QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle,
                    QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel,
                    QgsLayoutExporter, QgsCoordinateReferenceSystem,
                    QgsVectorFileWriter, QgsFeature, QgsGeometry, QgsPointXY,
                    QgsLayoutPoint, QgsLayoutSize, QgsUnitTypes,
                    QgsCoordinateTransform, QgsLayoutItemLegend, QgsLayoutItemScaleBar
                )
                from PyQt5.QtCore import QSize, QByteArray, QBuffer, QIODevice
                from PyQt5.QtGui import QColor, QFont, QImage, QPainter
                if not QgsApplication.instance():
                    self.qgs_app = QgsApplication([], False)
                    self.qgs_app.initQgis()
                else:
                    self.qgs_app = QgsApplication.instance()
                self.classes = {
                    'QgsApplication': QgsApplication,
                    'QgsProject': QgsProject,
                    'QgsVectorLayer': QgsVectorLayer,
                    'QgsRasterLayer': QgsRasterLayer,
                    'QgsMapSettings': QgsMapSettings,
                    'QgsMapRendererParallelJob': QgsMapRendererParallelJob,
                    'QgsRectangle': QgsRectangle,
                    'QgsPrintLayout': QgsPrintLayout,
                    'QgsLayoutItemMap': QgsLayoutItemMap,
                    'QgsLayoutItemLabel': QgsLayoutItemLabel,
                    'QgsLayoutExporter': QgsLayoutExporter,
                    'QgsCoordinateReferenceSystem': QgsCoordinateReferenceSystem,
                    'QgsVectorFileWriter': QgsVectorFileWriter,
                    'QgsFeature': QgsFeature,
                    'QgsGeometry': QgsGeometry,
                    'QgsPointXY': QgsPointXY,
                    'QgsLayoutPoint': QgsLayoutPoint,
                    'QgsLayoutSize': QgsLayoutSize,
                    'QgsUnitTypes': QgsUnitTypes,
                    'QgsCoordinateTransform': QgsCoordinateTransform,
                    'QgsLayoutItemLegend': QgsLayoutItemLegend,
                    'QgsLayoutItemScaleBar': QgsLayoutItemScaleBar,
                    'QSize': QSize,
                    'QByteArray': QByteArray,
                    'QBuffer': QBuffer,
                    'QIODevice': QIODevice,
                    'QColor': QColor,
                    'QFont': QFont,
                    'QImage': QImage,
                    'QPainter': QPainter
                }
                self._initialized = True
                log.info("✅ QGIS initialisé")
                return True, None
            except Exception as e:
                error_msg = f"Erreur initialisation QGIS: {e}"
                log.error(error_msg, exc_info=True)
                return False, error_msg
    def _setup_qgis_environment(self):
        os.environ.update({
            'QT_QPA_PLATFORM': 'offscreen',
            'QT_DEBUG_PLUGINS': '0',
            'QT_QPA_FONTDIR': '/usr/share/fonts/truetype',
            'QGIS_DISABLE_MESSAGE_HOOKS': '1',
            'QGIS_NO_OVERRIDE_IMPORT': '1'
        })
    def is_initialized(self):
        return self._initialized
    def get_classes(self):
        if not self._initialized:
            raise RuntimeError("QGIS non initialisé")
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
        self.last_accessed = datetime.now()
        return self.project
    def is_expired(self):
        return datetime.now() - self.last_accessed > SESSION_TIMEOUT
def get_qgis_manager() -> QgisManager:
    global qgis_manager
    if qgis_manager is None:
        qgis_manager = QgisManager()
    return qgis_manager
def get_project_session(session_id: str = None) -> tuple:
    with project_sessions_lock:
        if session_id and session_id in project_sessions:
            session = project_sessions[session_id]
        else:
            new_session_id = session_id or str(uuid.uuid4())
            session = ProjectSession(new_session_id)
            project_sessions[new_session_id] = session
    return session, session.session_id
def cleanup_expired_sessions():
    """Nettoyage périodique des sessions expirées"""
    while True:
        time.sleep(300)  # 5 minutes
        with project_sessions_lock:
            expired = [sid for sid, sess in project_sessions.items() if sess.is_expired()]
            for sid in expired:
                # Fermer le projet si nécessaire (optionnel, dépend de la gestion de QGIS)
                if expired[0].project:
                    expired[0].project.clear() # Nettoyer le projet avant suppression
                del project_sessions[sid]
                log.info(f"🧹 Session expirée: {sid}")
# ================================================================
# Modèles Pydantic
# ================================================================
class ParcelCreateModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    geometry: Dict[str, Any]
    commune: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    numero: str = Field(..., min_length=1)
    superficie: Optional[float] = None
    proprietaire: Optional[str] = None
    usage: Optional[str] = None
    crs: str = Field(DEFAULT_CRS_WGS84)
    @validator('geometry')
    def validate_geometry(cls, v):
        if not isinstance(v, dict) or 'type' not in v or 'coordinates' not in v:
            raise ValueError('Géométrie GeoJSON invalide')
        return v
    @validator('crs')
    def validate_crs(cls, v):
        if not v.startswith('EPSG:'):
            raise ValueError('CRS doit être EPSG:xxxx')
        try:
            CRS.from_epsg(int(v.split(':')[1]))
        except:
            raise ValueError(f'CRS invalide: {v}')
        return v

class ParcelUpdateModel(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    geometry: Optional[Dict[str, Any]] = None
    commune: Optional[str] = Field(None, min_length=1)
    section: Optional[str] = Field(None, min_length=1)
    numero: Optional[str] = Field(None, min_length=1)
    superficie: Optional[float] = None
    proprietaire: Optional[str] = None
    usage: Optional[str] = None
    crs: Optional[str] = Field(None, pattern=r'^EPSG:\d+$')

    @validator('geometry')
    def validate_geometry(cls, v):
        if v is not None: # Ne valider que si la géométrie est fournie
            if not isinstance(v, dict) or 'type' not in v or 'coordinates' not in v:
                raise ValueError('Géométrie GeoJSON invalide')
        return v

class BoundsModel(BaseModel):
    """Modèle pour requête par emprise géographique"""
    minx: float
    miny: float
    maxx: float
    maxy: float
    crs: str = Field(DEFAULT_CRS_WGS84)
    buffer_m: Optional[float] = Field(None, ge=0, le=10000)

class CoordinateTransformModel(BaseModel):
    coordinates: List[List[float]]
    from_crs: str = Field(DEFAULT_CRS_WGS84)
    to_crs: str = Field(DEFAULT_CRS)
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if not v or not all(len(c) >= 2 for c in v):
            raise ValueError('Coordonnées invalides')
        return v

# --- Ajout des nouveaux modèles Pydantic ---
class ProjectModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    project_file_path: str = Field(..., min_length=1) # Chemin relatif ou absolu vers le .qgs/.qgz
    # Ajouter d'autres champs pertinents pour un projet QGIS

class SearchModel(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BulkParcelCreateModel(BaseModel):
    parcels: List[ParcelCreateModel]

class BulkParcelUpdateModel(BaseModel):
    updates: List[Dict[str, Any]] # { "id": "...", "data": { ... } }

class BulkParcelDeleteModel(BaseModel):
    ids: List[str]

class ReportFormatModel(BaseModel):
    format: str = Field("pdf", pattern="^(pdf|png|jpg|jpeg)$") # Exemple de formats supportés

# --- Modèles Analytics/Performance ---
class AnalyticsDataModel(BaseModel):
    event_type: str
    Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PerformanceMetricsModel(BaseModel):
    metric_name: str
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserBehaviorModel(BaseModel):
    user_id: Optional[str] = None
    action: str
    details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ================================================================
# Service Projets (esquisse) ---
class ProjectService:
    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir
        self.projects_file = projects_dir / "projects.json"
        self._ensure_projects_file()

    def _ensure_projects_file(self):
        if not self.projects_file.exists():
            with open(self.projects_file, 'w') as f:
                json.dump([], f)

    def get_projects(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        with open(self.projects_file, 'r') as f:
            projects = json.load(f)
        start = offset or 0
        end = start + (limit or len(projects))
        return projects[start:end]

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        projects = self.get_projects()
        for proj in projects:
            if proj['id'] == project_id:
                return proj
        return None

    def create_project(self, project_data: ProjectModel) -> Dict[str, Any]:
        project_id = str(uuid.uuid4())
        project_dict = project_data.dict()
        project_dict['id'] = project_id
        project_dict['created_at'] = datetime.now(timezone.utc).isoformat()

        projects = self.get_projects()
        projects.append(project_dict)
        with open(self.projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
        log.info(f"✅ Projet créé: {project_id}")
        return project_dict

    def update_project(self, project_id: str, project_ ProjectModel) -> Optional[Dict[str, Any]]:
        projects = self.get_projects()
        for i, proj in enumerate(projects):
            if proj['id'] == project_id:
                update_data = project_data.dict(exclude_unset=True)
                projects[i].update(update_data)
                with open(self.projects_file, 'w') as f:
                    json.dump(projects, f, indent=2)
                log.info(f"✅ Projet mis à jour: {project_id}")
                return projects[i]
        return None

    def delete_project(self, project_id: str) -> bool:
        projects = self.get_projects()
        for i, proj in enumerate(projects):
            if proj['id'] == project_id:
                del projects[i]
                with open(self.projects_file, 'w') as f:
                    json.dump(projects, f, indent=2)
                log.info(f"✅ Projet supprimé: {project_id}")
                return True
        return False

# --- Initialisation du service Projets ---
project_service = ProjectService(PROJECTS_DIR)

# ================================================================
# Service Analytics
# ================================================================
class AnalyticsService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.analytics_file = self.base_dir / "analytics.json"
        self.events_file = self.base_dir / "analytics_events.json"
        self.behavior_file = self.base_dir / "user_behavior.json"
        self._ensure_files()

    def _ensure_files(self):
        for f in [self.analytics_file, self.events_file, self.behavior_file]:
            if not f.exists():
                with open(f, 'w') as file:
                    json.dump([], file)

    def log_event(self, event_ AnalyticsDataModel):
        events = self._read_events()
        events.append(event_data.dict())
        self._write_events(events)
        log.info(f"📊 Événement loggué: {event_data.event_type}")

    def get_summary(self):
        events = self._read_events()
        summary = {}
        for event in events:
            et = event['event_type']
            summary[et] = summary.get(et, 0) + 1
        # Retourner un format conforme à ce que le dashboard attend
        # Le dashboard appelle analyticsService.getAnalyticsSummary().toPromise()
        # et s'attend à ce que `summary?.period?.overview?.sessions` soit disponible
        # On adapte la réponse pour que `data.period.overview.sessions` contienne un nombre
        # arbitraire ou un calcul basé sur les événements.
        session_count = len([e for e in events if e['event_type'] == 'session_start'])
        return {
            "period": {
                "overview": {
                    "sessions": session_count,
                    "events": len(events),
                    "last_activity": events[-1]['timestamp'] if events else None
                }
            }
        }


    def log_user_behavior(self, behavior_data: UserBehaviorModel):
        behavior = self._read_behavior()
        behavior.append(behavior_data.dict())
        self._write_behavior(behavior)
        log.info(f"👤 Comportement utilisateur loggué: {behavior_data.action}")

    def _read_events(self) -> List[Dict[str, Any]]:
        with open(self.events_file, 'r') as f:
            return json.load(f)

    def _write_events(self, data: List[Dict[str, Any]]):
        with open(self.events_file, 'w') as f:
            json.dump(data, f, indent=2, default=str) # default=str pour sérialiser les datetime

    def _read_behavior(self) -> List[Dict[str, Any]]:
        with open(self.behavior_file, 'r') as f:
            return json.load(f)

    def _write_behavior(self, data: List[Dict[str, Any]]):
        with open(self.behavior_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

# ================================================================
# Service Performance Metrics
# ================================================================
class PerformanceService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.metrics_file = self.base_dir / "performance_metrics.json"
        self._ensure_file()

    def _ensure_file(self):
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                json.dump([], f)

    def log_metric(self, metric_ PerformanceMetricsModel):
        metrics = self._read_metrics()
        metrics.append(metric_data.dict())
        self._write_metrics(metrics)
        log.info(f"📈 Métrique logguée: {metric_data.metric_name}")

    def get_metrics(self, limit: Optional[int] = 100):
        metrics = self._read_metrics()
        # Trier par timestamp descendant et limiter
        sorted_metrics = sorted(metrics, key=lambda x: x['timestamp'], reverse=True)
        return sorted_metrics[:limit]

    def _read_metrics(self) -> List[Dict[str, Any]]:
        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def _write_metrics(self, data: List[Dict[str, Any]]):
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2, default=str) # default=str pour sérialiser les datetime

# ================================================================
# Service Parcelles optimisé
# ================================================================
class ParcelService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / "parcels"
        self.base_dir.mkdir(exist_ok=True)
        self.default_crs = DEFAULT_CRS
        self.all_parcels_file = self.base_dir / "all_parcels.geojson"
        self._ensure_aggregate()
    def _ensure_aggregate(self):
        """S'assure que le fichier agrégé existe"""
        if not self.all_parcels_file.exists():
            empty_gdf = gpd.GeoDataFrame(
                columns=["id", "name", "commune", "section", "numero", "superficie_m2", "geometry"],
                crs=self.default_crs
            )
            empty_gdf.to_file(self.all_parcels_file, driver="GeoJSON")
    def create_parcel(self, parcel_ ParcelCreateModel) -> Dict[str, Any]:
        parcel_id = str(uuid.uuid4())
        try:
            # Transformation géométrie
            if parcel_data.crs != self.default_crs:
                geometry = self._transform_geometry(
                    parcel_data.geometry, parcel_data.crs, self.default_crs
                )
            else:
                geometry = shape(parcel_data.geometry)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            # Calculs
            superficie_m2 = round(geometry.area, 2)
            perimetre_m = round(geometry.length, 2)
            centroid = geometry.centroid
            # Enregistrement
            gdf = gpd.GeoDataFrame([{
                'id': parcel_id,
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'superficie_m2': superficie_m2,
                'superficie_ha': round(superficie_m2 / 10000, 4),
                'perimetre_m': perimetre_m,
                'proprietaire': parcel_data.proprietaire or '',
                'usage': parcel_data.usage or '',
                'centroid_x': round(centroid.x, 2),
                'centroid_y': round(centroid.y, 2),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'geometry': geometry
            }], crs=self.default_crs)
            parcel_file = self.base_dir / f"{parcel_id}.geojson"
            gdf.to_file(parcel_file, driver='GeoJSON')
            # Mise à jour agrégat
            self._update_aggregate()
            log.info(f"✅ Parcelle créée: {parcel_id}")
            return {
                'id': parcel_id,
                'name': parcel_data.name,
                'superficie_m2': superficie_m2,
                'superficie_ha': round(superficie_m2 / 10000, 4),
                'perimetre_m': perimetre_m,
                'centroid': {'x': round(centroid.x, 2), 'y': round(centroid.y, 2)},
                'crs': self.default_crs
            }
        except Exception as e:
            log.error(f"❌ Erreur création parcelle: {e}")
            raise

    def update_parcel(self, parcel_id: str, parcel_data: ParcelUpdateModel) -> Optional[Dict[str, Any]]:
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            return None

        try:
            gdf = gpd.read_file(parcel_file)
            if gdf.empty:
                return None

            update_dict = parcel_data.dict(exclude_unset=True)
            # Mettre à jour les propriétés
            for key, value in update_dict.items():
                if key in gdf.columns:
                    gdf.loc[0, key] = value

            # Recalculer si la géométrie change
            if 'geometry' in update_dict:
                geom_shape = shape(update_dict['geometry'])
                if not geom_shape.is_valid:
                    geom_shape = geom_shape.buffer(0)
                gdf.loc[0, 'geometry'] = geom_shape
                gdf.loc[0, 'superficie_m2'] = round(geom_shape.area, 2)
                gdf.loc[0, 'superficie_ha'] = round(geom_shape.area / 10000, 4)
                gdf.loc[0, 'perimetre_m'] = round(geom_shape.length, 2)
                centroid = geom_shape.centroid
                gdf.loc[0, 'centroid_x'] = round(centroid.x, 2)
                gdf.loc[0, 'centroid_y'] = round(centroid.y, 2)

            # Réécrire le fichier
            gdf.to_file(parcel_file, driver='GeoJSON')
            # Mettre à jour l'agrégat
            self._update_aggregate()
            log.info(f"✅ Parcelle mise à jour: {parcel_id}")

            # Retourner les données mises à jour
            updated_data = gdf.iloc[0].to_dict()
            updated_data['geometry'] = mapping(gdf.geometry.iloc[0])
            return updated_data
        except Exception as e:
            log.error(f"❌ Erreur mise à jour parcelle: {e}")
            return None

    def _transform_geometry(self, geometry: Dict, from_crs: str, to_crs: str):
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        shapely_geom = shape(geometry)
        return transform(lambda x, y, z=None: transformer.transform(x, y), shapely_geom)
    def _update_aggregate(self):
        """Mise à jour fichier agrégé optimisée"""
        try:
            parcel_files = list(self.base_dir.glob("[!all_]*.geojson"))
            if not parcel_files:
                self._ensure_aggregate()
                return
            gdfs = [gpd.read_file(f) for f in parcel_files if f.stat().st_size > 0]
            if gdfs:
                merged = pd.concat(gdfs, ignore_index=True)
                merged_gdf = gpd.GeoDataFrame(merged, crs=self.default_crs)
                merged_gdf.to_file(self.all_parcels_file, driver="GeoJSON")
                log.info(f"✅ Agrégat mis à jour: {len(gdfs)} parcelles")
        except Exception as e:
            log.error(f"❌ Erreur mise à jour agrégat: {e}")
    def get_parcels_by_bounds(self, bounds: BoundsModel) -> Dict[str, Any]:
        """Récupération parcelles dans une emprise (optimisé mobile)"""
        try:
            # Transformation emprise si nécessaire
            if bounds.crs != self.default_crs:
                transformer = Transformer.from_crs(bounds.crs, self.default_crs, always_xy=True)
                minx, miny = transformer.transform(bounds.minx, bounds.miny)
                maxx, maxy = transformer.transform(bounds.maxx, bounds.maxy)
            else:
                minx, miny, maxx, maxy = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
            # Buffer optionnel
            if bounds.buffer_m:
                minx -= bounds.buffer_m
                miny -= bounds.buffer_m
                maxx += bounds.buffer_m
                maxy += bounds.buffer_m
            # Lecture optimisée avec bbox
            bbox = (minx, miny, maxx, maxy)
            gdf = gpd.read_file(self.all_parcels_file, bbox=bbox)
            if gdf.empty:
                return {'count': 0, 'data': [], 'bbox': list(bbox)}
            # Conversion GeoJSON optimisée
            features = json.loads(gdf.to_json())['features']
            return {
                'count': len(features),
                'data': features,
                'bbox': list(bbox),
                'crs': self.default_crs
            }
        except Exception as e:
            log.error(f"❌ Erreur récupération par emprise: {e}")
            raise
    def get_parcel(self, parcel_id: str, output_crs: str = None) -> Optional[Dict]:
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            return None
        try:
            gdf = gpd.read_file(parcel_file)
            if gdf.empty:
                return None
            if output_crs and output_crs != self.default_crs:
                gdf = gdf.to_crs(output_crs)
            parcel_data = gdf.iloc[0].to_dict()
            parcel_data['geometry'] = mapping(gdf.geometry.iloc[0])
            parcel_data['crs'] = output_crs or self.default_crs
            return parcel_data
        except Exception as e:
            log.error(f"❌ Erreur lecture parcelle: {e}")
            return None
    def delete_parcel(self, parcel_id: str) -> bool:
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            return False
        try:
            parcel_file.unlink()
            self._update_aggregate()
            log.info(f"✅ Parcelle supprimée: {parcel_id}")
            return True
        except Exception as e:
            log.error(f"❌ Erreur suppression: {e}")
            return False
# ================================================================
# Service de transformation de coordonnées
# ================================================================
class CoordinateService:
    def __init__(self):
        self._transformers_cache = {}
    def transform_coordinates(self, data: CoordinateTransformModel) -> Dict[str, Any]:
        cache_key = f"{data.from_crs}:{data.to_crs}"
        if cache_key not in self._transformers_cache:
            self._transformers_cache[cache_key] = Transformer.from_crs(
                data.from_crs, data.to_crs, always_xy=True
            )
        transformer = self._transformers_cache[cache_key]
        transformed = [
            [round(x, 6), round(y, 6)]
            for x, y in [transformer.transform(c[0], c[1]) for c in data.coordinates]
        ]
        return {
            'coordinates': transformed,
            'from_crs': data.from_crs,
            'to_crs': data.to_crs,
            'count': len(transformed)
        }
    def get_crs_info(self, epsg_code: str) -> Dict[str, Any]:
        epsg_num = epsg_code.replace('EPSG:', '')
        crs = CRS.from_epsg(int(epsg_num))
        return {
            'epsg': f"EPSG:{epsg_num}",
            'name': crs.name,
            'type': 'projected' if crs.is_projected else 'geographic',
            'units': str(crs.axis_info[0].unit_name) if crs.axis_info else 'unknown'
        }
# ================================================================
# Initialisation services
# ================================================================
parcel_service = ParcelService(BASE_DIR)
coord_service = CoordinateService()
analytics_service = AnalyticsService(BASE_DIR)
performance_service = PerformanceService(BASE_DIR)

# ================================================================
# Décorateurs
# ================================================================
def handle_errors(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            return jsonify({"error": "Validation", "details": e.errors()}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError:
            return jsonify({"error": "Ressource introuvable"}), 404
        except Exception as e:
            log.error(f"Erreur: {e}", exc_info=True)
            return jsonify({"error": "Erreur serveur"}), 500
    return wrapper
def require_session(f):
    """Middleware pour gérer les sessions"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        session_id = request.headers.get('X-Session-ID')
        session, new_session_id = get_project_session(session_id)
        response = f(*args, session=session, **kwargs)
        if isinstance(response, tuple):
            data, status = response
            if isinstance(data, Response):
                data.headers['X-Session-ID'] = new_session_id
                return data, status
        elif isinstance(response, Response):
            response.headers['X-Session-ID'] = new_session_id
        return response
    return wrapper
# ================================================================
# Routes API
# ================================================================

# --- Authentification (esquisse) ---
# Ces routes nécessitent une implémentation complète de gestion des utilisateurs et de sessions/tokens.
# Pour l'instant, elles renvoient des erreurs 501 (Not Implemented).
@app.route('/api/auth/login', methods=['POST'])
@handle_errors
def login():
    return jsonify({"error": "Authentification non implémentée"}), 501

@app.route('/api/auth/register', methods=['POST'])
@handle_errors
def register():
    return jsonify({"error": "Enregistrement non implémenté"}), 501

@app.route('/api/auth/refresh', methods=['POST'])
@handle_errors
def refresh():
    return jsonify({"error": "Actualisation de token non implémentée"}), 501

@app.route('/api/auth/logout', methods=['POST'])
@handle_errors
def logout():
    return jsonify({"error": "Déconnexion non implémentée"}), 501

# --- Projets ---
@app.route('/api/projects', methods=['GET'])
@handle_errors
def list_projects():
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', type=int)
    projects = project_service.get_projects(limit=limit, offset=offset)
    return jsonify({"data": projects})

@app.route('/api/projects', methods=['POST'])
@handle_errors
def create_project():
    data = ProjectModel.parse_raw(request.data)
    result = project_service.create_project(data)
    return jsonify(result), 201

@app.route('/api/projects/<project_id>', methods=['GET'])
@handle_errors
def get_project(project_id):
    project = project_service.get_project(project_id)
    if not project:
        return jsonify({"error": "Projet introuvable"}), 404
    return jsonify(project)

@app.route('/api/projects/<project_id>', methods=['PUT'])
@handle_errors
def update_project(project_id):
    data = ProjectModel.parse_raw(request.data)
    project = project_service.update_project(project_id, data)
    if not project:
        return jsonify({"error": "Projet introuvable"}), 404
    return jsonify(project)

@app.route('/api/projects/<project_id>', methods=['DELETE'])
@handle_errors
def delete_project(project_id):
    success = project_service.delete_project(project_id)
    if not success:
        return jsonify({"error": "Projet introuvable"}), 404
    return jsonify({"message": "Projet supprimé", "id": project_id})

# --- Parcelles - Adaptation GET /parcels ---
@app.route('/api/parcels', methods=['GET'])
@handle_errors
def list_parcels():
    limit = request.args.get('limit', default=100, type=int)
    offset = request.args.get('offset', default=0, type=int)
    project_id = request.args.get('project_id') # Peut être utilisé pour filtrer par projet si implémenté dans ParcelService

    # Pour l'instant, on ignore project_id car ParcelService ne le gère pas
    # On pourrait lister les fichiers .geojson et les paginer manuellement,
    # ou utiliser une base de données.
    # Pour simplifier, on retourne une liste paginée à partir des fichiers existants.
    # Ici, on simule une liste paginée à partir des fichiers existants.
    all_parcel_files = list((BASE_DIR / "parcels").glob("[!all_]*.geojson"))
    start_idx = offset
    end_idx = start_idx + limit
    selected_files = all_parcel_files[start_idx:end_idx]

    parcels_data = []
    for file_path in selected_files:
        parcel_id = file_path.stem
        try:
            gdf = gpd.read_file(file_path)
            if not gdf.empty:
                parcel_data = gdf.iloc[0].to_dict()
                parcel_data['geometry'] = mapping(gdf.geometry.iloc[0])
                parcel_data['id'] = parcel_id
                parcels_data.append(parcel_data)
        except Exception as e:
            log.error(f"Erreur lecture parcelle {file_path.stem}: {e}")

    return jsonify({
        "data": parcels_data,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(all_parcel_files) # Total approximatif
        }
    })

# --- Parcelles - Adaptation POST /parcels/bounds ---
@app.route('/api/parcels/bounds', methods=['POST']) # Toujours POST car le body peut contenir des filtres complexes à l'avenir
@handle_errors
def get_parcels_by_bounds_adapted():
    # Lire les paramètres de la requête GET ou POST (params ou body)
    # Flask met les paramètres GET dans request.args et POST body dans request.data/form
    # On va prioriser les params GET s'ils sont présents, sinon on essaie le body POST (ancienne façon)
    minx = request.args.get('minx', type=float)
    miny = request.args.get('miny', type=float)
    maxx = request.args.get('maxx', type=float)
    maxy = request.args.get('maxy', type=float)
    crs = request.args.get('crs', DEFAULT_CRS_WGS84)
    buffer_m = request.args.get('buffer_m', type=float)
    # project_id = request.args.get('project_id') # Peut être utilisé pour filtrer si implémenté

    # Si les params GET ne sont pas trouvés, on essaie le body POST (JSON)
    if minx is None or miny is None or maxx is None or maxy is None:
        try:
            data = BoundsModel.parse_raw(request.data)
            bounds = data
        except ValidationError:
            # Si ce n'est pas non plus un JSON valide, erreur
            return jsonify({"error": "Paramètres minx, miny, maxx, maxy requis"}), 400
    else:
        # Si params GET sont trouvés, on les utilise
        bounds = BoundsModel(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=crs, buffer_m=buffer_m
        )

    result = parcel_service.get_parcels_by_bounds(bounds)
    # 'features' est déjà renommé en 'data' dans get_parcels_by_bounds
    return jsonify(result)

# --- Parcelles - Recherche ---
@app.route('/api/parcels/search', methods=['POST'])
@handle_errors
def search_parcels():
    data = SearchModel.parse_raw(request.data)
    query = data.query
    limit = data.limit
    offset = data.offset
    filters = data.filters or {}

    # --- Logique de recherche (simplifiée) ---
    # Lire le fichier agrégé
    try:
        gdf = gpd.read_file(parcel_service.all_parcels_file)
    except Exception as e:
        log.error(f"Erreur lecture fichier agrégé pour recherche: {e}")
        return jsonify({"error": "Erreur interne"}), 500

    # Filtrer par attributs (simplifié)
    mask = pd.Series([True] * len(gdf))
    for key, value in filters.items():
        if key in gdf.columns:
            if isinstance(value, list):
                mask &= gdf[key].isin(value)
            else:
                mask &= gdf[key].astype(str).str.contains(str(value), na=False, case=False)

    # Filtrer par texte (simplifié - recherche dans name, commune, etc.)
    if query:
        text_mask = (
            gdf['name'].astype(str).str.contains(query, na=False, case=False) |
            gdf['commune'].astype(str).str.contains(query, na=False, case=False) |
            gdf['numero'].astype(str).str.contains(query, na=False, case=False)
        )
        mask &= text_mask

    filtered_gdf = gdf[mask]

    # Appliquer pagination
    start_idx = offset
    end_idx = start_idx + limit
    paginated_gdf = filtered_gdf.iloc[start_idx:end_idx]

    # Convertir en GeoJSON-like
    features = json.loads(paginated_gdf.to_json())['features']

    return jsonify({
        "data": features,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(filtered_gdf)
        }
    })

# --- Parcelles - Opérations groupées ---
@app.route('/api/parcels/bulk', methods=['POST']) # Create
@handle_errors
def bulk_create_parcels():
    data = BulkParcelCreateModel.parse_raw(request.data)
    created_ids = []
    errors = []
    for i, parcel_data in enumerate(data.parcels):
        try:
            result = parcel_service.create_parcel(parcel_data)
            created_ids.append(result['id'])
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    return jsonify({"created_ids": created_ids, "errors": errors})

@app.route('/api/parcels/bulk', methods=['PUT']) # Update
@handle_errors
def bulk_update_parcels():
    data = BulkParcelUpdateModel.parse_raw(request.data)
    updated_ids = []
    errors = []
    for i, update_info in enumerate(data.updates):
        parcel_id = update_info.get("id")
        update_data = update_info.get("data", {})
        # Charger la parcelle existante
        existing_parcel = parcel_service.get_parcel(parcel_id)
        if not existing_parcel:
            errors.append({"id": parcel_id, "error": "Parcelle introuvable"})
            continue

        # Charger le GeoJSON existant
        parcel_file = BASE_DIR / "parcels" / f"{parcel_id}.geojson"
        gdf = gpd.read_file(parcel_file)
        if gdf.empty:
             errors.append({"id": parcel_id, "error": "Erreur lecture parcelle"})
             continue

        # Mettre à jour les propriétés
        for key, value in update_data.items():
            if key in gdf.columns:
                 gdf.loc[0, key] = value # Mettre à jour la première ligne

        # Recalculer si nécessaire (ex: superficie si la géométrie change)
        if 'geometry' in update_
            geom_shape = shape(update_data['geometry'])
            if not geom_shape.is_valid:
                geom_shape = geom_shape.buffer(0)
            gdf.loc[0, 'geometry'] = geom_shape
            gdf.loc[0, 'superficie_m2'] = round(geom_shape.area, 2)
            gdf.loc[0, 'superficie_ha'] = round(geom_shape.area / 10000, 4)
            gdf.loc[0, 'perimetre_m'] = round(geom_shape.length, 2)
            centroid = geom_shape.centroid
            gdf.loc[0, 'centroid_x'] = round(centroid.x, 2)
            gdf.loc[0, 'centroid_y'] = round(centroid.y, 2)


        # Réécrire le fichier
        gdf.to_file(parcel_file, driver='GeoJSON')
        # Mettre à jour l'agrégat
        parcel_service._update_aggregate()
        updated_ids.append(parcel_id)
        log.info(f"✅ Parcelle mise à jour: {parcel_id}")

    return jsonify({"updated_ids": updated_ids, "errors": errors})

@app.route('/api/parcels/bulk', methods=['DELETE']) # Delete
@handle_errors
def bulk_delete_parcels():
    data = BulkParcelDeleteModel.parse_raw(request.data)
    deleted_ids = []
    errors = []
    for i, parcel_id in enumerate(data.ids):
        success = parcel_service.delete_parcel(parcel_id)
        if success:
            deleted_ids.append(parcel_id)
        else:
            errors.append({"id": parcel_id, "error": "Parcelle introuvable"})
    return jsonify({"deleted_ids": deleted_ids, "errors": errors})

@app.route('/api/parcels/<parcel_id>', methods=['GET'])
@handle_errors
def get_parcel(parcel_id):
    output_crs = request.args.get('crs', DEFAULT_CRS)
    parcel = parcel_service.get_parcel(parcel_id, output_crs)
    if not parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify(parcel)

@app.route('/api/parcels/<parcel_id>', methods=['PUT'])
@handle_errors
def update_parcel(parcel_id):
    data = ParcelUpdateModel.parse_raw(request.data)
    updated_parcel = parcel_service.update_parcel(parcel_id, data)
    if not updated_parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify(updated_parcel)

@app.route('/api/parcels/<parcel_id>', methods=['DELETE'])
@handle_errors
def delete_parcel(parcel_id):
    if not parcel_service.delete_parcel(parcel_id):
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify({"message": "Parcelle supprimée", "id": parcel_id})

# --- CRS Info ---
@app.route('/api/crs/info', methods=['GET'])
@handle_errors
def get_crs_info_adapted():
    epsg = request.args.get('epsg', '32630') # Lire le paramètre epsg
    if not epsg.startswith('EPSG:'):
        epsg = f"EPSG:{epsg}"
    try:
        info = coord_service.get_crs_info(epsg)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": f"Erreur CRS: {e}"}), 400

# --- Routes OGC (WMS/WFS) -----
import subprocess
QGIS_BIN = "/usr/lib/cgi-bin/qgis_mapserv.fcgi"
@app.route('/api/ogc/<service>', methods=['GET'])
@handle_errors
def ogc_service(service):
    """Service OGC optimisé pour Leaflet"""
    if service.upper() not in ['WMS', 'WFS', 'WCS']:
        return jsonify({"error": "Service non supporté"}), 400
    if not os.path.isfile(QGIS_BIN):
        return jsonify({"error": "QGIS Server absent"}), 503
    qs = request.query_string.decode()
    map_param = request.args.get('MAP')
    # Détermination projet
    if map_param:
        project_file = str((PROJECTS_DIR / map_param).resolve())
        if not project_file.startswith(str(PROJECTS_DIR)):
            return jsonify({"error": "Projet non autorisé"}), 403
    else:
        project_file = str(DEFAULT_PROJECT)
    if not os.path.exists(project_file):
        return jsonify({"error": "Projet introuvable"}), 404
    # Ajout CRS par défaut si absent
    if 'GETMAP' in qs.upper() and 'CRS=' not in qs.upper() and 'SRS=' not in qs.upper():
        qs += f"&CRS={DEFAULT_CRS}"
    env = os.environ.copy()
    env.update({
        "QUERY_STRING": qs,
        "QGIS_PROJECT_FILE": project_file,
        "QT_QPA_PLATFORM": "offscreen",
        "REQUEST_METHOD": "GET"
    })
    try:
        result = subprocess.run(
            [QGIS_BIN], env=env, capture_output=True, timeout=30
        )
        if result.returncode != 0:
            log.error(f"Erreur QGIS Server: {result.stderr.decode()}")
            return jsonify({"error": "Erreur service OGC"}), 500
        # Détection type contenu
        output = result.stdout
        if output.startswith(b"\x89PNG"):
            content_type = "image/png"
        elif output.startswith(b"%PDF"):
            content_type = "application/pdf"
        elif b"<?xml" in output or b"<ServiceException" in output:
            content_type = "text/xml"
        else:
            content_type = "application/octet-stream"
        response = Response(output, content_type=content_type)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout service OGC"}), 504
    except Exception as e:
        log.error(f"Erreur OGC: {e}")
        return jsonify({"error": str(e)}), 500

# --- Routes Rapports PDF -----
@app.route('/api/reports/parcel/<parcel_id>', methods=['GET'])
@handle_errors
@require_session
def generate_parcel_report_adapted(parcel_id, session):
    format_requested = request.args.get('format', 'pdf').lower() # Lire le format
    if format_requested not in ['pdf', 'png', 'jpg', 'jpeg']:
        return jsonify({"error": "Format non supporté"}), 400

    qgis_mgr = get_qgis_manager()
    if not qgis_mgr.is_initialized():
        return jsonify({"error": "QGIS non initialisé"}), 500

    parcel = parcel_service.get_parcel(parcel_id)
    if not parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404

    # Vérification cache spécifique au format
    cache_key = f"report:{parcel_id}:{format_requested}"
    cached_file = cache_get(cache_key)
    if cached_file:
        # Pour les PDF, on le décode en latin1 car il a été encodé comme ça
        if format_requested == 'pdf':
            pdf_bytes = cached_file.encode('latin1')
            return send_file(
                BytesIO(pdf_bytes),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"parcel_{parcel_id}.{format_requested}"
            )
        else: # Image
            # Pour les images, on suppose que c'était une chaîne encodée en base64 ou directement des bytes encodées en latin1
            # Pour simplifier, on suppose que c'était encodé en latin1 comme les PDFs
            img_bytes = cached_file.encode('latin1')
            return send_file(
                BytesIO(img_bytes),
                mimetype=f"image/{format_requested}",
                as_attachment=True,
                download_name=f"parcel_{parcel_id}.{format_requested}"
            )

    classes = qgis_mgr.get_classes()
    QgsProject = classes['QgsProject']
    QgsVectorLayer = classes['QgsVectorLayer']
    QgsPrintLayout = classes['QgsPrintLayout']
    QgsLayoutItemMap = classes['QgsLayoutItemMap']
    QgsLayoutItemLabel = classes['QgsLayoutItemLabel']
    QgsLayoutExporter = classes['QgsLayoutExporter']
    QgsRectangle = classes['QgsRectangle']
    QgsLayoutPoint = classes['QgsLayoutPoint']
    QgsLayoutSize = classes['QgsLayoutSize']
    QgsUnitTypes = classes['QgsUnitTypes']
    QByteArray = classes['QByteArray']
    QBuffer = classes['QBuffer']
    QIODevice = classes['QIODevice']
    QFont = classes['QFont']
    QImage = classes['QImage']
    QPainter = classes['QPainter']

    try:
        geometry = shape(parcel['geometry'])
        bounds = geometry.bounds
        project = session.get_project(QgsProject)
        project.clear()

        parcels_path = str(BASE_DIR / "parcels" / "all_parcels.geojson")
        vlayer = QgsVectorLayer(parcels_path, "Parcelles", "ogr")
        if not vlayer.isValid():
            return jsonify({"error": "Couche parcelles invalide"}), 500
        project.addMapLayer(vlayer)

        layout = QgsPrintLayout(project)
        layout.initializeDefaults()
        layout.setName("Rapport Parcelle")

        map_item = QgsLayoutItemMap(layout)
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.15
        map_extent = QgsRectangle(
            bounds[0] - margin, bounds[1] - margin,
            bounds[2] + margin, bounds[3] + margin
        )
        map_item.setExtent(map_extent)
        map_item.setLayers([vlayer])
        layout.addLayoutItem(map_item)
        map_item.attemptMove(QgsLayoutPoint(10, 30, QgsUnitTypes.LayoutMillimeters))
        map_item.attemptResize(QgsLayoutSize(190, 150, QgsUnitTypes.LayoutMillimeters))

        title = QgsLayoutItemLabel(layout)
        title.setText(f"Rapport de Parcelle")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addLayoutItem(title)
        title.attemptMove(QgsLayoutPoint(10, 5, QgsUnitTypes.LayoutMillimeters))
        title.attemptResize(QgsLayoutSize(190, 12, QgsUnitTypes.LayoutMillimeters))

        info_text = (
            f"Parcelle : {parcel['name']}\n"
            f"Commune : {parcel['commune']}\n"
            f"Section : {parcel['section']} - N° {parcel['numero']}\n"
            f"Superficie : {parcel['superficie_ha']} ha ({parcel['superficie_m2']} m²)\n"
            f"Périmètre : {parcel.get('perimetre_m', 'N/A')} m\n"
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        )
        info_label = QgsLayoutItemLabel(layout)
        info_label.setText(info_text)
        info_label.setFont(QFont("Arial", 9))
        layout.addLayoutItem(info_label)
        info_label.attemptMove(QgsLayoutPoint(10, 185, QgsUnitTypes.LayoutMillimeters))
        info_label.attemptResize(QgsLayoutSize(190, 40, QgsUnitTypes.LayoutMillimeters))

        if format_requested == 'pdf':
            exporter = QgsLayoutExporter(layout)
            pdf_data = QByteArray()
            pdf_buffer = QBuffer(pdf_data)
            pdf_buffer.open(QIODevice.WriteOnly)
            result = exporter.exportToPdf(pdf_buffer, QgsLayoutExporter.PdfExportSettings())
            if result != QgsLayoutExporter.Success:
                return jsonify({"error": "Échec export PDF"}), 500
            output_bytes = bytes(pdf_data)
            mimetype = 'application/pdf'
        else: # Image (png, jpg, jpeg)
            # Création d'une image via le renderer
            map_settings = classes['QgsMapSettings']()
            map_settings.setLayers([vlayer])
            map_settings.setDestinationCrs(project.crs())
            map_settings.setExtent(map_extent)
            map_settings.setOutputSize(classes['QSize'](800, 600)) # Taille arbitraire
            renderer = classes['QgsMapRendererParallelJob'](map_settings)
            renderer.start()
            renderer.waitForFinished()
            image = renderer.renderedImage()

            # Convertir QImage en bytes
            ba = QByteArray()
            buffer = QBuffer(ba)
            buffer.open(QIODevice.WriteOnly)
            # Sauvegarder dans le format demandé
            success = image.save(buffer, format_requested.upper() if format_requested != 'jpg' else 'JPEG')
            if not success:
                 return jsonify({"error": "Échec export image"}), 500
            output_bytes = bytes(ba)
            mimetype = f"image/{format_requested}"


        # Mise en cache - on encode les bytes en latin1 pour le stockage
        cache_set(cache_key, output_bytes, expire=7200)

        return send_file(
            BytesIO(output_bytes),
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"parcel_{parcel_id}.{format_requested}"
        )
    except Exception as e:
        log.error(f"Erreur génération {format_requested.upper()}: {e}", exc_info=True)
        return jsonify({"error": f"Erreur génération rapport: {str(e)}"}), 500

# --- Rapports - Projet (esquisse) ---
@app.route('/api/reports/project/<project_id>', methods=['GET'])
@handle_errors
def generate_project_report(project_id):
    # Cette route nécessite une logique plus complexe pour agréger les données
    # d'un projet QGIS (potentiellement plusieurs couches) et générer un rapport.
    # Pour l'instant, on renvoie une erreur.
    return jsonify({"error": "Génération de rapport projet non implémentée"}), 501

# --- Routes Fichiers -----
@app.route('/api/files/upload', methods=['POST'])
@handle_errors
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400
    file = request.files['file']
    category = request.form.get('category', 'other')
    if not file.filename or category not in CATEGORIES:
        return jsonify({"error": "Fichier ou catégorie invalide"}), 400
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
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        # Métadonnées géospatiales
        if ext in ['shp', 'geojson', 'gpkg']:
            try:
                gdf = gpd.read_file(str(filepath))
                file_info['geo_info'] = {
                    'crs': str(gdf.crs) if gdf.crs else 'unknown',
                    'features': len(gdf),
                    'bounds': gdf.total_bounds.tolist() if not gdf.empty else None
                }
            except Exception as e:
                log.warning(f"Impossible de lire métadonnées géo: {e}")
        log.info(f"Fichier uploadé: {filename}")
        return jsonify(file_info), 201
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise

@app.route('/api/files/<category>', methods=['GET'])
@handle_errors
def list_files(category):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    files = []
    for file_path in (BASE_DIR / category).glob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
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
    if not file_path.exists():
        return jsonify({"error": "Fichier introuvable"}), 404
    return send_from_directory(BASE_DIR / category, safe_filename, as_attachment=True)

@app.route('/api/files/<category>/<filename>', methods=['DELETE'])
@handle_errors
def delete_file(category, filename):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    safe_filename = secure_filename(filename)
    file_path = BASE_DIR / category / safe_filename
    if not file_path.exists():
        return jsonify({"error": "Fichier introuvable"}), 404
    file_path.unlink()
    log.info(f"Fichier supprimé: {safe_filename}")
    return jsonify({"message": "Fichier supprimé", "filename": safe_filename})

# --- Routes Analytics ---
@app.route('/api/analytics/data', methods=['POST'])
@handle_errors
def log_analytics_data():
    data = AnalyticsDataModel.parse_raw(request.data)
    analytics_service.log_event(data)
    return jsonify({"message": "Données analytics logguées", "event_type": data.event_type})

@app.route('/api/analytics/summary', methods=['GET'])
@handle_errors
def get_analytics_summary():
    summary = analytics_service.get_summary()
    return jsonify({"data": summary}) # Toujours envelopper dans "data"

@app.route('/api/analytics/events', methods=['GET'])
@handle_errors
def get_analytics_events():
    limit = request.args.get('limit', default=100, type=int)
    # Lire les événements du fichier
    events = analytics_service._read_events()
    # Trier par timestamp descendant et limiter
    sorted_events = sorted(events, key=lambda x: x['timestamp'], reverse=True)
    paginated_events = sorted_events[:limit]
    return jsonify({
        "data": paginated_events,
        "pagination": {
            "limit": limit,
            "offset": 0,
            "total": len(events)
        }
    })

@app.route('/api/analytics/behavior', methods=['POST'])
@handle_errors
def log_user_behavior():
    data = UserBehaviorModel.parse_raw(request.data)
    analytics_service.log_user_behavior(data)
    return jsonify({"message": "Comportement utilisateur loggué", "action": data.action})

# --- Routes Performance Metrics ---
@app.route('/api/performance/metrics', methods=['POST'])
@handle_errors
def log_performance_metric():
    data = PerformanceMetricsModel.parse_raw(request.data)
    performance_service.log_metric(data)
    return jsonify({"message": "Métrique performance logguée", "metric_name": data.metric_name})

@app.route('/api/performance/metrics', methods=['GET'])
@handle_errors
def get_performance_metrics():
    limit = request.args.get('limit', default=100, type=int)
    metrics = performance_service.get_metrics(limit=limit)
    return jsonify({
        "data": metrics,
        "pagination": {
            "limit": limit,
            "offset": 0,
            "total": len(metrics) # Note: total est approximatif ici car on lit tout le fichier
        }
    })

# --- Routes Admin -----
@app.route('/api/admin/stats', methods=['GET'])
@handle_errors
def admin_stats():
    stats = {
        "parcels": {
            "total": len(list((BASE_DIR / "parcels").glob("[!all_]*.geojson"))),
            "storage_mb": round(sum(
                f.stat().st_size for f in (BASE_DIR / "parcels").glob("*.geojson")
            ) / (1024 * 1024), 2)
        },
        "sessions": {
            "active": len(project_sessions),
            "total_created": len(project_sessions)
        },
        "storage": {"categories": {}},
        "system": {
            "qgis": get_qgis_manager().is_initialized(),
            "redis": redis_client.ping() if redis_client else False,
            "default_crs": DEFAULT_CRS,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    for category in CATEGORIES:
        cat_dir = BASE_DIR / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*"))
            stats["storage"]["categories"][category] = {
                "files": len(files),
                "storage_mb": round(sum(
                    f.stat().st_size for f in files if f.is_file()
                ) / (1024 * 1024), 2)
            }
    return jsonify(stats)

# --- Système - Stats alias ---
@app.route('/api/stats', methods=['GET'])
@handle_errors
def system_stats_alias():
    return admin_stats() # Réutilise la route existante

@app.route('/api/admin/cache/clear', methods=['POST'])
@handle_errors
def clear_cache():
    cleared = {"redis": False, "file_cache": False}
    if redis_client:
        try:
            redis_client.flushdb()
            cleared["redis"] = True
        except Exception as e:
            log.error(f"Erreur nettoyage Redis: {e}")
    with cache_lock:
        file_cache.clear()
        cleared["file_cache"] = True
    return jsonify({
        "message": "Cache nettoyé",
        "cleared": cleared,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

# --- Routes Documentation (mise à jour) ---
@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({
        "version": "3.1.0", # Version mise à jour
        "title": "API QGIS Server - Optimisée Mobile",
        "default_crs": DEFAULT_CRS,
        "endpoints": {
            "auth": {
                "login": "POST /api/auth/login",
                "register": "POST /api/auth/register",
                "refresh": "POST /api/auth/refresh",
                "logout": "POST /api/auth/logout",
                "note": "Implémentation partielle"
            },
            "projects": {
                "list": "GET /api/projects?limit=100&offset=0",
                "create": "POST /api/projects",
                "get": "GET /api/projects/{id}",
                "update": "PUT /api/projects/{id}",
                "delete": "DELETE /api/projects/{id}"
            },
            "parcels": {
                "list": "GET /api/parcels?limit=100&offset=0&project_id=...",
                "create": "POST /api/parcels",
                "update": "PUT /api/parcels/{id}",
                "by_bounds": "POST /api/parcels/bounds (body ou params)",
                "search": "POST /api/parcels/search",
                "bulk": {
                    "create": "POST /api/parcels/bulk",
                    "update": "PUT /api/parcels/bulk",
                    "delete": "DELETE /api/parcels/bulk"
                },
                "get": "GET /api/parcels/{id}",
                "delete": "DELETE /api/parcels/{id}"
            },
            "ogc": {
                "wms": "GET /api/ogc/WMS?REQUEST=GetMap&LAYERS=...",
                "wfs": "GET /api/ogc/WFS?REQUEST=GetFeature&TYPENAME=...",
                "wcs": "GET /api/ogc/WCS?...",
                "note": "Compatible Leaflet, supporte param MAP= pour projets personnalisés"
            },
            "reports": {
                "parcel_pdf": "GET /api/reports/parcel/{id}?format=pdf", # Exemple
                "parcel_image": "GET /api/reports/parcel/{id}?format=png",
                "project": "GET /api/reports/project/{id}",
                "note": "Génération avec cache, nécessite header X-Session-ID pour PDF"
            },
            "crs": {
                "transform": "POST /api/crs/transform",
                "info": "GET /api/crs/info?epsg=32630"
            },
            "analytics": {
                "log_data": "POST /api/analytics/data",
                "log_behavior": "POST /api/analytics/behavior",
                "summary": "GET /api/analytics/summary",
                "events": "GET /api/analytics/events?limit=100"
            },
            "performance": {
                "log_metric": "POST /api/performance/metrics",
                "get_metrics": "GET /api/performance/metrics?limit=100"
            },
            "files": {
                "upload": "POST /api/files/upload",
                "list": "GET /api/files/{category}",
                "download": "GET /api/files/{category}/{filename}",
                "delete": "DELETE /api/files/{category}/{filename}"
            },
            "system": {
                "health": "GET /api/health",
                "docs": "GET /api/docs",
                "stats": "GET /api/stats"
            }
        },
        "mobile_optimization": {
            "session_management": "Utilisez header X-Session-ID pour sessions persistantes (PDF)",
            "bounds_query": "Endpoint /api/parcels/bounds optimisé pour viewport mobile (params ou body)",
            "compression": "Toutes réponses JSON compressées automatiquement",
            "caching": "WMS tiles, rapports PDF/images mis en cache (Redis ou fichier)"
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.1.0", # Version mise à jour
        "services": {
            "qgis": get_qgis_manager().is_initialized(),
            "redis": redis_client.ping() if redis_client else False,
            "default_crs": DEFAULT_CRS
        }
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "API QGIS Server",
        "version": "3.1.0", # Version mise à jour
        "status": "operational",
        "documentation": "/api/docs"
    })

# ================================================================
# Middleware & Error Handlers
# ================================================================
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Session-ID'
    response.headers['Access-Control-Expose-Headers'] = 'X-Session-ID'
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint introuvable",
        "documentation": "/api/docs"
    }), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "Fichier trop volumineux",
        "max_size_mb": app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    }), 413

@app.errorhandler(500)
def internal_error(error):
    log.error(f"Erreur 500: {error}")
    return jsonify({"error": "Erreur serveur interne"}), 500

# ================================================================
# Point d'entrée
# ================================================================
if __name__ == '__main__':
    # Initialisation QGIS
    qgis_mgr = get_qgis_manager()
    success, error = qgis_mgr.initialize()
    if not success:
        log.error(f"Échec initialisation QGIS: {error}")
        sys.exit(1)
    # Démarrage nettoyage sessions
    cleanup_thread = Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    log.info("=" * 70)
    log.info("🚀 API QGIS Server v3.1.0 - Optimisée pour Render & Mobile")
    log.info(f"📁 Données: {BASE_DIR}")
    log.info(f"🗺️  CRS par défaut: {DEFAULT_CRS}")
    log.info(f"🔌 Redis: {'✅ Activé' if redis_client else '⚠️ Cache fichier'}")
    log.info(f"🧠 QGIS: {'✅ Initialisé' if qgis_mgr.is_initialized() else '❌ Erreur'}")
    log.info(f"📊 Sessions: Timeout {SESSION_TIMEOUT.seconds // 60}min")
    log.info("=" * 70)
    # Configuration serveur
    port = int(os.getenv('PORT', 10000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
