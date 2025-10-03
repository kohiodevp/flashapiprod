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
            return redis_client.get(key)
        except:
            pass
    with cache_lock:
        return file_cache.get(key)

def cache_set(key: str, value: str, expire: int = 3600):
    """Enregistrement cache avec fallback"""
    if redis_client:
        try:
            redis_client.setex(key, expire, value)
            return
        except:
            pass
    with cache_lock:
        file_cache[key] = value

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

    def create_parcel(self, parcel_data: ParcelCreateModel) -> Dict[str, Any]:
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
                return {'count': 0, 'features': [], 'bbox': list(bbox)}
            
            # Conversion GeoJSON optimisée
            features = json.loads(gdf.to_json())['features']
            
            return {
                'count': len(features),
                'features': features,
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
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.0",
        "services": {
            "qgis": get_qgis_manager().is_initialized(),
            "redis": redis_client.ping() if redis_client else False,
            "default_crs": DEFAULT_CRS
        }
    })

# ----- Routes Parcelles -----
@app.route('/api/parcels', methods=['POST'])
@handle_errors
def create_parcel():
    data = ParcelCreateModel.parse_raw(request.data)
    result = parcel_service.create_parcel(data)
    return jsonify(result), 201

@app.route('/api/parcels/bounds', methods=['POST'])
@handle_errors
def get_parcels_by_bounds():
    """Route optimisée pour mobile : récupération par emprise visible"""
    data = BoundsModel.parse_raw(request.data)
    result = parcel_service.get_parcels_by_bounds(data)
    return jsonify(result)

@app.route('/api/parcels/<parcel_id>', methods=['GET'])
@handle_errors
def get_parcel(parcel_id):
    output_crs = request.args.get('crs', DEFAULT_CRS)
    parcel = parcel_service.get_parcel(parcel_id, output_crs)
    if not parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify(parcel)

@app.route('/api/parcels/<parcel_id>', methods=['DELETE'])
@handle_errors
def delete_parcel(parcel_id):
    if not parcel_service.delete_parcel(parcel_id):
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify({"message": "Parcelle supprimée", "id": parcel_id})

# ----- Routes Coordonnées -----
@app.route('/api/crs/transform', methods=['POST'])
@handle_errors
def transform_coordinates():
    data = CoordinateTransformModel.parse_raw(request.data)
    result = coord_service.transform_coordinates(data)
    return jsonify(result)

@app.route('/api/crs/info', methods=['GET'])
@handle_errors
def get_crs_info():
    epsg = request.args.get('epsg', '32630')
    return jsonify(coord_service.get_crs_info(epsg))

# ----- Routes OGC (WMS/WFS) -----
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

# ----- Routes Rapports PDF -----
@app.route('/api/reports/parcel/<parcel_id>', methods=['GET'])
@handle_errors
@require_session
def generate_parcel_report(parcel_id, session):
    """Génération rapport PDF optimisée"""
    qgis_mgr = get_qgis_manager()
    if not qgis_mgr.is_initialized():
        return jsonify({"error": "QGIS non initialisé"}), 500
    
    parcel = parcel_service.get_parcel(parcel_id)
    if not parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    
    # Vérification cache
    cache_key = f"report:{parcel_id}"
    cached_pdf = cache_get(cache_key)
    if cached_pdf:
        return send_file(
            BytesIO(cached_pdf.encode('latin1')),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"parcel_{parcel_id}.pdf"
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
    
    try:
        geometry = shape(parcel['geometry'])
        bounds = geometry.bounds
        
        # Création projet temporaire
        project = session.get_project(QgsProject)
        project.clear()
        
        # Chargement couche parcelles
        parcels_path = str(BASE_DIR / "parcels" / "all_parcels.geojson")
        vlayer = QgsVectorLayer(parcels_path, "Parcelles", "ogr")
        if not vlayer.isValid():
            return jsonify({"error": "Couche parcelles invalide"}), 500
        project.addMapLayer(vlayer)
        
        # Création layout
        layout = QgsPrintLayout(project)
        layout.initializeDefaults()
        layout.setName("Rapport Parcelle")
        
        # Carte principale
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
        
        # Titre
        title = QgsLayoutItemLabel(layout)
        title.setText(f"Rapport de Parcelle")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addLayoutItem(title)
        title.attemptMove(QgsLayoutPoint(10, 5, QgsUnitTypes.LayoutMillimeters))
        title.attemptResize(QgsLayoutSize(190, 12, QgsUnitTypes.LayoutMillimeters))
        
        # Informations parcelle
        info_text = (
            f"Parcelle : {parcel['name']}\n"
            f"Commune : {parcel['commune']}\n"
            f"Section : {parcel['section']} - N° {parcel['numero']}\n"
            f"Superficie : {parcel['superficie_ha']} ha ({parcel['superficie_m2']} m²)\n"
            f"Périmètre : {parcel.get('perimetre_m', 'N/A')} m\n"
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )
        info_label = QgsLayoutItemLabel(layout)
        info_label.setText(info_text)
        info_label.setFont(QFont("Arial", 9))
        layout.addLayoutItem(info_label)
        info_label.attemptMove(QgsLayoutPoint(10, 185, QgsUnitTypes.LayoutMillimeters))
        info_label.attemptResize(QgsLayoutSize(190, 40, QgsUnitTypes.LayoutMillimeters))
        
        # Export PDF
        exporter = QgsLayoutExporter(layout)
        pdf_data = QByteArray()
        pdf_buffer = QBuffer(pdf_data)
        pdf_buffer.open(QIODevice.WriteOnly)
        
        result = exporter.exportToPdf(pdf_buffer, QgsLayoutExporter.PdfExportSettings())
        if result != QgsLayoutExporter.Success:
            return jsonify({"error": "Échec export PDF"}), 500
        
        pdf_bytes = bytes(pdf_data)
        
        # Mise en cache
        cache_set(cache_key, pdf_bytes.decode('latin1'), expire=7200)
        
        return send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"parcel_{parcel_id}.pdf"
        )
        
    except Exception as e:
        log.error(f"Erreur génération PDF: {e}", exc_info=True)
        return jsonify({"error": f"Erreur génération rapport: {str(e)}"}), 500

# ----- Routes Fichiers -----
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

# ----- Routes Admin -----
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

# ----- Routes Documentation -----
@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({
        "version": "3.0.0",
        "title": "API QGIS Server - Optimisée Mobile",
        "default_crs": DEFAULT_CRS,
        "endpoints": {
            "health": {
                "path": "/api/health",
                "method": "GET",
                "description": "État du service"
            },
            "parcels": {
                "create": "POST /api/parcels",
                "by_bounds": "POST /api/parcels/bounds (optimisé mobile)",
                "get": "GET /api/parcels/{id}",
                "delete": "DELETE /api/parcels/{id}"
            },
            "ogc": {
                "wms": "GET /api/ogc/WMS?REQUEST=GetMap&LAYERS=...",
                "wfs": "GET /api/ogc/WFS?REQUEST=GetFeature&TYPENAME=...",
                "note": "Compatible Leaflet, supporte param MAP= pour projets personnalisés"
            },
            "reports": {
                "parcel_pdf": "GET /api/reports/parcel/{id}",
                "note": "Génération PDF avec cache, nécessite header X-Session-ID"
            },
            "coordinates": {
                "transform": "POST /api/crs/transform",
                "info": "GET /api/crs/info?epsg=32630"
            },
            "files": {
                "upload": "POST /api/files/upload",
                "list": "GET /api/files/{category}",
                "download": "GET /api/files/{category}/{filename}",
                "delete": "DELETE /api/files/{category}/{filename}"
            }
        },
        "mobile_optimization": {
            "session_management": "Utilisez header X-Session-ID pour sessions persistantes",
            "bounds_query": "Endpoint /api/parcels/bounds optimisé pour viewport mobile",
            "compression": "Toutes réponses JSON compressées automatiquement",
            "caching": "WMS tiles et rapports mis en cache (Redis ou fichier)"
        }
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "API QGIS Server",
        "version": "3.0.0",
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
    log.info("🚀 API QGIS Server v3.0.0 - Optimisée pour Render & Mobile")
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