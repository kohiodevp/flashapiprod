"""
API QGIS Server - Version Production
Version: 3.0.0
Optimisée pour: Sécurité, Performance, Robustesse
"""

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
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple, Union
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, shape, mapping
from shapely.ops import transform
from shapely.validation import make_valid
from pyproj import Transformer, CRS
import fiona

from flask import Flask, request, jsonify, send_from_directory, Response, g
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
from pydantic import BaseModel, Field, validator, ValidationError, constr, conint
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache

import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import bcrypt

# =============================================================================
# CONFIGURATION ET CONSTANTES
# =============================================================================

class Config:
    """Configuration centralisée"""
    # Application
    APP_NAME = "QGIS-API-Production"
    VERSION = "3.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Sécurité
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    JWT_SECRET = os.getenv('JWT_SECRET', secrets.token_hex(32))
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'shp', 'shx', 'dbf', 'prj', 'geojson', 'json', 'gpkg', 'kml', 'csv'}
    
    # CRS
    DEFAULT_CRS = "EPSG:32630"
    DEFAULT_CRS_WGS84 = "EPSG:4326"
    ALLOWED_CRS = ['EPSG:32630', 'EPSG:4326', 'EPSG:3857', 'EPSG:2154']
    
    # Chemins
    BASE_DIR = Path(os.getenv('BASE_DIR', '/data'))
    QGIS_PROJECT_FILE = os.getenv('QGIS_PROJECT_FILE', '/etc/qgis/projects/project.qgs')
    QGIS_SERVER_BIN = '/usr/lib/cgi-bin/qgis_mapserv.fcgi'
    
    # Catégories de fichiers
    CATEGORIES = ["shapefiles", "csv", "geojson", "projects", "other", "tiles", "parcels", "documents"]
    
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
    REDIS_SOCKET_TIMEOUT = 5
    REDIS_SOCKET_CONNECT_TIMEOUT = 5
    REDIS_MAX_CONNECTIONS = 50
    REDIS_RETRY_ON_TIMEOUT = True
    
    # Cache
    CACHE_TYPE = 'RedisCache'
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_KEY_PREFIX = 'qgis_api:'
    
    # Rate Limiting
    RATELIMIT_STORAGE_URI = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_HEADERS_ENABLED = True
    
    # Pagination
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 500
    
    # Timeouts
    QGIS_SERVER_TIMEOUT = 30
    FILE_OPERATION_TIMEOUT = 60
    
    # Logging
    LOG_LEVEL = logging.INFO if not DEBUG else logging.DEBUG
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'

# =============================================================================
# INITIALISATION APPLICATION
# =============================================================================

app = Flask(__name__)
app.config.from_object(Config)

# CORS avec restrictions
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["X-Total-Count", "X-Page", "X-Page-Size"],
        "max_age": 3600
    }
})

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATELIMIT_DEFAULT],
    storage_uri=Config.RATELIMIT_STORAGE_URI if not Config.DEBUG else None,
    strategy="fixed-window"
)

# Cache
cache = Cache(app, config={
    'CACHE_TYPE': Config.CACHE_TYPE if not Config.DEBUG else 'SimpleCache',
    'CACHE_REDIS_HOST': Config.REDIS_HOST,
    'CACHE_REDIS_PORT': Config.REDIS_PORT,
    'CACHE_REDIS_DB': Config.REDIS_DB,
    'CACHE_REDIS_PASSWORD': Config.REDIS_PASSWORD,
    'CACHE_DEFAULT_TIMEOUT': Config.CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': Config.CACHE_KEY_PREFIX
})

# Logging configuré
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            '/var/log/qgis_api.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ) if not Config.DEBUG else logging.NullHandler()
    ]
)
log = logging.getLogger(Config.APP_NAME)

# Création des répertoires avec permissions appropriées
for directory in Config.CATEGORIES:
    dir_path = Config.BASE_DIR / directory
    dir_path.mkdir(parents=True, exist_ok=True)
    os.chmod(dir_path, 0o755)

# =============================================================================
# CONNEXION REDIS ROBUSTE
# =============================================================================

class RedisManager:
    """Gestionnaire Redis avec retry et fallback"""
    
    def __init__(self):
        self.client = None
        self.connection_pool = None
        self._connect()
    
    def _connect(self):
        """Établit la connexion avec retry"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.connection_pool = redis.ConnectionPool(
                    host=Config.REDIS_HOST,
                    port=Config.REDIS_PORT,
                    db=Config.REDIS_DB,
                    password=Config.REDIS_PASSWORD,
                    decode_responses=True,
                    socket_timeout=Config.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=Config.REDIS_SOCKET_CONNECT_TIMEOUT,
                    max_connections=Config.REDIS_MAX_CONNECTIONS,
                    retry_on_timeout=Config.REDIS_RETRY_ON_TIMEOUT
                )
                
                self.client = redis.Redis(connection_pool=self.connection_pool)
                self.client.ping()
                log.info(f"Redis connecté avec succès (tentative {attempt + 1}/{max_retries})")
                return
                
            except (RedisError, RedisConnectionError) as e:
                log.warning(f"Échec connexion Redis (tentative {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    log.error("Redis non disponible - Mode dégradé activé")
                    self.client = None
    
    def get(self, key: str, default=None):
        """GET avec gestion d'erreur"""
        if not self.client:
            return default
        try:
            return self.client.get(key) or default
        except RedisError as e:
            log.error(f"Erreur Redis GET: {e}")
            return default
    
    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """SET avec gestion d'erreur"""
        if not self.client:
            return False
        try:
            return self.client.set(key, value, ex=ex)
        except RedisError as e:
            log.error(f"Erreur Redis SET: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """DELETE avec gestion d'erreur"""
        if not self.client:
            return False
        try:
            return bool(self.client.delete(key))
        except RedisError as e:
            log.error(f"Erreur Redis DELETE: {e}")
            return False
    
    def ping(self) -> bool:
        """Vérifie la connexion"""
        if not self.client:
            return False
        try:
            return self.client.ping()
        except RedisError:
            return False
    
    def flushdb(self) -> bool:
        """Vide la base"""
        if not self.client:
            return False
        try:
            return self.client.flushdb()
        except RedisError as e:
            log.error(f"Erreur Redis FLUSHDB: {e}")
            return False

redis_manager = RedisManager()

# =============================================================================
# MODÈLES PYDANTIC VALIDÉS
# =============================================================================

class LayerModel(BaseModel):
    """Modèle de couche géographique"""
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    source: constr(min_length=1, max_length=500)
    geom: constr(regex="^(Point|LineString|Polygon|MultiPoint|MultiLineString|MultiPolygon)$")
    lid: Optional[str] = None
    crs: constr(regex=r"^EPSG:\d+$") = Config.DEFAULT_CRS
    
    @validator('crs')
    def validate_crs(cls, v):
        if v not in Config.ALLOWED_CRS:
            raise ValueError(f"CRS non autorisé. Valeurs acceptées: {', '.join(Config.ALLOWED_CRS)}")
        return v
    
    class Config:
        extra = 'forbid'

class ProjectModel(BaseModel):
    """Modèle de projet QGIS"""
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    crs_authid: constr(regex=r"^EPSG:\d+$") = Config.DEFAULT_CRS
    crs_proj4: str = "+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs"
    crs_wkt: str = ""
    srsid: conint(gt=0) = 3452
    srid: conint(gt=0) = 32630
    layers: List[LayerModel] = []
    description: Optional[constr(max_length=1000)] = None
    
    class Config:
        extra = 'forbid'

class ParcelCreateModel(BaseModel):
    """Modèle de création de parcelle"""
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    geometry: Dict[str, Any]
    commune: constr(min_length=1, max_length=100, strip_whitespace=True)
    section: constr(min_length=1, max_length=10, strip_whitespace=True)
    numero: constr(min_length=1, max_length=20, strip_whitespace=True)
    superficie: Optional[float] = Field(None, gt=0, le=1000000000)  # Max 1 million km²
    proprietaire: Optional[constr(max_length=200)] = None
    usage: Optional[constr(max_length=100)] = None
    crs: constr(regex=r"^EPSG:\d+$") = Config.DEFAULT_CRS_WGS84
    
    @validator('geometry')
    def validate_geometry(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Géométrie doit être un objet GeoJSON')
        
        if not v.get('type') or not v.get('coordinates'):
            raise ValueError('Géométrie GeoJSON invalide - type et coordinates requis')
        
        valid_types = ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon']
        if v.get('type') not in valid_types:
            raise ValueError(f"Type de géométrie invalide. Types acceptés: {', '.join(valid_types)}")
        
        # Validation basique des coordonnées
        coords = v.get('coordinates')
        if not coords or len(coords) == 0:
            raise ValueError('Coordonnées vides')
        
        return v
    
    @validator('crs')
    def validate_crs(cls, v):
        if v not in Config.ALLOWED_CRS:
            raise ValueError(f'CRS non autorisé. Valeurs acceptées: {", ".join(Config.ALLOWED_CRS)}')
        try:
            CRS.from_string(v)
        except Exception:
            raise ValueError(f'CRS invalide: {v}')
        return v
    
    class Config:
        extra = 'forbid'

class ParcelAnalysisModel(BaseModel):
    """Modèle d'analyse de parcelle"""
    parcel_id: constr(min_length=36, max_length=36)  # UUID
    analysis_type: constr(regex="^(superficie|perimetre|distance|buffer|centroid|intersection)$")
    parameters: Optional[Dict[str, Any]] = None
    output_crs: constr(regex=r"^EPSG:\d+$") = Config.DEFAULT_CRS
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        if not v:
            return v
        
        analysis_type = values.get('analysis_type')
        
        if analysis_type == 'buffer':
            if 'distance' not in v:
                raise ValueError('Le paramètre "distance" est requis pour l\'analyse buffer')
            if not isinstance(v['distance'], (int, float)) or v['distance'] <= 0:
                raise ValueError('La distance doit être un nombre positif')
            if v['distance'] > 100000:  # Max 100km
                raise ValueError('Distance buffer limitée à 100000m')
        
        if analysis_type == 'intersection':
            if 'geometry' not in v:
                raise ValueError('Le paramètre "geometry" est requis pour l\'intersection')
        
        return v
    
    class Config:
        extra = 'forbid'

class CoordinateTransformModel(BaseModel):
    """Modèle de transformation de coordonnées"""
    coordinates: List[List[float]]
    from_crs: constr(regex=r"^EPSG:\d+$") = Config.DEFAULT_CRS_WGS84
    to_crs: constr(regex=r"^EPSG:\d+$") = Config.DEFAULT_CRS
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if not v:
            raise ValueError('La liste de coordonnées ne peut pas être vide')
        if len(v) > 10000:
            raise ValueError('Maximum 10000 coordonnées par requête')
        
        for i, coord in enumerate(v):
            if not isinstance(coord, list) or len(coord) < 2:
                raise ValueError(f'Coordonnée {i} invalide - doit contenir au moins [x, y]')
            if not all(isinstance(c, (int, float)) for c in coord[:2]):
                raise ValueError(f'Coordonnée {i} invalide - x et y doivent être numériques')
        
        return v
    
    @validator('from_crs', 'to_crs')
    def validate_crs(cls, v):
        if v not in Config.ALLOWED_CRS:
            raise ValueError(f'CRS non autorisé. Valeurs acceptées: {", ".join(Config.ALLOWED_CRS)}')
        try:
            CRS.from_string(v)
        except Exception:
            raise ValueError(f'CRS invalide: {v}')
        return v
    
    class Config:
        extra = 'forbid'

# =============================================================================
# SYSTÈME D'AUTHENTIFICATION
# =============================================================================

class AuthService:
    """Service d'authentification JWT"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash un mot de passe"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Vérifie un mot de passe"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def create_token(user_id: str, roles: List[str] = None) -> str:
        """Crée un token JWT"""
        payload = {
            'user_id': user_id,
            'roles': roles or ['user'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=Config.JWT_EXPIRATION_HOURS),
            'iat': datetime.now(timezone.utc),
            'jti': str(uuid.uuid4())
        }
        return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
            
            # Vérifier si le token est blacklisté
            jti = payload.get('jti')
            if jti and redis_manager.get(f"blacklist:{jti}"):
                log.warning(f"Token blacklisté utilisé: {jti}")
                return None
            
            return payload
        except ExpiredSignatureError:
            log.warning("Token expiré")
            return None
        except InvalidTokenError as e:
            log.warning(f"Token invalide: {e}")
            return None
    
    @staticmethod
    def blacklist_token(token: str):
        """Ajoute un token à la blacklist"""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM], options={"verify_exp": False})
            jti = payload.get('jti')
            exp = payload.get('exp')
            
            if jti and exp:
                ttl = max(0, exp - int(datetime.now(timezone.utc).timestamp()))
                redis_manager.set(f"blacklist:{jti}", "1", ex=ttl)
        except Exception as e:
            log.error(f"Erreur blacklist token: {e}")

auth_service = AuthService()

# =============================================================================
# DÉCORATEURS DE SÉCURITÉ
# =============================================================================

def require_auth(roles: List[str] = None):
    """Décorateur pour exiger une authentification"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({"error": "Authorization header manquant"}), 401
            
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != 'bearer':
                return jsonify({"error": "Format Authorization invalide. Utilisez: Bearer <token>"}), 401
            
            token = parts[1]
            payload = auth_service.verify_token(token)
            
            if not payload:
                return jsonify({"error": "Token invalide ou expiré"}), 401
            
            # Vérification des rôles
            if roles:
                user_roles = payload.get('roles', [])
                if not any(role in user_roles for role in roles):
                    return jsonify({
                        "error": "Permissions insuffisantes",
                        "required_roles": roles
                    }), 403
            
            # Stocker les infos utilisateur dans le contexte de la requête
            g.user_id = payload.get('user_id')
            g.user_roles = payload.get('roles', [])
            g.token_jti = payload.get('jti')
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def handle_errors(f):
    """Décorateur pour la gestion centralisée des erreurs"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        request_id = g.get('request_id', 'unknown')
        
        try:
            return f(*args, **kwargs)
        
        except ValidationError as e:
            log.error(f"[{request_id}] Erreur validation: {e}")
            return jsonify({
                "error": "Erreur de validation",
                "details": [
                    {
                        "field": ".".join(str(x) for x in err['loc']),
                        "message": err['msg'],
                        "type": err['type']
                    }
                    for err in e.errors()
                ],
                "request_id": request_id
            }), 400
        
        except ValueError as e:
            log.error(f"[{request_id}] Erreur valeur: {e}")
            return jsonify({
                "error": str(e),
                "request_id": request_id
            }), 400
        
        except FileNotFoundError as e:
            log.error(f"[{request_id}] Fichier non trouvé: {e}")
            return jsonify({
                "error": "Ressource non trouvée",
                "request_id": request_id
            }), 404
        
        except PermissionError as e:
            log.error(f"[{request_id}] Permission refusée: {e}")
            return jsonify({
                "error": "Permission refusée",
                "request_id": request_id
            }), 403
        
        except TimeoutError as e:
            log.error(f"[{request_id}] Timeout: {e}")
            return jsonify({
                "error": "Timeout de l'opération",
                "request_id": request_id
            }), 504
        
        except Exception as e:
            log.error(f"[{request_id}] Erreur inattendue: {e}", exc_info=True)
            
            if Config.DEBUG:
                return jsonify({
                    "error": "Erreur serveur",
                    "message": str(e),
                    "type": type(e).__name__,
                    "request_id": request_id
                }), 500
            else:
                return jsonify({
                    "error": "Erreur serveur interne",
                    "request_id": request_id
                }), 500
    
    return wrapper

# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.before_request
def before_request():
    """Exécuté avant chaque requête"""
    # Génération d'un ID de requête unique
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()
    
    # Logging de la requête
    log.info(f"[{g.request_id}] {request.method} {request.path} - IP: {get_remote_address()}")

@app.after_request
def after_request(response):
    """Exécuté après chaque requête"""
    # Headers de sécurité
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
    
    # Temps de traitement
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        response.headers['X-Response-Time'] = f"{duration:.3f}s"
        log.info(f"[{g.request_id}] Response: {response.status_code} - Duration: {duration:.3f}s")
    
    return response

# =============================================================================
# SERVICE DE GESTION DES PARCELLES
# =============================================================================

class ParcelService:
    """Service de gestion des parcelles avec transactions atomiques"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / "parcels"
        self.base_dir.mkdir(exist_ok=True)
        self.default_crs = Config.DEFAULT_CRS
        self.metadata_file = self.base_dir / "metadata.json"
        self.lock_file = self.base_dir / ".lock"
        self._ensure_metadata()
    
    def _ensure_metadata(self):
        """Initialise le fichier de métadonnées"""
        if not self.metadata_file.exists():
            with self._file_lock():
                self.metadata_file.write_text(json.dumps({}, indent=2))
    
    @contextmanager
    def _file_lock(self, timeout: int = 10):
        """Lock pour opérations fichiers atomiques"""
        lock_path = self.lock_file
        lock_fd = None
        
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Impossible d'acquérir le lock")
                    time.sleep(0.1)
            
            yield
            
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except Exception as e:
                    log.error(f"Erreur libération lock: {e}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Charge les métadonnées"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Erreur lecture métadonnées: {e}")
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Sauvegarde les métadonnées de façon atomique"""
        temp_file = self.metadata_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.metadata_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def _validate_and_fix_geometry(self, geometry) -> Any:
        """Valide et corrige une géométrie"""
        if not geometry.is_valid:
            log.warning("Géométrie invalide détectée, tentative de correction")
            
            # Tentative de correction avec make_valid
            geometry = make_valid(geometry)
            
            if not geometry.is_valid:
                # Dernière tentative avec buffer(0)
                geometry = geometry.buffer(0)
                
                if not geometry.is_valid:
                    raise ValueError("Impossible de corriger la géométrie invalide")
            
            log.info("Géométrie corrigée avec succès")
        
        return geometry
    
    def _transform_geometry(self, geometry: Dict, from_crs: str, to_crs: str):
        """Transforme une géométrie entre CRS"""
        try:
            transformer = Transformer.from_crs(
                from_crs,
                to_crs,
                always_xy=True
            )
            shapely_geom = shape(geometry)
            
            def transform_coords(x, y, z=None):
                return transformer.transform(x, y)
            
            return transform(transform_coords, shapely_geom)
        
        except Exception as e:
            raise ValueError(f"Erreur transformation géométrique: {e}")
    
    def _calculate_metrics(self, geometry, crs: str) -> Dict[str, float]:
        """Calcule les métriques d'une géométrie"""
        metrics = {}
        
        # Superficie (seulement pour polygones)
        if geometry.geom_type in ['Polygon', 'MultiPolygon']:
            area_m2 = geometry.area
            metrics['superficie_m2'] = round(area_m2, 2)
            metrics['superficie_ha'] = round(area_m2 / 10000, 4)
            metrics['superficie_ares'] = round(area_m2 / 100, 2)
        
        # Périmètre
        metrics['perimetre_m'] = round(geometry.length, 2)
        
        # Centroïde
        centroid = geometry.centroid
        metrics['centroid_x'] = round(centroid.x, 6)
        metrics['centroid_y'] = round(centroid.y, 6)
        
        return metrics
    
    @contextmanager
    def _atomic_operation(self, parcel_id: str):
        """Opération atomique avec rollback automatique"""
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        temp_file = self.base_dir / f"{parcel_id}.tmp"
        backup_file = self.base_dir / f"{parcel_id}.backup"
        
        # Backup si le fichier existe
        if parcel_file.exists():
            shutil.copy2(parcel_file, backup_file)
        
        try:
            with self._file_lock():
                yield temp_file
                
                # Commit atomique
                if temp_file.exists():
                    temp_file.replace(parcel_file)
                
                # Nettoyage backup
                if backup_file.exists():
                    backup_file.unlink()
        
        except Exception:
            # Rollback
            if backup_file.exists():
                backup_file.replace(parcel_file)
            if temp_file.exists():
                temp_file.unlink()
            raise
        
        finally:
            # Nettoyage final
            if backup_file.exists():
                backup_file.unlink()
            if temp_file.exists():
                temp_file.unlink()
    
    def create_parcel(self, parcel_data: ParcelCreateModel) -> Dict[str, Any]:
        """Crée une parcelle avec transaction atomique"""
        parcel_id = str(uuid.uuid4())
        
        try:
            # Conversion géométrie si nécessaire
            if parcel_data.crs != self.default_crs:
                log.info(f"Conversion de {parcel_data.crs} vers {self.default_crs}")
                geometry = self._transform_geometry(
                    parcel_data.geometry,
                    parcel_data.crs,
                    self.default_crs
                )
            else:
                geometry = shape(parcel_data.geometry)
            
            # Validation et correction
            geometry = self._validate_and_fix_geometry(geometry)
            
            # Calcul des métriques
            metrics = self._calculate_metrics(geometry, self.default_crs)
            
            # Vérification superficie si fournie
            if parcel_data.superficie:
                calculated = metrics.get('superficie_m2', 0)
                if abs(calculated - parcel_data.superficie) > calculated * 0.1:
                    log.warning(
                        f"Différence >10% entre superficie fournie ({parcel_data.superficie}) "
                        f"et calculée ({calculated})"
                    )
            
            # Création GeoDataFrame
            gdf = gpd.GeoDataFrame([{
                'id': parcel_id,
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'proprietaire': parcel_data.proprietaire or '',
                'usage': parcel_data.usage or '',
                'crs': self.default_crs,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'geometry': geometry,
                **metrics
            }], crs=self.default_crs)
            
            # Sauvegarde atomique
            with self._atomic_operation(parcel_id) as temp_file:
                gdf.to_file(temp_file, driver='GeoJSON')
            
            # Mise à jour métadonnées
            metadata = self._load_metadata()
            metadata[parcel_id] = {
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'file': f"{parcel_id}.geojson"
            }
            self._save_metadata(metadata)
            
            # Invalidation cache
            cache.delete_memoized(self.get_parcel, parcel_id)
            cache.delete_memoized(self.list_parcels)
            
            log.info(f"Parcelle {parcel_id} créée avec succès")
            
            return {
                'id': parcel_id,
                'name': parcel_data.name,
                'commune': parcel_data.commune,
                'section': parcel_data.section,
                'numero': parcel_data.numero,
                'crs': self.default_crs,
                'centroid': {
                    'x': metrics['centroid_x'],
                    'y': metrics['centroid_y']
                },
                **{k: v for k, v in metrics.items() if k not in ['centroid_x', 'centroid_y']},
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            log.error(f"Erreur création parcelle: {e}", exc_info=True)
            raise
    
    @cache.memoize(timeout=300)
    def get_parcel(self, parcel_id: str, output_crs: str = None) -> Optional[Dict]:
        """Récupère une parcelle avec cache"""
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        
        if not parcel_file.exists():
            log.warning(f"Parcelle {parcel_id} non trouvée")
            return None
        
        try:
            gdf = gpd.read_file(parcel_file)
            
            if gdf.empty:
                log.error(f"Fichier parcelle {parcel_id} vide")
                return None
            
            # Conversion CRS si demandée
            if output_crs and output_crs != self.default_crs:
                gdf = gdf.to_crs(output_crs)
                parcel_data = gdf.iloc[0].to_dict()
                parcel_data['geometry'] = mapping(gdf.iloc[0].geometry)
                parcel_data['crs'] = output_crs
            else:
                parcel_data = gdf.iloc[0].to_dict()
                parcel_data['geometry'] = mapping(gdf.iloc[0].geometry)
                parcel_data['crs'] = self.default_crs
            
            return parcel_data
        
        except Exception as e:
            log.error(f"Erreur lecture parcelle {parcel_id}: {e}")
            return None
    
    @cache.memoize(timeout=300)
    def list_parcels(self, commune: str = None, page: int = 1, page_size: int = Config.DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
        """Liste les parcelles avec pagination"""
        page_size = min(page_size, Config.MAX_PAGE_SIZE)
        parcels = []
        
        for parcel_file in self.base_dir.glob("*.geojson"):
            try:
                gdf = gpd.read_file(parcel_file)
                if not gdf.empty:
                    parcel = gdf.iloc[0].to_dict()
                    
                    # Filtre par commune
                    if commune and parcel.get('commune', '').lower() != commune.lower():
                        continue
                    
                    # Suppression géométrie pour alléger
                    parcel.pop('geometry', None)
                    parcels.append(parcel)
            except Exception as e:
                log.error(f"Erreur lecture {parcel_file}: {e}")
                continue
        
        # Tri par date de création
        parcels.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Pagination
        total = len(parcels)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            'parcels': parcels[start:end],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            },
            'filter': {'commune': commune} if commune else None
        }
    
    def delete_parcel(self, parcel_id: str) -> bool:
        """Supprime une parcelle"""
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        
        if not parcel_file.exists():
            return False
        
        try:
            with self._file_lock():
                parcel_file.unlink()
                
                # Mise à jour métadonnées
                metadata = self._load_metadata()
                if parcel_id in metadata:
                    del metadata[parcel_id]
                    self._save_metadata(metadata)
            
            # Invalidation cache
            cache.delete_memoized(self.get_parcel, parcel_id)
            cache.delete_memoized(self.list_parcels)
            
            log.info(f"Parcelle {parcel_id} supprimée")
            return True
        
        except Exception as e:
            log.error(f"Erreur suppression parcelle {parcel_id}: {e}")
            return False
    
    def analyze_parcel(self, analysis_data: ParcelAnalysisModel) -> Dict[str, Any]:
        """Analyse une parcelle"""
        parcel = self.get_parcel(analysis_data.parcel_id, analysis_data.output_crs)
        
        if not parcel:
            raise ValueError(f"Parcelle {analysis_data.parcel_id} non trouvée")
        
        geometry = shape(parcel['geometry'])
        result = {
            'parcel_id': analysis_data.parcel_id,
            'analysis_type': analysis_data.analysis_type,
            'crs': parcel['crs'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if analysis_data.analysis_type == "superficie":
            result['data'] = self._analyze_area(geometry, parcel['crs'])
        elif analysis_data.analysis_type == "perimetre":
            result['data'] = self._analyze_perimeter(geometry)
        elif analysis_data.analysis_type == "buffer":
            result['data'] = self._analyze_buffer(geometry, analysis_data.parameters)
        elif analysis_data.analysis_type == "centroid":
            result['data'] = self._analyze_centroid(geometry)
        elif analysis_data.analysis_type == "intersection":
            result['data'] = self._analyze_intersection(geometry, analysis_data.parameters)
        
        return result
    
    def _analyze_area(self, geometry, crs: str) -> Dict[str, Any]:
        """Analyse la superficie"""
        if geometry.geom_type not in ['Polygon', 'MultiPolygon']:
            raise ValueError("Analyse superficie uniquement pour polygones")
        
        area_m2 = geometry.area
        
        return {
            'superficie_m2': round(area_m2, 2),
            'superficie_ha': round(area_m2 / 10000, 4),
            'superficie_ares': round(area_m2 / 100, 2),
            'superficie_km2': round(area_m2 / 1000000, 6),
            'precision': 'exacte' if crs == Config.DEFAULT_CRS else 'approximative'
        }
    
    def _analyze_perimeter(self, geometry) -> Dict[str, Any]:
        """Analyse le périmètre"""
        perimeter = geometry.length
        return {
            'perimetre_m': round(perimeter, 2),
            'perimetre_km': round(perimeter / 1000, 4)
        }
    
    def _analyze_buffer(self, geometry, params: Dict) -> Dict[str, Any]:
        """Crée un buffer"""
        distance = params.get('distance', 10)
        resolution = params.get('resolution', 16)
        
        buffer_geom = geometry.buffer(distance, resolution=resolution)
        
        result = {
            'buffer_geometry': mapping(buffer_geom),
            'distance_m': distance,
            'resolution': resolution
        }
        
        if buffer_geom.geom_type in ['Polygon', 'MultiPolygon']:
            result['superficie_buffer_m2'] = round(buffer_geom.area, 2)
            result['superficie_buffer_ha'] = round(buffer_geom.area / 10000, 4)
        
        return result
    
    def _analyze_centroid(self, geometry) -> Dict[str, Any]:
        """Calcule le centroïde"""
        centroid = geometry.centroid
        return {
            'x': round(centroid.x, 6),
            'y': round(centroid.y, 6),
            'geometry': mapping(centroid)
        }
    
    def _analyze_intersection(self, geometry, params: Dict) -> Dict[str, Any]:
        """Calcule l'intersection avec une autre géométrie"""
        other_geom = shape(params.get('geometry'))
        intersection = geometry.intersection(other_geom)
        
        result = {
            'has_intersection': not intersection.is_empty,
            'intersection_geometry': mapping(intersection) if not intersection.is_empty else None
        }
        
        if not intersection.is_empty and intersection.geom_type in ['Polygon', 'MultiPolygon']:
            result['intersection_area_m2'] = round(intersection.area, 2)
            result['intersection_area_ha'] = round(intersection.area / 10000, 4)
            result['percentage_overlap'] = round((intersection.area / geometry.area) * 100, 2)
        
        return result

# =============================================================================
# SERVICE DE TRANSFORMATION DE COORDONNÉES
# =============================================================================

class CoordinateService:
    """Service de transformation de coordonnées"""
    
    def __init__(self):
        self.common_crs = {
            '32630': {
                'name': 'EPSG:32630 - UTM zone 30N',
                'unit': 'mètres',
                'type': 'projected',
                'area': 'Afrique de l\'Ouest'
            },
            '4326': {
                'name': 'EPSG:4326 - WGS84',
                'unit': 'degrés',
                'type': 'geographic',
                'area': 'Monde'
            },
            '3857': {
                'name': 'EPSG:3857 - Web Mercator',
                'unit': 'mètres',
                'type': 'projected',
                'area': 'Web mapping'
            },
            '2154': {
                'name': 'EPSG:2154 - RGF93/Lambert-93',
                'unit': 'mètres',
                'type': 'projected',
                'area': 'France métropolitaine'
            }
        }
    
    @cache.memoize(timeout=3600)
    def transform_coordinates(self, transform_data: CoordinateTransformModel) -> Dict[str, Any]:
        """Transforme des coordonnées avec cache"""
        try:
            transformer = Transformer.from_crs(
                transform_data.from_crs,
                transform_data.to_crs,
                always_xy=True
            )
            
            transformed_coords = []
            errors = []
            
            for i, coord in enumerate(transform_data.coordinates):
                try:
                    x, y = coord[0], coord[1]
                    x_trans, y_trans = transformer.transform(x, y)
                    transformed_coords.append([round(x_trans, 6), round(y_trans, 6)])
                except Exception as e:
                    errors.append({'index': i, 'error': str(e)})
                    log.warning(f"Erreur transformation coordonnée {i}: {e}")
            
            return {
                'coordinates': transformed_coords,
                'from_crs': transform_data.from_crs,
                'to_crs': transform_data.to_crs,
                'count_success': len(transformed_coords),
                'count_input': len(transform_data.coordinates),
                'count_errors': len(errors),
                'errors': errors if errors else None
            }
        
        except Exception as e:
            raise ValueError(f"Erreur transformation: {str(e)}")
    
    @cache.memoize(timeout=3600)
    def get_crs_info(self, epsg_code: str) -> Dict[str, Any]:
        """Retourne les informations d'un CRS avec cache"""
        try:
            if epsg_code.startswith('EPSG:'):
                epsg_code = epsg_code.split(':')[1]
            
            crs = CRS.from_epsg(int(epsg_code))
            
            return {
                'epsg': f"EPSG:{epsg_code}",
                'name': crs.name,
                'type': 'projected' if crs.is_projected else 'geographic',
                'units': str(crs.axis_info[0].unit_name) if crs.axis_info else 'unknown',
                'area_of_use': {
                    'name': crs.area_of_use.name if crs.area_of_use else 'Unknown',
                    'bounds': crs.area_of_use.bounds if crs.area_of_use else None
                },
                'proj4': crs.to_proj4(),
                'wkt': crs.to_wkt()[:500] + '...' if len(crs.to_wkt()) > 500 else crs.to_wkt()
            }
        except Exception as e:
            raise ValueError(f"CRS non trouvé: EPSG:{epsg_code} - {str(e)}")
    
    def list_common_crs(self) -> Dict[str, Any]:
        """Liste les CRS couramment utilisés"""
        return {
            'default': Config.DEFAULT_CRS,
            'allowed': Config.ALLOWED_CRS,
            'common_crs': self.common_crs
        }

# =============================================================================
# INITIALISATION DES SERVICES
# =============================================================================

parcel_service = ParcelService(Config.BASE_DIR)
coord_service = CoordinateService()

# =============================================================================
# UTILITAIRES DE SÉCURITÉ
# =============================================================================

def sanitize_ogc_query(query_string: str) -> str:
    """Sanitise les query strings OGC"""
    import urllib.parse
    
    # Whitelist des paramètres OGC autorisés
    allowed_params = {
        'SERVICE', 'REQUEST', 'VERSION', 'LAYERS', 'STYLES', 'CRS', 'SRS',
        'BBOX', 'WIDTH', 'HEIGHT', 'FORMAT', 'TRANSPARENT', 'BGCOLOR',
        'EXCEPTIONS', 'TIME', 'ELEVATION', 'SLD', 'SLD_BODY',
        'TYPENAME', 'PROPERTYNAME', 'FEATUREID', 'FILTER', 'MAXFEATURES',
        'OUTPUTFORMAT', 'SORTBY', 'STARTINDEX', 'COUNT'
    }
    
    try:
        parsed = urllib.parse.parse_qs(query_string, keep_blank_values=True)
        sanitized = {}
        
        for key, values in parsed.items():
            key_upper = key.upper()
            if key_upper in allowed_params:
                # Validation basique des valeurs
                safe_values = []
                for value in values:
                    if len(value) <= 1000:  # Limite de taille
                        safe_values.append(value)
                if safe_values:
                    sanitized[key] = safe_values
        
        return urllib.parse.urlencode(sanitized, doseq=True)
    
    except Exception as e:
        log.error(f"Erreur sanitization query: {e}")
        raise ValueError("Query string invalide")

def validate_file_extension(filename: str) -> bool:
    """Valide l'extension d'un fichier"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS

# =============================================================================
# ROUTES API - AUTHENTIFICATION
# =============================================================================

@app.route('/api/auth/token', methods=['POST'])
@limiter.limit("10 per minute")
@handle_errors
def create_token():
    """Crée un token JWT (pour démo - à remplacer par auth réel)"""
    data = request.get_json()
    
    if not data or 'user_id' not in data:
        return jsonify({"error": "user_id requis"}), 400
    
    user_id = data.get('user_id')
    roles = data.get('roles', ['user'])
    
    token = auth_service.create_token(user_id, roles)
    
    return jsonify({
        'token': token,
        'type': 'Bearer',
        'expires_in': Config.JWT_EXPIRATION_HOURS * 3600,
        'user_id': user_id,
        'roles': roles
    }), 201

@app.route('/api/auth/logout', methods=['POST'])
@require_auth()
@handle_errors
def logout():
    """Révoque un token (blacklist)"""
    auth_header = request.headers.get('Authorization')
    token = auth_header.split()[1]
    
    auth_service.blacklist_token(token)
    
    return jsonify({
        'message': 'Token révoqué avec succès'
    }), 200

# =============================================================================
# ROUTES API - SANTÉ ET CONFIGURATION
# =============================================================================

@app.route('/api/health', methods=['GET'])
@limiter.limit("60 per minute")
def health():
    """Endpoint de santé détaillé"""
    redis_status = redis_manager.ping()
    qgis_project_exists = Path(Config.QGIS_PROJECT_FILE).exists()
    
    return jsonify({
        "status": "healthy" if redis_status and qgis_project_exists else "degraded",
        "version": Config.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crs_default": Config.DEFAULT_CRS,
        "services": {
            "qgis_project": qgis_project_exists,
            "redis": redis_status,
            "cache": cache.cache._write_client.ping() if hasattr(cache.cache, '_write_client') else False
        },
        "stats": {
            "parcels_count": len(list((Config.BASE_DIR / 'parcels').glob('*.geojson'))),
            "base_dir": str(Config.BASE_DIR)
        }
    })

@app.route('/api/config', methods=['GET'])
@require_auth(roles=['admin'])
@handle_errors
def get_config():
    """Configuration de l'API (admin uniquement)"""
    return jsonify({
        "version": Config.VERSION,
        "crs": {
            "default": Config.DEFAULT_CRS,
            "allowed": Config.ALLOWED_CRS
        },
        "limits": {
            "max_file_size_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024),
            "max_page_size": Config.MAX_PAGE_SIZE,
            "default_page_size": Config.DEFAULT_PAGE_SIZE
        },
        "categories": Config.CATEGORIES,
        "timeouts": {
            "qgis_server": Config.QGIS_SERVER_TIMEOUT,
            "file_operation": Config.FILE_OPERATION_TIMEOUT
        }
    })

# =============================================================================
# ROUTES API - CRS
# =============================================================================

@app.route('/api/crs/info', methods=['GET'])
@limiter.limit("60 per minute")
@handle_errors
def get_crs_info():
    """Informations sur un CRS"""
    epsg = request.args.get('epsg', '32630')
    info = coord_service.get_crs_info(epsg)
    return jsonify(info)

@app.route('/api/crs/list', methods=['GET'])
@limiter.limit("60 per minute")
def list_crs():
    """Liste les CRS couramment utilisés"""
    return jsonify(coord_service.list_common_crs())

@app.route('/api/crs/transform', methods=['POST'])
@limiter.limit("30 per minute")
@require_auth()
@handle_errors
def transform_coordinates():
    """Transforme des coordonnées entre CRS"""
    data = CoordinateTransformModel.parse_raw(request.data)
    result = coord_service.transform_coordinates(data)
    return jsonify(result)

# =============================================================================
# ROUTES API - PARCELLES
# =============================================================================

@app.route('/api/parcels', methods=['POST'])
@limiter.limit("20 per minute")
@require_auth()
@handle_errors
def create_parcel():
    """Crée une parcelle"""
    data = ParcelCreateModel.parse_raw(request.data)
    result = parcel_service.create_parcel(data)
    return jsonify(result), 201

@app.route('/api/parcels', methods=['GET'])
@limiter.limit("60 per minute")
@require_auth()
@handle_errors
def list_parcels():
    """Liste les parcelles avec pagination"""
    commune = request.args.get('commune')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', Config.DEFAULT_PAGE_SIZE))
    
    if page < 1:
        return jsonify({"error": "page doit être >= 1"}), 400
    if page_size < 1 or page_size > Config.MAX_PAGE_SIZE:
        return jsonify({"error": f"page_size doit être entre 1 et {Config.MAX_PAGE_SIZE}"}), 400
    
    result = parcel_service.list_parcels(commune, page, page_size)
    
    response = jsonify(result)
    response.headers['X-Total-Count'] = result['pagination']['total']
    response.headers['X-Page'] = page
    response.headers['X-Page-Size'] = page_size
    
    return response

@app.route('/api/parcels/<parcel_id>', methods=['GET'])
@limiter.limit("60 per minute")
@require_auth()
@handle_errors
def get_parcel(parcel_id):
    """Récupère une parcelle"""
    # Validation UUID
    try:
        uuid.UUID(parcel_id)
    except ValueError:
        return jsonify({"error": "ID parcelle invalide"}), 400
    
    output_crs = request.args.get('crs', Config.DEFAULT_CRS)
    
    if output_crs not in Config.ALLOWED_CRS:
        return jsonify({"error": f"CRS non autorisé: {output_crs}"}), 400
    
    parcel = parcel_service.get_parcel(parcel_id, output_crs)
    
    if not parcel:
        return jsonify({"error": "Parcelle non trouvée"}), 404
    
    return jsonify(parcel)

@app.route('/api/parcels/<parcel_id>', methods=['DELETE'])
@limiter.limit("20 per minute")
@require_auth(roles=['admin', 'editor'])
@handle_errors
def delete_parcel(parcel_id):
    """Supprime une parcelle"""
    try:
        uuid.UUID(parcel_id)
    except ValueError:
        return jsonify({"error": "ID parcelle invalide"}), 400
    
    success = parcel_service.delete_parcel(parcel_id)
    
    if not success:
        return jsonify({"error": "Parcelle non trouvée"}), 404
    
    return jsonify({
        "message": "Parcelle supprimée avec succès",
        "id": parcel_id
    }), 200

@app.route('/api/parcels/<parcel_id>/analyze', methods=['POST'])
@limiter.limit("30 per minute")
@require_auth()
@handle_errors
def analyze_parcel(parcel_id):
    """Analyse une parcelle"""
    try:
        uuid.UUID(parcel_id)
    except ValueError:
        return jsonify({"error": "ID parcelle invalide"}), 400
    
    data = ParcelAnalysisModel.parse_raw(request.data)
    data.parcel_id = parcel_id
    result = parcel_service.analyze_parcel(data)
    return jsonify(result)

# =============================================================================
# ROUTES API - SERVICES OGC
# =============================================================================

@app.route('/api/ogc/<service>', methods=['GET'])
@limiter.limit("100 per minute")
@handle_errors
def ogc_service(service):
    """Services OGC sécurisés via QGIS Server"""
    service_upper = service.upper()
    
    if service_upper not in ['WMS', 'WFS', 'WCS']:
        return jsonify({"error": f"Service non supporté: {service}"}), 400
    
    if not Path(Config.QGIS_SERVER_BIN).exists():
        return jsonify({"error": "QGIS Server non disponible"}), 503
    
    try:
        # Sanitization de la query string
        qs = sanitize_ogc_query(request.query_string.decode())
        
        # Forcer EPSG:32630 si non spécifié
        if 'CRS=' not in qs.upper() and 'SRS=' not in qs.upper():
            separator = '&' if qs else ''
            qs += f"{separator}CRS={Config.DEFAULT_CRS}"
        
        env = os.environ.copy()
        env.update({
            "QUERY_STRING": qs,
            "QGIS_PROJECT_FILE": str(Config.QGIS_PROJECT_FILE),
            "SERVICE": service_upper,
            "QT_QPA_PLATFORM": "offscreen",
            "REQUEST_METHOD": "GET",
            "SERVER_NAME": request.host,
            "HTTPS": "on" if request.is_secure else "off"
        })
        
        result = subprocess.run(
            [Config.QGIS_SERVER_BIN],
            env=env,
            capture_output=True,
            timeout=Config.QGIS_SERVER_TIMEOUT
        )
        
        if result.returncode != 0:
            log.error(f"Erreur QGIS Server: {result.stderr.decode()[:500]}")
            return jsonify({"error": "Erreur service OGC"}), 500
        
        # Détection du type de contenu
        content_type = "text/xml"
        if result.stdout.startswith(b"\x89PNG"):
            content_type = "image/png"
        elif result.stdout.startswith(b"%PDF"):
            content_type = "application/pdf"
        elif result.stdout.startswith(b"GIF"):
            content_type = "image/gif"
        elif result.stdout.startswith(b"\xff\xd8\xff"):
            content_type = "image/jpeg"
        elif b"<ServiceException" in result.stdout:
            content_type = "text/xml"
        
        response = Response(result.stdout, content_type=content_type)
        response.headers['X-Service'] = service_upper
        return response
    
    except subprocess.TimeoutExpired:
        log.error(f"Timeout QGIS Server pour service {service_upper}")
        return jsonify({"error": "Timeout du service OGC"}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(f"Erreur OGC service {service_upper}: {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du service OGC"}), 500

# =============================================================================
# ROUTES API - GESTION DE FICHIERS
# =============================================================================

@app.route('/api/files/upload', methods=['POST'])
@limiter.limit("10 per minute")
@require_auth()
@handle_errors
def upload_file():
    """Upload de fichiers géospatiaux sécurisé"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400
    
    file = request.files['file']
    category = request.form.get('category', 'other')
    
    if not file.filename:
        return jsonify({"error": "Nom de fichier invalide"}), 400
    
    if category not in Config.CATEGORIES:
        return jsonify({
            "error": f"Catégorie invalide",
            "allowed": Config.CATEGORIES
        }), 400
    
    # Validation extension
    if not validate_file_extension(file.filename):
        return jsonify({
            "error": "Extension de fichier non autorisée",
            "allowed": list(Config.ALLOWED_EXTENSIONS)
        }), 400
    
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    new_filename = f"{file_id}.{ext}"
    
    filepath = Config.BASE_DIR / category / new_filename
    
    try:
        # Sauvegarde sécurisée
        file.save(str(filepath))
        os.chmod(filepath, 0o644)
        
        file_info = {
            "id": file_id,
            "filename": filename,
            "category": category,
            "size_bytes": filepath.stat().st_size,
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_by": g.user_id
        }
        
        # Métadonnées géospatiales
        if ext in ['shp', 'geojson', 'gpkg', 'kml']:
            try:
                gdf = gpd.read_file(str(filepath))
                file_info['geo_info'] = {
                    'crs': str(gdf.crs) if gdf.crs else 'unknown',
                    'features': len(gdf),
                    'geometry_type': str(gdf.geometry.type.iloc[0]) if not gdf.empty else 'unknown',
                    'bounds': gdf.total_bounds.tolist() if not gdf.empty else None
                }
            except Exception as e:
                log.warning(f"Impossible de lire métadonnées géospatiales: {e}")
        
        log.info(f"Fichier uploadé: {filename} -> {new_filename} par {g.user_id}")
        return jsonify(file_info), 201
    
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        log.error(f"Erreur upload fichier: {e}")
        raise

@app.route('/api/files/<category>', methods=['GET'])
@limiter.limit("60 per minute")
@require_auth()
@handle_errors
def list_files(category):
    """Liste les fichiers d'une catégorie"""
    if category not in Config.CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    
    cat_dir = Config.BASE_DIR / category
    files = []
    
    for file_path in cat_dir.glob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(
                    file_path.stat().st_mtime,
                    tz=timezone.utc
                ).isoformat()
            })
    
    return jsonify({
        "category": category,
        "count": len(files),
        "files": sorted(files, key=lambda x: x['modified_at'], reverse=True)
    })

@app.route('/api/files/<category>/<filename>', methods=['GET'])
@limiter.limit("30 per minute")
@require_auth()
@handle_errors
def download_file(category, filename):
    """Télécharge un fichier de façon sécurisée"""
    if category not in Config.CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    
    safe_filename = secure_filename(filename)
    
    # Protection contre path traversal
    try:
        file_path = safe_join(str(Config.BASE_DIR / category), safe_filename)
    except Exception:
        return jsonify({"error": "Chemin fichier invalide"}), 400
    
    file_path = Path(file_path)
    
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "Fichier non trouvé"}), 404
    
    # Vérification que le fichier est bien dans le bon répertoire
    if not str(file_path).startswith(str(Config.BASE_DIR / category)):
        return jsonify({"error": "Accès refusé"}), 403
    
    return send_from_directory(
        Config.BASE_DIR / category,
        safe_filename,
        as_attachment=True
    )

@app.route('/api/files/<category>/<filename>', methods=['DELETE'])
@limiter.limit("20 per minute")
@require_auth(roles=['admin', 'editor'])
@handle_errors
def delete_file(category, filename):
    """Supprime un fichier"""
    if category not in Config.CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    
    safe_filename = secure_filename(filename)
    
    try:
        file_path = safe_join(str(Config.BASE_DIR / category), safe_filename)
    except Exception:
        return jsonify({"error": "Chemin fichier invalide"}), 400
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return jsonify({"error": "Fichier non trouvé"}), 404
    
    if not str(file_path).startswith(str(Config.BASE_DIR / category)):
        return jsonify({"error": "Accès refusé"}), 403
    
    try:
        file_path.unlink()
        log.info(f"Fichier supprimé: {safe_filename} par {g.user_id}")
        return jsonify({
            "message": "Fichier supprimé avec succès",
            "filename": safe_filename
        })
    except Exception as e:
        log.error(f"Erreur suppression fichier: {e}")
        raise

# =============================================================================
# ROUTES API - ADMINISTRATION
# =============================================================================

@app.route('/api/admin/stats', methods=['GET'])
@require_auth(roles=['admin'])
@handle_errors
def admin_stats():
    """Statistiques complètes de l'API"""
    stats = {
        "system": {
            "version": Config.VERSION,
            "qgis_project": Path(Config.QGIS_PROJECT_FILE).exists(),
            "redis": redis_manager.ping(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "parcels": {
            "total": len(list((Config.BASE_DIR / 'parcels').glob('*.geojson'))),
            "storage_mb": round(
                sum(f.stat().st_size for f in (Config.BASE_DIR / 'parcels').glob('*.geojson')) / (1024 * 1024),
                2
            )
        },
        "storage": {
            "base_dir": str(Config.BASE_DIR),
            "categories": {}
        }
    }
    
    # Stats par catégorie
    total_size = 0
    total_files = 0
    
    for category in Config.CATEGORIES:
        cat_dir = Config.BASE_DIR / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*"))
            file_count = sum(1 for f in files if f.is_file())
            size_bytes = sum(f.stat().st_size for f in files if f.is_file())
            
            stats["storage"]["categories"][category] = {
                "files": file_count,
                "storage_mb": round(size_bytes / (1024 * 1024), 2)
            }
            
            total_files += file_count
            total_size += size_bytes
    
    stats["storage"]["total_files"] = total_files
    stats["storage"]["total_mb"] = round(total_size / (1024 * 1024), 2)
    stats["storage"]["total_gb"] = round(total_size / (1024 * 1024 * 1024), 2)
    
    return jsonify(stats)

@app.route('/api/admin/cache/clear', methods=['POST'])
@require_auth(roles=['admin'])
@handle_errors
def clear_cache():
    """Nettoie le cache"""
    try:
        cache.clear()
        if redis_manager.client:
            redis_manager.flushdb()
        
        log.info(f"Cache nettoyé par {g.user_id}")
        
        return jsonify({
            "message": "Cache nettoyé avec succès",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        log.error(f"Erreur nettoyage cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/logs', methods=['GET'])
@require_auth(roles=['admin'])
@handle_errors
def get_logs():
    """Récupère les derniers logs (admin uniquement)"""
    lines = int(request.args.get('lines', 100))
    lines = min(lines, 1000)  # Maximum 1000 lignes
    
    log_file = Path('/var/log/qgis_api.log')
    
    if not log_file.exists():
        return jsonify({"logs": [], "message": "Fichier de logs non trouvé"})
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:]
        
        return jsonify({
            "logs": [line.strip() for line in recent_lines],
            "count": len(recent_lines),
            "total": len(all_lines)
        })
    except Exception as e:
        log.error(f"Erreur lecture logs: {e}")
        return jsonify({"error": str(e)}), 500

# =============================================================================
# ROUTES API - DOCUMENTATION
# =============================================================================

@app.route('/api/docs', methods=['GET'])
@limiter.limit("30 per minute")
def api_docs():
    """Documentation complète de l'API"""
    docs = {
        "version": Config.VERSION,
        "title": "API QGIS Server - Production Ready",
        "description": "API sécurisée pour la gestion de données géospatiales avec QGIS Server",
        "base_url": request.url_root.rstrip('/'),
        "default_crs": Config.DEFAULT_CRS,
        "allowed_crs": Config.ALLOWED_CRS,
        
        "authentication": {
            "type": "JWT Bearer Token",
            "header": "Authorization: Bearer <token>",
            "endpoints": {
                "create_token": {
                    "path": "/api/auth/token",
                    "method": "POST",
                    "description": "Crée un token JWT"
                },
                "logout": {
                    "path": "/api/auth/logout",
                    "method": "POST",
                    "description": "Révoque le token actuel"
                }
            }
        },
        
        "rate_limits": {
            "default": Config.RATELIMIT_DEFAULT,
            "headers": {
                "X-RateLimit-Limit": "Limite par fenêtre",
                "X-RateLimit-Remaining": "Requêtes restantes",
                "X-RateLimit-Reset": "Timestamp de reset"
            }
        },
        
        "endpoints": {
            "health": {
                "path": "/api/health",
                "method": "GET",
                "auth": False,
                "description": "État de santé de l'API"
            },
            "config": {
                "path": "/api/config",
                "method": "GET",
                "auth": True,
                "roles": ["admin"],
                "description": "Configuration de l'API"
            },
            
            "crs": {
                "info": {
                    "path": "/api/crs/info",
                    "method": "GET",
                    "auth": False,
                    "params": {"epsg": "Code EPSG (ex: 32630)"},
                    "description": "Informations sur un CRS"
                },
                "list": {
                    "path": "/api/crs/list",
                    "method": "GET",
                    "auth": False,
                    "description": "Liste des CRS couramment utilisés"
                },
                "transform": {
                    "path": "/api/crs/transform",
                    "method": "POST",
                    "auth": True,
                    "description": "Transforme des coordonnées entre CRS"
                }
            },
            
            "parcels": {
                "create": {
                    "path": "/api/parcels",
                    "method": "POST",
                    "auth": True,
                    "description": "Crée une nouvelle parcelle"
                },
                "list": {
                    "path": "/api/parcels",
                    "method": "GET",
                    "auth": True,
                    "params": {
                        "commune": "Filtre par commune (optionnel)",
                        "page": "Numéro de page (défaut: 1)",
                        "page_size": f"Taille de page (défaut: {Config.DEFAULT_PAGE_SIZE}, max: {Config.MAX_PAGE_SIZE})"
                    },
                    "description": "Liste toutes les parcelles"
                },
                "get": {
                    "path": "/api/parcels/{id}",
                    "method": "GET",
                    "auth": True,
                    "params": {"crs": "CRS de sortie (optionnel)"},
                    "description": "Récupère une parcelle"
                },
                "delete": {
                    "path": "/api/parcels/{id}",
                    "method": "DELETE",
                    "auth": True,
                    "roles": ["admin", "editor"],
                    "description": "Supprime une parcelle"
                },
                "analyze": {
                    "path": "/api/parcels/{id}/analyze",
                    "method": "POST",
                    "auth": True,
                    "description": "Analyse une parcelle (superficie, périmètre, buffer, centroid, intersection)"
                }
            },
            
            "files": {
                "upload": {
                    "path": "/api/files/upload",
                    "method": "POST",
                    "auth": True,
                    "description": "Upload un fichier géospatial"
                },
                "list": {
                    "path": "/api/files/{category}",
                    "method": "GET",
                    "auth": True,
                    "description": "Liste les fichiers d'une catégorie"
                },
                "download": {
                    "path": "/api/files/{category}/{filename}",
                    "method": "GET",
                    "auth": True,
                    "description": "Télécharge un fichier"
                },
                "delete": {
                    "path": "/api/files/{category}/{filename}",
                    "method": "DELETE",
                    "auth": True,
                    "roles": ["admin", "editor"],
                    "description": "Supprime un fichier"
                }
            },
            
            "ogc": {
                "path": "/api/ogc/{service}",
                "method": "GET",
                "auth": False,
                "services": ["WMS", "WFS", "WCS"],
                "description": "Services OGC via QGIS Server"
            },
            
            "admin": {
                "stats": {
                    "path": "/api/admin/stats",
                    "method": "GET",
                    "auth": True,
                    "roles": ["admin"],
                    "description": "Statistiques de l'API"
                },
                "cache_clear": {
                    "path": "/api/admin/cache/clear",
                    "method": "POST",
                    "auth": True,
                    "roles": ["admin"],
                    "description": "Nettoie le cache"
                },
                "logs": {
                    "path": "/api/admin/logs",
                    "method": "GET",
                    "auth": True,
                    "roles": ["admin"],
                    "params": {"lines": "Nombre de lignes (max: 1000)"},
                    "description": "Récupère les logs"
                }
            }
        },
        
        "categories": Config.CATEGORIES,
        "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
        
        "examples": {
            "create_token": {
                "method": "POST",
                "url": "/api/auth/token",
                "body": {
                    "user_id": "user123",
                    "roles": ["user"]
                }
            },
            "create_parcel": {
                "method": "POST",
                "url": "/api/parcels",
                "headers": {
                    "Authorization": "Bearer <token>",
                    "Content-Type": "application/json"
                },
                "body": {
                    "name": "Parcelle Test",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-1.5, 12.5], [-1.5, 12.6], [-1.4, 12.6], [-1.4, 12.5], [-1.5, 12.5]]]
                    },
                    "commune": "Bobo-Dioulasso",
                    "section": "A",
                    "numero": "123",
                    "crs": "EPSG:4326"
                }
            },
            "transform_coords": {
                "method": "POST",
                "url": "/api/crs/transform",
                "headers": {
                    "Authorization": "Bearer <token>",
                    "Content-Type": "application/json"
                },
                "body": {
                    "coordinates": [[-1.5, 12.5], [-1.4, 12.6]],
                    "from_crs": "EPSG:4326",
                    "to_crs": "EPSG:32630"
                }
            }
        }
    }
    
    return jsonify(docs)

# =============================================================================
# GESTIONNAIRES D'ERREURS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404"""
    return jsonify({
        "error": "Endpoint non trouvé",
        "path": request.path,
        "method": request.method,
        "documentation": f"{request.url_root}api/docs",
        "request_id": g.get('request_id', 'unknown')
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Gestion des erreurs 405"""
    return jsonify({
        "error": "Méthode HTTP non autorisée",
        "path": request.path,
        "method": request.method,
        "request_id": g.get('request_id', 'unknown')
    }), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    """Gestion des fichiers trop volumineux"""
    return jsonify({
        "error": "Fichier trop volumineux",
        "max_size_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024),
        "request_id": g.get('request_id', 'unknown')
    }), 413

@app.errorhandler(429)
def ratelimit_handler(error):
    """Gestion du rate limiting"""
    return jsonify({
        "error": "Trop de requêtes",
        "message": "Limite de taux dépassée. Veuillez réessayer plus tard.",
        "request_id": g.get('request_id', 'unknown')
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs serveur"""
    request_id = g.get('request_id', 'unknown')
    log.error(f"[{request_id}] Erreur serveur 500: {error}", exc_info=True)
    
    if Config.DEBUG:
        return jsonify({
            "error": "Erreur interne du serveur",
            "message": str(error),
            "request_id": request_id
        }), 500
    else:
        return jsonify({
            "error": "Erreur interne du serveur",
            "request_id": request_id,
            "message": "Veuillez contacter l'administrateur"
        }), 500

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def initialize_app():
    """Initialisation de l'application"""
    log.info("=" * 80)
    log.info(f"Démarrage de {Config.APP_NAME} v{Config.VERSION}")
    log.info("=" * 80)
    log.info(f"CRS par défaut: {Config.DEFAULT_CRS}")
    log.info(f"CRS autorisés: {', '.join(Config.ALLOWED_CRS)}")
    log.info(f"Répertoire données: {Config.BASE_DIR}")
    log.info(f"Projet QGIS: {Config.QGIS_PROJECT_FILE}")
    log.info(f"Redis: {'Connecté' if redis_manager.ping() else 'Non disponible (mode dégradé)'}")
    log.info(f"Cache: {'Activé' if not Config.DEBUG else 'Désactivé (mode debug)'}")
    log.info(f"Rate Limiting: {'Activé' if not Config.DEBUG else 'Désactivé (mode debug)'}")
    log.info(f"Documentation: {request.url_root if request else 'http://localhost:10000/'}api/docs")
    log.info("=" * 80)
    
    # Vérifications
    if not Path(Config.QGIS_PROJECT_FILE).exists():
        log.warning(f"Projet QGIS non trouvé: {Config.QGIS_PROJECT_FILE}")
    
    if not Path(Config.QGIS_SERVER_BIN).exists():
        log.warning(f"QGIS Server non trouvé: {Config.QGIS_SERVER_BIN}")
    
    log.info("Initialisation terminée - Serveur prêt")

if __name__ == '__main__':
    initialize_app()
    
    # Configuration serveur
    port = int(os.getenv('PORT', 10000))
    host = os.getenv('HOST', '0.0.0.0')
    workers = int(os.getenv('WORKERS', 4))
    
    if Config.DEBUG:
        # Mode développement
        log.warning("MODE DEBUG ACTIVÉ - NE PAS UTILISER EN PRODUCTION")
        app.run(
            host=host,
            port=port,
            debug=True,
            threaded=True
        )
    else:
        # Mode production avec Gunicorn
        log.info(f"Démarrage en mode production sur {host}:{port}")
        log.info("Utilisez Gunicorn pour le déploiement:")
        log.info(f"  gunicorn -w {workers} -b {host}:{port} --timeout 120 --access-logfile - --error-logfile - api_production:app")
        
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )