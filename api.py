# api.py - Version optimisée pour Render avec QGIS Server + Auth JWT
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
from collections import defaultdict

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
from passlib.context import CryptContext
import jwt

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

# Configuration JWT
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_USE_ENV_VAR")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# Hash des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
            redis_client = redis.Redis.from_url(redis_url, decode_responses=False, socket_timeout=3)
        else:
            redis_client = redis.Redis(
                host=redis_url, 
                port=int(os.getenv("REDIS_PORT", 6379)), 
                db=0,
                decode_responses=False, 
                socket_timeout=3
            )
        redis_client.ping()
        log.info("Redis connecté")
except Exception as e:
    log.warning(f"Redis indisponible, utilisation cache fichier: {e}")
    redis_client = None

def cache_get(key: str) -> Optional[bytes]:
    """Récupération cache avec fallback - retourne bytes"""
    try:
        if redis_client:
            try:
                value = redis_client.get(key)
                if value is not None:
                    return value
            except Exception as e:
                log.warning(f"Erreur cache Redis: {e}")
        
        # Fallback sur cache fichier
        with cache_lock:
            return file_cache.get(key)
            
    except Exception as e:
        log.error(f"Erreur générale cache_get: {e}")
        return None

def cache_set(key: str, value: bytes, expire: int = 3600):
    """Enregistrement cache avec fallback - accepte bytes"""
    try:
        if not isinstance(value, bytes):
            try:
                value = json.dumps(value).encode('utf-8')
            except Exception as e:
                log.error(f"Erreur sérialisation cache: {e}")
                return
        
        if redis_client:
            try:
                redis_client.setex(key, expire, value)
                return
            except Exception as e:
                log.warning(f"Erreur cache_set Redis: {e}")
        
        # Fallback sur cache mémoire
        with cache_lock:
            file_cache[key] = value
            
    except Exception as e:
        log.error(f"Erreur générale cache_set: {e}")

# ================================================================
# Rate limiting
# ================================================================
rate_limit_storage = defaultdict(list)
rate_limit_lock = Lock()

def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Décorateur de rate limiting simple"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            client_id = request.remote_addr
            auth_header = request.headers.get('Authorization')
            if auth_header:
                try:
                    parts = auth_header.split()
                    if len(parts) == 2:
                        payload = decode_token(parts[1])
                        client_id = payload.get('sub', client_id)
                except:
                    pass
            
            now = time.time()
            window_start = now - window_seconds
            
            with rate_limit_lock:
                rate_limit_storage[client_id] = [
                    req_time for req_time in rate_limit_storage[client_id]
                    if req_time > window_start
                ]
                
                if len(rate_limit_storage[client_id]) >= max_requests:
                    return jsonify({
                        "error": "Trop de requêtes",
                        "retry_after": window_seconds
                    }), 429
                
                rate_limit_storage[client_id].append(now)
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

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
        self._init_success = False
    
    def initialize(self):
        with self._lock:
            if self._initialized:
                return self._init_success, "Déjà initialisé"
            
            log.info("Initialisation QGIS...")
            try:
                self._setup_qgis_environment()
                
                # Import Qt et QGIS
                from PyQt5.QtCore import QSize, QByteArray, QBuffer, QIODevice
                from PyQt5.QtGui import QColor, QFont, QImage, QPainter
                
                # Import QGIS components
                from qgis.core import (
                    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
                    QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle,
                    QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel,
                    QgsLayoutExporter, QgsCoordinateReferenceSystem,
                    QgsVectorFileWriter, QgsFeature, QgsGeometry, QgsPointXY,
                    QgsLayoutPoint, QgsLayoutSize, QgsUnitTypes,
                    QgsCoordinateTransform, QgsLayoutItemLegend, QgsLayoutItemScaleBar
                )
                
                if not QgsApplication.instance():
                    self.qgs_app = QgsApplication([], False)
                    self.qgs_app.initQgis()
                    log.info("QGIS Application créée")
                else:
                    self.qgs_app = QgsApplication.instance()
                    log.info("QGIS Application existante réutilisée")
                
                # Stocker les classes
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
                self._init_success = True
                log.info("QGIS initialisé avec succès")
                return True, None
                
            except Exception as e:
                error_msg = f"Erreur initialisation QGIS: {str(e)}"
                log.error(error_msg, exc_info=True)
                self._init_success = False
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
        return self._initialized and self._init_success
    
    def get_classes(self):
        if not self.is_initialized():
            raise RuntimeError("QGIS non initialisé ou initialisation échouée")
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
                sess = project_sessions[sid]
                if sess.project:
                    sess.project.clear()
                del project_sessions[sid]
                log.info(f"Session expirée: {sid}")

# ================================================================
# Modèles Pydantic
# ================================================================
class UserRegisterModel(BaseModel):
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Mot de passe doit contenir une majuscule')
        if not any(c.islower() for c in v):
            raise ValueError('Mot de passe doit contenir une minuscule')
        if not any(c.isdigit() for c in v):
            raise ValueError('Mot de passe doit contenir un chiffre')
        return v

class UserLoginModel(BaseModel):
    email: str
    password: str

class TokenRefreshModel(BaseModel):
    refresh_token: str

class PasswordChangeModel(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=8)

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
        if v is not None:
            if not isinstance(v, dict) or 'type' not in v or 'coordinates' not in v:
                raise ValueError('Géométrie GeoJSON invalide')
        return v

class BoundsModel(BaseModel):
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

class ProjectModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    project_file_path: str = Field(..., min_length=1)

class SearchModel(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BulkParcelCreateModel(BaseModel):
    parcels: List[ParcelCreateModel]

class BulkParcelUpdateModel(BaseModel):
    updates: List[Dict[str, Any]]

class BulkParcelDeleteModel(BaseModel):
    ids: List[str]

class AnalyticsDataModel(BaseModel):
    event_type: str
    details: Dict[str, Any]
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
# Fonctions JWT
# ================================================================
def create_access_token(user_id: str, user_email: str, is_admin: bool = False) -> str:
    """Crée un token d'accès JWT"""
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        'sub': user_id,
        'email': user_email,
        'is_admin': is_admin,
        'type': 'access',
        'exp': expire,
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    """Crée un token de rafraîchissement"""
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        'sub': user_id,
        'type': 'refresh',
        'exp': expire,
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Dict[str, Any]:
    """Décode et valide un token JWT"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expiré")
    except jwt.InvalidTokenError:
        raise ValueError("Token invalide")

def get_current_user_from_token(token: str) -> Dict[str, Any]:
    """Récupère l'utilisateur depuis un token"""
    payload = decode_token(token)
    
    if payload.get('type') != 'access':
        raise ValueError("Type de token invalide")
    
    user_id = payload.get('sub')
    if not user_id:
        raise ValueError("Token invalide")
    
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise ValueError("Utilisateur introuvable")
    
    if not user.get('is_active', True):
        raise ValueError("Compte désactivé")
    
    return user

# ================================================================
# Décorateurs
# ================================================================
def handle_errors(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            response = f(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log des performances
            if duration > 1.0:  # Log uniquement les requêtes lentes
                log.warning(f"Requête lente: {f.__name__} - {duration:.2f}s")
            
            return response
            
        except ValidationError as e:
            log.warning(f"Erreur validation: {e.errors()}")
            return jsonify({"error": "Données invalides", "details": e.errors()}), 400
        except ValueError as e:
            log.warning(f"Erreur valeur: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            log.warning(f"Fichier non trouvé: {str(e)}")
            return jsonify({"error": "Ressource introuvable"}), 404
        except PermissionError as e:
            log.error(f"Erreur permission: {str(e)}")
            return jsonify({"error": "Accès non autorisé"}), 403
        except Exception as e:
            log.error(f"Erreur inattendue dans {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({"error": "Erreur serveur interne"}), 500
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

def require_auth(f):
    """Décorateur pour routes nécessitant authentification"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "Token manquant"}), 401
        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({"error": "Format Authorization invalide"}), 401
        
        token = parts[1]
        
        try:
            user = get_current_user_from_token(token)
            return f(*args, current_user=user, **kwargs)
        except ValueError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            log.error(f"Erreur authentification: {e}")
            return jsonify({"error": "Erreur authentification"}), 401
    
    return wrapper

def require_admin(f):
    """Décorateur pour routes admin uniquement"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "Token manquant"}), 401
        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({"error": "Format Authorization invalide"}), 401
        
        token = parts[1]
        
        try:
            user = get_current_user_from_token(token)
            
            if not user.get('is_admin', False):
                return jsonify({"error": "Accès admin requis"}), 403
            
            return f(*args, current_user=user, **kwargs)
        except ValueError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            log.error(f"Erreur authentification admin: {e}")
            return jsonify({"error": "Erreur authentification"}), 401
    
    return wrapper

def optional_auth(f):
    """Décorateur pour routes avec auth optionnelle"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        current_user = None
        
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                try:
                    current_user = get_current_user_from_token(parts[1])
                except:
                    pass
        
        return f(*args, current_user=current_user, **kwargs)
    
    return wrapper

# ================================================================
# Services
# ================================================================
class UserService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.users_file = self.base_dir / "users.json"
        self._ensure_file()
    
    def _ensure_file(self):
        """Crée le fichier users.json s'il n'existe pas"""
        if not self.users_file.exists():
            try:
                with open(self.users_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                log.info(f"Fichier utilisateurs créé: {self.users_file}")
            except Exception as e:
                log.error(f"Erreur création fichier utilisateurs: {e}")
                raise
    
    def _read_users(self) -> List[Dict[str, Any]]:
        """Lecture des utilisateurs avec gestion d'erreur"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            log.error(f"Erreur lecture utilisateurs: {e}")
            # Recréer le fichier si corrompu
            self._ensure_file()
            return []
    
    def _write_users(self, users: List[Dict[str, Any]]):
        """Écriture des utilisateurs avec gestion d'erreur"""
        try:
            # Sauvegarde temporaire
            temp_file = self.users_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2, default=str)
            
            # Remplacer le fichier original
            temp_file.replace(self.users_file)
            
        except Exception as e:
            log.error(f"Erreur écriture utilisateurs: {e}")
            raise
    
    def _hash_password(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_user(self, user_data: UserRegisterModel) -> Dict[str, Any]:
        users = self._read_users()
        
        if any(u['email'] == user_data.email for u in users):
            raise ValueError("Email déjà utilisé")
        
        if any(u['username'] == user_data.username for u in users):
            raise ValueError("Nom d'utilisateur déjà utilisé")
        
        user_id = str(uuid.uuid4())
        user = {
            'id': user_id,
            'email': user_data.email,
            'username': user_data.username,
            'full_name': user_data.full_name,
            'password_hash': self._hash_password(user_data.password),
            'is_active': True,
            'is_admin': False,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_login': None
        }
        
        users.append(user)
        self._write_users(users)
        log.info(f"Utilisateur créé: {user_data.email}")
        
        user_safe = {k: v for k, v in user.items() if k != 'password_hash'}
        return user_safe
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        users = self._read_users()
        user = next((u for u in users if u['email'] == email), None)
        
        if not user:
            return None
        
        if not user.get('is_active', True):
            raise ValueError("Compte désactivé")
        
        if not self._verify_password(password, user['password_hash']):
            return None
        
        user['last_login'] = datetime.now(timezone.utc).isoformat()
        self._write_users(users)
        
        user_safe = {k: v for k, v in user.items() if k != 'password_hash'}
        return user_safe
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        users = self._read_users()
        user = next((u for u in users if u['id'] == user_id), None)
        if user:
            return {k: v for k, v in user.items() if k != 'password_hash'}
        return None
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        users = self._read_users()
        for i, user in enumerate(users):
            if user['id'] == user_id:
                if not self._verify_password(old_password, user['password_hash']):
                    raise ValueError("Ancien mot de passe incorrect")
                users[i]['password_hash'] = self._hash_password(new_password)
                self._write_users(users)
                log.info(f"Mot de passe changé pour: {user['email']}")
                return True
        return False

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
        log.info(f"Projet créé: {project_id}")
        return project_dict
    
    def update_project(self, project_id: str, project_data: ProjectModel) -> Optional[Dict[str, Any]]:
        projects = self.get_projects()
        for i, proj in enumerate(projects):
            if proj['id'] == project_id:
                update_data = project_data.dict(exclude_unset=True)
                projects[i].update(update_data)
                with open(self.projects_file, 'w') as f:
                    json.dump(projects, f, indent=2)
                log.info(f"Projet mis à jour: {project_id}")
                return projects[i]
        return None
    
    def delete_project(self, project_id: str) -> bool:
        projects = self.get_projects()
        for i, proj in enumerate(projects):
            if proj['id'] == project_id:
                del projects[i]
                with open(self.projects_file, 'w') as f:
                    json.dump(projects, f, indent=2)
                log.info(f"Projet supprimé: {project_id}")
                return True
        return False

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
    
    def log_event(self, event_data: AnalyticsDataModel):
        events = self._read_events()
        events.append(event_data.dict())
        self._write_events(events)
        log.info(f"Événement loggué: {event_data.event_type}")
    
    def get_summary(self):
        events = self._read_events()
        summary = {}
        for event in events:
            et = event['event_type']
            summary[et] = summary.get(et, 0) + 1
        
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
        log.info(f"Comportement utilisateur loggué: {behavior_data.action}")
    
    def _read_events(self) -> List[Dict[str, Any]]:
        with open(self.events_file, 'r') as f:
            return json.load(f)
    
    def _write_events(self, data: List[Dict[str, Any]]):
        with open(self.events_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _read_behavior(self) -> List[Dict[str, Any]]:
        with open(self.behavior_file, 'r') as f:
            return json.load(f)
    
    def _write_behavior(self, data: List[Dict[str, Any]]):
        with open(self.behavior_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

class PerformanceService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.metrics_file = self.base_dir / "performance_metrics.json"
        self._ensure_file()
    
    def _ensure_file(self):
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                json.dump([], f)
    
    def log_metric(self, metric_data: PerformanceMetricsModel):
        metrics = self._read_metrics()
        metrics.append(metric_data.dict())
        self._write_metrics(metrics)
        log.info(f"Métrique logguée: {metric_data.metric_name}")
    
    def get_metrics(self, limit: Optional[int] = 100):
        metrics = self._read_metrics()
        sorted_metrics = sorted(metrics, key=lambda x: x['timestamp'], reverse=True)
        return sorted_metrics[:limit]
    
    def _read_metrics(self) -> List[Dict[str, Any]]:
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def _write_metrics(self, data: List[Dict[str, Any]]):
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

class ParcelService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / "parcels"
        self.base_dir.mkdir(exist_ok=True)
        self.default_crs = DEFAULT_CRS
        self.all_parcels_file = self.base_dir / "all_parcels.geojson"
        self._ensure_aggregate()
    
    def _ensure_aggregate(self):
        """Crée le fichier agrégé des parcelles - version robuste"""
        if not self.all_parcels_file.exists():
            try:
                # Créer un GeoDataFrame avec une structure vide mais valide
                empty_gdf = gpd.GeoDataFrame({
                    'id': pd.Series(dtype='str'),
                    'name': pd.Series(dtype='str'),
                    'commune': pd.Series(dtype='str'),
                    'section': pd.Series(dtype='str'),
                    'numero': pd.Series(dtype='str'),
                    'superficie_m2': pd.Series(dtype='float'),
                    'superficie_ha': pd.Series(dtype='float'),
                    'perimetre_m': pd.Series(dtype='float'),
                    'proprietaire': pd.Series(dtype='str'),
                    'usage': pd.Series(dtype='str'),
                    'centroid_x': pd.Series(dtype='float'),
                    'centroid_y': pd.Series(dtype='float'),
                    'created_at': pd.Series(dtype='str'),
                    'geometry': []
                }, crs=self.default_crs)
                
                # Sauvegarder avec une géométrie vide
                empty_gdf.to_file(self.all_parcels_file, driver="GeoJSON")
                log.info(f"Fichier agrégé créé: {self.all_parcels_file}")
                
            except Exception as e:
                log.error(f"Erreur création fichier agrégé: {e}")
                # Créer un fichier GeoJSON vide manuellement
                empty_geojson = {
                    "type": "FeatureCollection",
                    "features": [],
                    "crs": {
                        "type": "name",
                        "properties": {
                            "name": self.default_crs
                        }
                    }
                }
                with open(self.all_parcels_file, 'w') as f:
                    json.dump(empty_geojson, f)
                log.info(f"Fichier agrégé créé (méthode alternative): {self.all_parcels_file}")

    def create_parcel(self, parcel_data: ParcelCreateModel) -> Dict[str, Any]:
        parcel_id = str(uuid.uuid4())
        try:
            if parcel_data.crs != self.default_crs:
                geometry = self._transform_geometry(
                    parcel_data.geometry, parcel_data.crs, self.default_crs
                )
            else:
                geometry = shape(parcel_data.geometry)
            
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            
            superficie_m2 = round(geometry.area, 2)
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
            
            self._update_aggregate()
            
            log.info(f"Parcelle créée: {parcel_id}")
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
            log.error(f"Erreur création parcelle: {e}")
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
            for key, value in update_dict.items():
                if key in gdf.columns:
                    gdf.loc[0, key] = value
            
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
            
            gdf.to_file(parcel_file, driver='GeoJSON')
            self._update_aggregate()
            log.info(f"Parcelle mise à jour: {parcel_id}")
            
            updated_data = gdf.iloc[0].to_dict()
            updated_data['geometry'] = mapping(gdf.geometry.iloc[0])
            return updated_data
        except Exception as e:
            log.error(f"Erreur mise à jour parcelle: {e}")
            return None
    
    def _transform_geometry(self, geometry: Dict, from_crs: str, to_crs: str):
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        shapely_geom = shape(geometry)
        return transform(lambda x, y, z=None: transformer.transform(x, y), shapely_geom)
    
    def _update_aggregate(self):
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
                log.info(f"Agrégat mis à jour: {len(gdfs)} parcelles")
        except Exception as e:
            log.error(f"Erreur mise à jour agrégat: {e}")
    
    def get_parcels_by_bounds(self, bounds: BoundsModel) -> Dict[str, Any]:
        try:
            if bounds.crs != self.default_crs:
                transformer = Transformer.from_crs(bounds.crs, self.default_crs, always_xy=True)
                minx, miny = transformer.transform(bounds.minx, bounds.miny)
                maxx, maxy = transformer.transform(bounds.maxx, bounds.maxy)
            else:
                minx, miny, maxx, maxy = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
            
            if bounds.buffer_m:
                minx -= bounds.buffer_m
                miny -= bounds.buffer_m
                maxx += bounds.buffer_m
                maxy += bounds.buffer_m
            
            bbox = (minx, miny, maxx, maxy)
            gdf = gpd.read_file(self.all_parcels_file, bbox=bbox)
            
            if gdf.empty:
                return {'count': 0, 'data': [], 'bbox': list(bbox)}
            
            features = json.loads(gdf.to_json())['features']
            return {
                'count': len(features),
                'data': features,
                'bbox': list(bbox),
                'crs': self.default_crs
            }
        except Exception as e:
            log.error(f"Erreur récupération par emprise: {e}")
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
            log.error(f"Erreur lecture parcelle: {e}")
            return None
    
    def delete_parcel(self, parcel_id: str) -> bool:
        parcel_file = self.base_dir / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            return False
        try:
            parcel_file.unlink()
            self._update_aggregate()
            log.info(f"Parcelle supprimée: {parcel_id}")
            return True
        except Exception as e:
            log.error(f"Erreur suppression: {e}")
            return False

class CoordinateService:
    def __init__(self):
        self._transformers_cache = {}
    
    def transform_coordinates(self, coord_data: CoordinateTransformModel) -> Dict[str, Any]:
        cache_key = f"{coord_data.from_crs}:{coord_data.to_crs}"
        if cache_key not in self._transformers_cache:
            self._transformers_cache[cache_key] = Transformer.from_crs(
                coord_data.from_crs, coord_data.to_crs, always_xy=True
            )
        transformer = self._transformers_cache[cache_key]
        transformed = [
            [round(x, 6), round(y, 6)]
            for x, y in [transformer.transform(c[0], c[1]) for c in coord_data.coordinates]
        ]
        return {
            'coordinates': transformed,
            'from_crs': coord_data.from_crs,
            'to_crs': coord_data.to_crs,
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

# Initialisation services
user_service = UserService(BASE_DIR)
parcel_service = ParcelService(BASE_DIR)
coord_service = CoordinateService()
analytics_service = AnalyticsService(BASE_DIR)
performance_service = PerformanceService(BASE_DIR)
project_service = ProjectService(PROJECTS_DIR)

# ================================================================
# Routes Authentification
# ================================================================
@app.route('/api/auth/register', methods=['POST'])
@handle_errors
def register():
    try:
        data = UserRegisterModel.parse_raw(request.data)
        user = user_service.create_user(data)
        
        access_token = create_access_token(
            user['id'], 
            user['email'], 
            user.get('is_admin', False)
        )
        refresh_token = create_refresh_token(user['id'])
        
        return jsonify({
            "message": "Utilisateur créé avec succès",
            "user": user,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/auth/login', methods=['POST'])
@rate_limit(max_requests=5, window_seconds=300)
@handle_errors
def login():
    try:
        data = UserLoginModel.parse_raw(request.data)
        user = user_service.authenticate_user(data.email, data.password)
        
        if not user:
            return jsonify({"error": "Email ou mot de passe incorrect"}), 401
        
        access_token = create_access_token(
            user['id'], 
            user['email'], 
            user.get('is_admin', False)
        )
        refresh_token = create_refresh_token(user['id'])
        
        return jsonify({
            "message": "Connexion réussie",
            "user": user,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

@app.route('/api/auth/refresh', methods=['POST'])
@handle_errors
def refresh():
    try:
        data = TokenRefreshModel.parse_raw(request.data)
        payload = decode_token(data.refresh_token)
        
        if payload.get('type') != 'refresh':
            return jsonify({"error": "Token de rafraîchissement invalide"}), 401
        
        user_id = payload.get('sub')
        user = user_service.get_user_by_id(user_id)
        
        if not user:
            return jsonify({"error": "Utilisateur introuvable"}), 401
        
        access_token = create_access_token(
            user['id'], 
            user['email'], 
            user.get('is_admin', False)
        )
        
        return jsonify({
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
@handle_errors
def logout(current_user):
    log.info(f"Déconnexion: {current_user['email']}")
    return jsonify({"message": "Déconnexion réussie"})

@app.route('/api/auth/me', methods=['GET'])
@require_auth
@handle_errors
def get_current_user_route(current_user):
    return jsonify({"user": current_user})

@app.route('/api/auth/change-password', methods=['POST'])
@require_auth
@handle_errors
def change_password_route(current_user):
    try:
        data = PasswordChangeModel.parse_raw(request.data)
        success = user_service.change_password(
            current_user['id'],
            data.old_password,
            data.new_password
        )
        
        if success:
            return jsonify({"message": "Mot de passe modifié"})
        return jsonify({"error": "Erreur modification"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# ================================================================
# Routes Projets
# ================================================================
@app.route('/api/projects', methods=['GET'])
@optional_auth
@handle_errors
def list_projects(current_user):
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', type=int)
    projects = project_service.get_projects(limit=limit, offset=offset)
    return jsonify({"data": projects})

@app.route('/api/projects', methods=['POST'])
@require_auth
@handle_errors
def create_project_route(current_user):
    data = ProjectModel.parse_raw(request.data)
    result = project_service.create_project(data)
    return jsonify(result), 201

@app.route('/api/projects/<project_id>', methods=['GET'])
@optional_auth
@handle_errors
def get_project(project_id, current_user):
    project = project_service.get_project(project_id)
    if not project:
        return jsonify({"error": "Projet introuvable"}), 404
    return jsonify(project)

@app.route('/api/projects/<project_id>', methods=['PUT'])
@require_auth
@handle_errors
def update_project_route(project_id, current_user):
    data = ProjectModel.parse_raw(request.data)
    project = project_service.update_project(project_id, data)
    if not project:
        return jsonify({"error": "Projet introuvable"}), 404
    return jsonify(project)

@app.route('/api/projects/<project_id>', methods=['DELETE'])
@require_auth
@handle_errors
def delete_project_route(project_id, current_user):
    success = project_service.delete_project(project_id)
    if not success:
        return jsonify({"error": "Projet introuvable"}), 404
    return jsonify({"message": "Projet supprimé", "id": project_id})

# ================================================================
# Routes Parcelles
# ================================================================
@app.route('/api/debug/parcels', methods=['GET'])
@handle_errors
def debug_parcels():
    """Endpoint de diagnostic des parcelles"""
    parcels_dir = BASE_DIR / "parcels"
    
    # Compter les fichiers
    all_files = list(parcels_dir.glob("*.geojson"))
    individual_files = list(parcels_dir.glob("[!all_]*.geojson"))
    
    # Analyser quelques fichiers
    sample_analysis = []
    for file_path in individual_files[:3]:  # Premier 3 fichiers
        try:
            file_info = {
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "valid": False
            }
            if file_path.stat().st_size > 50:
                gdf = gpd.read_file(file_path)
                file_info["valid"] = not gdf.empty
                if not gdf.empty:
                    file_info["columns"] = list(gdf.columns)
                    file_info["crs"] = str(gdf.crs)
            sample_analysis.append(file_info)
        except Exception as e:
            file_info["error"] = str(e)
            sample_analysis.append(file_info)
    
    # Vérifier l'agrégat
    agg_file = parcels_dir / "all_parcels.geojson"
    agg_info = {
        "exists": agg_file.exists(),
        "size": agg_file.stat().st_size if agg_file.exists() else 0
    }
    if agg_file.exists() and agg_file.stat().st_size > 100:
        try:
            gdf = gpd.read_file(agg_file)
            agg_info["parcel_count"] = len(gdf)
            agg_info["columns"] = list(gdf.columns) if not gdf.empty else []
        except Exception as e:
            agg_info["error"] = str(e)
    
    return jsonify({
        "files": {
            "total": len(all_files),
            "individual": len(individual_files),
            "aggregate": 1 if agg_file.exists() else 0
        },
        "aggregate_file": agg_info,
        "sample_files": sample_analysis
    })

@app.route('/api/parcels', methods=['GET'])
@optional_auth
@handle_errors
def list_parcels(current_user):
    limit = request.args.get('limit', default=100, type=int)
    offset = request.args.get('offset', default=0, type=int)
    
    try:
        # Méthode 1: Essayer d'abord avec le fichier agrégé (plus fiable)
        all_parcels_file = BASE_DIR / "parcels" / "all_parcels.geojson"
        if all_parcels_file.exists() and all_parcels_file.stat().st_size > 100:
            log.info("Utilisation du fichier agrégé pour le listing")
            gdf = gpd.read_file(all_parcels_file)
            
            if not gdf.empty:
                # Appliquer la pagination
                start_idx = offset
                end_idx = start_idx + limit
                paginated_gdf = gdf.iloc[start_idx:end_idx]
                
                # Convertir en format de réponse
                features = []
                for idx, row in paginated_gdf.iterrows():
                    feature = {
                        'id': row.get('id', f"unknown_{idx}"),
                        'name': row.get('name', ''),
                        'commune': row.get('commune', ''),
                        'section': row.get('section', ''),
                        'numero': row.get('numero', ''),
                        'superficie_m2': row.get('superficie_m2'),
                        'superficie_ha': row.get('superficie_ha'),
                        'proprietaire': row.get('proprietaire', ''),
                        'usage': row.get('usage', ''),
                        'geometry': mapping(row.geometry) if hasattr(row, 'geometry') else None
                    }
                    features.append(feature)
                
                return jsonify({
                    "data": features,
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "total": len(gdf)
                    }
                })
        
        # Méthode 2: Fallback - lire les fichiers individuels
        log.info("Fallback sur les fichiers individuels")
        parcels_dir = BASE_DIR / "parcels"
        all_parcel_files = list(parcels_dir.glob("[!all_]*.geojson"))
        
        # Trier par date de modification (plus récent en premier)
        all_parcel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        start_idx = offset
        end_idx = start_idx + limit
        selected_files = all_parcel_files[start_idx:end_idx]
        
        parcels_data = []
        valid_files = 0
        
        for file_path in selected_files:
            parcel_id = file_path.stem
            try:
                # Vérifier que le fichier n'est pas vide
                if file_path.stat().st_size < 50:
                    log.warning(f"Fichier vide ignoré: {file_path.name}")
                    continue
                
                gdf = gpd.read_file(file_path)
                if not gdf.empty:
                    # Extraire la première ligne (normalement une seule par fichier)
                    row = gdf.iloc[0]
                    parcel_data = {
                        'id': parcel_id,
                        'name': row.get('name', ''),
                        'commune': row.get('commune', ''),
                        'section': row.get('section', ''),
                        'numero': row.get('numero', ''),
                        'superficie_m2': row.get('superficie_m2'),
                        'superficie_ha': row.get('superficie_ha'),
                        'proprietaire': row.get('proprietaire', ''),
                        'usage': row.get('usage', ''),
                        'geometry': mapping(row.geometry) if hasattr(row, 'geometry') else None
                    }
                    parcels_data.append(parcel_data)
                    valid_files += 1
                else:
                    log.warning(f"GeoDataFrame vide: {file_path.name}")
                    
            except Exception as e:
                log.error(f"Erreur lecture parcelle {file_path.stem}: {e}")
                continue
        
        log.info(f"Fichiers traités: {valid_files}/{len(selected_files)} valides")
        
        return jsonify({
            "data": parcels_data,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(all_parcel_files)
            },
            "debug": {
                "fichiers_totaux": len(all_parcel_files),
                "fichiers_traites": len(selected_files),
                "fichiers_valides": valid_files
            }
        })
        
    except Exception as e:
        log.error(f"Erreur critique listing parcelles: {e}")
        return jsonify({
            "error": "Erreur lors du listing des parcelles",
            "details": str(e)
        }), 500

@app.route('/api/parcels', methods=['POST'])
@require_auth
@handle_errors
def create_parcel_route(current_user):
    data = ParcelCreateModel.parse_raw(request.data)
    result = parcel_service.create_parcel(data)
    
    analytics_service.log_event(AnalyticsDataModel(
        event_type='parcel_created',
        details={'parcel_id': result['id'], 'user_id': current_user['id']}
    ))
    
    return jsonify(result), 201

@app.route('/api/parcels/bounds', methods=['POST'])
@optional_auth
@handle_errors
def get_parcels_by_bounds_route(current_user):
    minx = request.args.get('minx', type=float)
    miny = request.args.get('miny', type=float)
    maxx = request.args.get('maxx', type=float)
    maxy = request.args.get('maxy', type=float)
    crs = request.args.get('crs', DEFAULT_CRS_WGS84)
    buffer_m = request.args.get('buffer_m', type=float)
    
    if minx is None or miny is None or maxx is None or maxy is None:
        try:
            data = BoundsModel.parse_raw(request.data)
            bounds = data
        except ValidationError:
            return jsonify({"error": "Paramètres minx, miny, maxx, maxy requis"}), 400
    else:
        bounds = BoundsModel(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=crs, buffer_m=buffer_m
        )
    
    result = parcel_service.get_parcels_by_bounds(bounds)
    return jsonify(result)

@app.route('/api/parcels/search', methods=['POST'])
@optional_auth
@handle_errors
def search_parcels(current_user):
    data = SearchModel.parse_raw(request.data)
    query = data.query
    limit = data.limit
    offset = data.offset
    filters = data.filters or {}
    
    try:
        gdf = gpd.read_file(parcel_service.all_parcels_file)
    except Exception as e:
        log.error(f"Erreur lecture fichier agrégat pour recherche: {e}")
        return jsonify({"error": "Erreur interne"}), 500
    
    mask = pd.Series([True] * len(gdf))
    for key, value in filters.items():
        if key in gdf.columns:
            if isinstance(value, list):
                mask &= gdf[key].isin(value)
            else:
                mask &= gdf[key].astype(str).str.contains(str(value), na=False, case=False)
    
    if query:
        text_mask = (
            gdf['name'].astype(str).str.contains(query, na=False, case=False) |
            gdf['commune'].astype(str).str.contains(query, na=False, case=False) |
            gdf['numero'].astype(str).str.contains(query, na=False, case=False)
        )
        mask &= text_mask
    
    filtered_gdf = gdf[mask]
    
    start_idx = offset
    end_idx = start_idx + limit
    paginated_gdf = filtered_gdf.iloc[start_idx:end_idx]
    
    features = json.loads(paginated_gdf.to_json())['features']
    
    return jsonify({
        "data": features,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(filtered_gdf)
        }
    })

@app.route('/api/parcels/bulk', methods=['POST'])
@require_auth
@handle_errors
def bulk_create_parcels(current_user):
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

@app.route('/api/parcels/bulk', methods=['PUT'])
@require_auth
@handle_errors
def bulk_update_parcels(current_user):
    data = BulkParcelUpdateModel.parse_raw(request.data)
    updated_ids = []
    errors = []
    for i, update_info in enumerate(data.updates):
        parcel_id = update_info.get("id")
        update_data = update_info.get("data", {})
        
        parcel_file = BASE_DIR / "parcels" / f"{parcel_id}.geojson"
        if not parcel_file.exists():
            errors.append({"id": parcel_id, "error": "Parcelle introuvable"})
            continue
        
        try:
            gdf = gpd.read_file(parcel_file)
            if gdf.empty:
                errors.append({"id": parcel_id, "error": "Erreur lecture parcelle"})
                continue
            
            for key, value in update_data.items():
                if key in gdf.columns:
                    gdf.loc[0, key] = value
            
            if 'geometry' in update_data:
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
            
            gdf.to_file(parcel_file, driver='GeoJSON')
            parcel_service._update_aggregate()
            updated_ids.append(parcel_id)
            log.info(f"Parcelle mise à jour: {parcel_id}")
        except Exception as e:
            errors.append({"id": parcel_id, "error": str(e)})
    
    return jsonify({"updated_ids": updated_ids, "errors": errors})

@app.route('/api/parcels/bulk', methods=['DELETE'])
@require_auth
@handle_errors
def bulk_delete_parcels(current_user):
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
@optional_auth
@handle_errors
def get_parcel(parcel_id, current_user):
    output_crs = request.args.get('crs', DEFAULT_CRS)
    parcel = parcel_service.get_parcel(parcel_id, output_crs)
    if not parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify(parcel)

@app.route('/api/parcels/<parcel_id>', methods=['PUT'])
@require_auth
@handle_errors
def update_parcel(parcel_id, current_user):
    data = ParcelUpdateModel.parse_raw(request.data)
    updated_parcel = parcel_service.update_parcel(parcel_id, data)
    if not updated_parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify(updated_parcel)

@app.route('/api/parcels/<parcel_id>', methods=['DELETE'])
@require_auth
@handle_errors
def delete_parcel(parcel_id, current_user):
    if not parcel_service.delete_parcel(parcel_id):
        return jsonify({"error": "Parcelle introuvable"}), 404
    return jsonify({"message": "Parcelle supprimée", "id": parcel_id})

# ================================================================
# Routes CRS
# ================================================================
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
    if not epsg.startswith('EPSG:'):
        epsg = f"EPSG:{epsg}"
    try:
        info = coord_service.get_crs_info(epsg)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": f"Erreur CRS: {e}"}), 400

# ================================================================
# Routes OGC (WMS/WFS)
# ================================================================
import subprocess
QGIS_BIN = "/usr/lib/cgi-bin/qgis_mapserv.fcgi"

@app.route('/api/ogc/<service>', methods=['GET'])
@handle_errors
def ogc_service(service):
    if service.upper() not in ['WMS', 'WFS', 'WCS']:
        return jsonify({"error": "Service non supporté"}), 400
    
    if not os.path.isfile(QGIS_BIN):
        return jsonify({"error": "QGIS Server absent"}), 503
    
    qs = request.query_string.decode()
    map_param = request.args.get('MAP')
    
    if map_param:
        project_file = str((PROJECTS_DIR / map_param).resolve())
        if not project_file.startswith(str(PROJECTS_DIR)):
            return jsonify({"error": "Projet non autorisé"}), 403
    else:
        project_file = str(DEFAULT_PROJECT)
    
    if not os.path.exists(project_file):
        return jsonify({
            "error": "Projet QGIS introuvable",
            "message": f"Le projet {project_file} n'existe pas",
            "available_projects": [f.name for f in PROJECTS_DIR.glob("*.qgs")],
            "solution": "Utilisez /api/projects pour creer un projet ou uploader un fichier .qgs"
        }), 404
    
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

@app.route('/api/projects/init-default', methods=['POST'])
@handle_errors
def init_default_project():
    """Initialise le projet QGIS par défaut"""
    try:
        project_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" projectname="Projet par défaut - EPSG:32630">
  <homePath path=""/>
  <title>Projet par défaut - EPSG:32630</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    <spatialrefsys>
      <wkt>PROJCRS["WGS 84 / UTM zone 30N",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["UTM zone 30N",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",-3,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Engineering survey, topographic mapping."],AREA["Between 6°W and 0°W, northern hemisphere between equator and 84°N"],BBOX[0,-6,84,0]],ID["EPSG",32630]]</wkt>
      <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>
      <srsid>3452</srsid>
      <srid>32630</srid>
      <authid>EPSG:32630</authid>
      <description>WGS 84 / UTM zone 30N</description>
      <projectionacronym>utm</projectionacronym>
      <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
      <geographicflag>false</geographicflag>
    </spatialrefsys>
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
    <layer-tree-layer expanded="1" checked="Qt::Checked" id="parcels" name="Parcelles" source="./data/parcels/all_parcels.geojson" providerKey="ogr">
      <customproperties/>
    </layer-tree-layer>
  </layer-tree-group>
  <mapcanvas name="theMapCanvas" annotationsVisible="1">
    <units>meters</units>
    <extent>
      <xmin>166021.44</xmin>
      <ymin>1195948.28</ymin>
      <xmax>833978.56</xmax>
      <ymax>1863805.72</ymax>
    </extent>
    <rotation>0</rotation>
    <destinationsrs>
      <spatialrefsys>
        <wkt>PROJCRS["WGS 84 / UTM zone 30N",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["UTM zone 30N",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",-3,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Engineering survey, topographic mapping."],AREA["Between 6°W and 0°W, northern hemisphere between equator and 84°N"],BBOX[0,-6,84,0]],ID["EPSG",32630]]</wkt>
        <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>
        <srsid>3452</srsid>
        <srid>32630</srid>
        <authid>EPSG:32630</authid>
        <description>WGS 84 / UTM zone 30N</description>
        <projectionacronym>utm</projectionacronym>
        <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
        <geographicflag>false</geographicflag>
      </spatialrefsys>
    </destinationsrs>
    <rendermaptile>0</rendermaptile>
    <expressionContextScope/>
  </mapcanvas>
  <properties>
    <WMSServiceCapabilities type="bool">true</WMSServiceCapabilities>
    <WMSServiceTitle type="QString">API QGIS - EPSG:32630</WMSServiceTitle>
    <WMSServiceAbstract type="QString">Service WMS optimisé pour applications mobiles</WMSServiceAbstract>
    <WMSOnlineResource type="QString">https://flashapiprod-ywyr.onrender.com/api/ogc/WMS</WMSOnlineResource>
    <WFSServiceCapabilities type="bool">true</WFSServiceCapabilities>
    <WFSServiceTitle type="QString">API QGIS - WFS</WFSServiceTitle>
  </properties>
</qgis>"""
        
        with open(DEFAULT_PROJECT, 'w') as f:
            f.write(project_content)
        
        log.info("Projet par defaut cree")
        return jsonify({
            "message": "Projet QGIS par defaut cree",
            "path": str(DEFAULT_PROJECT),
            "crs": "EPSG:4326"
        })
        
    except Exception as e:
        log.error(f"Erreur creation projet: {e}")
        return jsonify({"error": f"Erreur creation projet: {str(e)}"}), 500

# ================================================================
# Routes Rapports PDF
# ================================================================
@app.route('/api/reports/parcel/<parcel_id>', methods=['GET'])
@handle_errors
@require_session
def generate_parcel_report(parcel_id, session):
    format_requested = request.args.get('format', 'pdf').lower()
    if format_requested not in ['pdf', 'png', 'jpg', 'jpeg']:
        return jsonify({"error": "Format non supporté"}), 400
    
    qgis_mgr = get_qgis_manager()
    if not qgis_mgr.is_initialized():
        return jsonify({"error": "QGIS non initialisé"}), 500
    
    parcel = parcel_service.get_parcel(parcel_id)
    if not parcel:
        return jsonify({"error": "Parcelle introuvable"}), 404
    
    cache_key = f"report:{parcel_id}:{format_requested}"
    cached_file = cache_get(cache_key)
    if cached_file:
        return send_file(
            BytesIO(cached_file),
            mimetype='application/pdf' if format_requested == 'pdf' else f"image/{format_requested}",
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
            f"Section : {parcel['section']} - N {parcel['numero']}\n"
            f"Superficie : {parcel['superficie_ha']} ha ({parcel['superficie_m2']} m2)\n"
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
        else:
            map_settings = classes['QgsMapSettings']()
            map_settings.setLayers([vlayer])
            map_settings.setDestinationCrs(project.crs())
            map_settings.setExtent(map_extent)
            map_settings.setOutputSize(classes['QSize'](800, 600))
            renderer = classes['QgsMapRendererParallelJob'](map_settings)
            renderer.start()
            renderer.waitForFinished()
            image = renderer.renderedImage()
            
            ba = QByteArray()
            buffer = QBuffer(ba)
            buffer.open(QIODevice.WriteOnly)
            success = image.save(buffer, format_requested.upper() if format_requested != 'jpg' else 'JPEG')
            if not success:
                return jsonify({"error": "Échec export image"}), 500
            output_bytes = bytes(ba)
            mimetype = f"image/{format_requested}"
        
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

@app.route('/api/reports/project/<project_id>', methods=['GET'])
@handle_errors
def generate_project_report(project_id):
    return jsonify({"error": "Génération de rapport projet non implémentée"}), 501

# ================================================================
# Routes Fichiers
# ================================================================
@app.route('/api/files/upload', methods=['POST'])
@require_auth
@handle_errors
def upload_file(current_user):
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
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_by": current_user['id']
        }
        
        if ext in ['shp', 'geojson', 'gpkg']:
            try:
                gdf = gpd.read_file(str(filepath))
                file_info['geo_info'] = {
                    'crs': str(gdf.crs) if gdf.crs else 'unknown',
                    'features': len(gdf),
                    'bounds': gdf.total_bounds.tolist() if not gdf.empty else None
                }
            except Exception as e:
                log.warning(f"Impossible de lire métadonnées geo: {e}")
        
        log.info(f"Fichier uploadé: {filename}")
        return jsonify(file_info), 201
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise

@app.route('/api/files/<category>', methods=['GET'])
@optional_auth
@handle_errors
def list_files(category, current_user):
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
@optional_auth
@handle_errors
def download_file(category, filename, current_user):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    
    safe_filename = secure_filename(filename)
    file_path = BASE_DIR / category / safe_filename
    
    if not file_path.exists():
        return jsonify({"error": "Fichier introuvable"}), 404
    
    return send_from_directory(BASE_DIR / category, safe_filename, as_attachment=True)

@app.route('/api/files/<category>/<filename>', methods=['DELETE'])
@require_auth
@handle_errors
def delete_file(category, filename, current_user):
    if category not in CATEGORIES:
        return jsonify({"error": "Catégorie invalide"}), 400
    
    safe_filename = secure_filename(filename)
    file_path = BASE_DIR / category / safe_filename
    
    if not file_path.exists():
        return jsonify({"error": "Fichier introuvable"}), 404
    
    file_path.unlink()
    log.info(f"Fichier supprimé: {safe_filename}")
    return jsonify({"message": "Fichier supprimé", "filename": safe_filename})

# ================================================================
# Routes Analytics
# ================================================================
@app.route('/api/analytics/data', methods=['POST'])
@optional_auth
@handle_errors
def log_analytics_data(current_user):
    data = AnalyticsDataModel.parse_raw(request.data)
    analytics_service.log_event(data)
    return jsonify({"message": "Données analytics logguées", "event_type": data.event_type})

@app.route('/api/analytics/summary', methods=['GET'])
@optional_auth
@handle_errors
def get_analytics_summary(current_user):
    summary = analytics_service.get_summary()
    return jsonify({"data": summary})

@app.route('/api/analytics/events', methods=['GET'])
@optional_auth
@handle_errors
def get_analytics_events(current_user):
    limit = request.args.get('limit', default=100, type=int)
    events = analytics_service._read_events()
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
@optional_auth
@handle_errors
def log_user_behavior(current_user):
    data = UserBehaviorModel.parse_raw(request.data)
    if current_user:
        data.user_id = current_user['id']
    analytics_service.log_user_behavior(data)
    return jsonify({"message": "Comportement utilisateur loggué", "action": data.action})

# ================================================================
# Routes Performance Metrics
# ================================================================
@app.route('/api/performance/metrics', methods=['POST'])
@optional_auth
@handle_errors
def log_performance_metric(current_user):
    data = PerformanceMetricsModel.parse_raw(request.data)
    performance_service.log_metric(data)
    return jsonify({"message": "Métrique performance logguée", "metric_name": data.metric_name})

@app.route('/api/performance/metrics', methods=['GET'])
@optional_auth
@handle_errors
def get_performance_metrics(current_user):
    limit = request.args.get('limit', default=100, type=int)
    metrics = performance_service.get_metrics(limit=limit)
    return jsonify({
        "data": metrics,
        "pagination": {
            "limit": limit,
            "offset": 0,
            "total": len(metrics)
        }
    })

# ================================================================
# Routes Admin
# ================================================================
@app.route('/api/admin/stats', methods=['GET'])
@require_admin
@handle_errors
def admin_stats(current_user):
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
@require_admin
@handle_errors
def clear_cache(current_user):
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

# ================================================================
# Routes Documentation
# ================================================================
@app.route('/api/stats', methods=['GET'])
@optional_auth
@handle_errors
def system_stats(current_user):
    if current_user and current_user.get('is_admin'):
        return admin_stats(current_user)
    
    return jsonify({
        "parcels": {
            "total": len(list((BASE_DIR / "parcels").glob("[!all_]*.geojson")))
        },
        "system": {
            "qgis": get_qgis_manager().is_initialized(),
            "default_crs": DEFAULT_CRS,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    })

@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({
        "version": "3.2.0",
        "title": "API QGIS Server - Optimisée Mobile avec Auth JWT",
        "default_crs": DEFAULT_CRS,
        "authentication": {
            "type": "JWT Bearer Token",
            "endpoints": {
                "register": {
                    "method": "POST",
                    "url": "/api/auth/register",
                    "body": {
                        "email": "user@example.com",
                        "password": "SecurePass123",
                        "username": "username",
                        "full_name": "John Doe (optional)"
                    }
                },
                "login": {
                    "method": "POST",
                    "url": "/api/auth/login",
                    "body": {
                        "email": "user@example.com",
                        "password": "SecurePass123"
                    },
                    "response": {
                        "access_token": "eyJ...",
                        "refresh_token": "eyJ...",
                        "expires_in": 3600
                    }
                },
                "refresh": {
                    "method": "POST",
                    "url": "/api/auth/refresh",
                    "body": {
                        "refresh_token": "eyJ..."
                    }
                },
                "usage": "Inclure header: Authorization: Bearer <access_token>"
            },
            "password_requirements": {
                "min_length": 8,
                "must_contain": ["majuscule", "minuscule", "chiffre"]
            },
            "token_expiry": {
                "access_token": f"{JWT_ACCESS_TOKEN_EXPIRE_MINUTES} minutes",
                "refresh_token": f"{JWT_REFRESH_TOKEN_EXPIRE_DAYS} jours"
            }
        },
        "protected_routes": {
            "admin_only": [
                "GET /api/admin/stats",
                "POST /api/admin/cache/clear"
            ],
            "auth_required": [
                "POST /api/parcels",
                "PUT /api/parcels/{id}",
                "DELETE /api/parcels/{id}",
                "POST /api/projects",
                "PUT /api/projects/{id}",
                "DELETE /api/projects/{id}",
                "POST /api/files/upload",
                "DELETE /api/files/{category}/{filename}"
            ],
            "public": [
                "GET /api/parcels",
                "GET /api/parcels/{id}",
                "POST /api/parcels/bounds",
                "GET /api/health",
                "GET /api/docs"
            ]
        },
        "endpoints": {
            "auth": {
                "register": "POST /api/auth/register",
                "login": "POST /api/auth/login",
                "refresh": "POST /api/auth/refresh",
                "logout": "POST /api/auth/logout",
                "me": "GET /api/auth/me",
                "change_password": "POST /api/auth/change-password"
            },
            "projects": {
                "list": "GET /api/projects?limit=100&offset=0",
                "create": "POST /api/projects",
                "get": "GET /api/projects/{id}",
                "update": "PUT /api/projects/{id}",
                "delete": "DELETE /api/projects/{id}"
            },
            "parcels": {
                "list": "GET /api/parcels?limit=100&offset=0",
                "create": "POST /api/parcels",
                "update": "PUT /api/parcels/{id}",
                "by_bounds": "POST /api/parcels/bounds",
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
                "wcs": "GET /api/ogc/WCS?..."
            },
            "reports": {
                "parcel": "GET /api/reports/parcel/{id}?format=pdf|png|jpg"
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
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de santé complet"""
    health_status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.2.0",
        "services": {
            "qgis": get_qgis_manager().is_initialized(),
            "redis": bool(redis_client),
            "storage": {
                "available": BASE_DIR.exists(),
                "writable": os.access(BASE_DIR, os.W_OK)
            }
        }
    }
    
    # Vérifications détaillées
    try:
        # Test écriture storage
        test_file = BASE_DIR / "health_check.txt"
        test_file.write_text("health_check")
        test_file.unlink()
        health_status["services"]["storage"]["writable"] = True
    except Exception as e:
        health_status["services"]["storage"]["writable"] = False
        health_status["status"] = "degraded"
        log.warning(f"Storage non accessible: {e}")
    
    # Vérification Redis
    if redis_client:
        try:
            redis_client.ping()
            health_status["services"]["redis"] = True
        except Exception as e:
            health_status["services"]["redis"] = False
            health_status["status"] = "degraded"
            log.warning(f"Redis non accessible: {e}")
    
    # Vérification QGIS
    if not health_status["services"]["qgis"]:
        health_status["status"] = "degraded"
    
    return jsonify(health_status)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "API QGIS Server",
        "version": "3.2.0",
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
# Initialisation sécurisée
# ================================================================
def initialize_services():
    """Initialisation sécurisée de tous les services"""
    log.info("Initialisation des services...")
    
    # Initialisation QGIS
    qgis_mgr = get_qgis_manager()
    success, error = qgis_mgr.initialize()
    
    if not success:
        log.error(f"Échec initialisation QGIS: {error}")
        # Ne pas quitter, l'API peut fonctionner sans QGIS pour certaines routes
    
    # Vérification des dossiers
    required_dirs = [BASE_DIR, PROJECTS_DIR, CACHE_DIR, TILES_DIR]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.exists():
            log.error(f"Impossible de créer le dossier: {directory}")
    
    # Démarrage du nettoyage des sessions
    try:
        cleanup_thread = Thread(target=cleanup_expired_sessions, daemon=True)
        cleanup_thread.start()
        log.info("Thread de nettoyage des sessions démarré")
    except Exception as e:
        log.error(f"Erreur démarrage thread nettoyage: {e}")
    
    log.info("Initialisation des services terminée")

# ================================================================
# Point d'entrée
# ================================================================
if __name__ == '__main__':
    # Initialisation avant démarrage
    initialize_services()
    
    port = int(os.getenv('PORT', 10000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    log.info("=" * 70)
    log.info("API QGIS Server v3.2.0 - Optimisée pour Render & Mobile + Auth JWT")
    log.info(f"Données: {BASE_DIR}")
    log.info(f"CRS par défaut: {DEFAULT_CRS}")
    log.info(f"Redis: {'Actif' if redis_client else 'Cache fichier'}")
    log.info(f"QGIS: {'Initialisé' if get_qgis_manager().is_initialized() else 'Échec'}")
    log.info(f"Sessions: Timeout {SESSION_TIMEOUT.seconds // 60}min")
    log.info(f"JWT: Access {JWT_ACCESS_TOKEN_EXPIRE_MINUTES}min / Refresh {JWT_REFRESH_TOKEN_EXPIRE_DAYS}j")
    log.info("=" * 70)
    
    if os.getenv('ENVIRONMENT') == 'development':
        app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
    else:
        log.info("Mode production : utiliser gunicorn")
        log.info("Commande: gunicorn --workers 2 --threads 4 --bind 0.0.0.0:$PORT api:app")