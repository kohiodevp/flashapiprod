# Dockerfile Production - QGIS Server + API Sécurisée
FROM qgis/qgis-server:latest

LABEL maintainer="kohiodevp@gmail.com"
LABEL version="3.0.0"
LABEL description="QGIS Server avec API REST sécurisée - Production Ready"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    QGIS_SERVER_LOG_LEVEL=0 \
    QGIS_PROJECT_FILE=/etc/qgis/projects/project.qgs \
    BASE_DIR=/data \
    DEFAULT_CRS=EPSG:32630 \
    WORKERS=4 \
    TIMEOUT=120 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=50

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    redis-server \
    nginx \
    supervisor \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Installation des dépendances Python avec versions fixes
RUN pip3 install --no-cache-dir --break-system-packages \
    # Web Framework
    flask==2.3.3 \
    flask-cors==4.0.0 \
    flask-limiter==3.5.0 \
    flask-caching==2.1.0 \
    gunicorn==21.2.0 \
    werkzeug==2.3.7 \
    # Sécurité
    pyjwt==2.8.0 \
    bcrypt==4.0.1 \
    cryptography==41.0.7 \
    # Cache et Storage
    redis==5.0.1 \
    hiredis==2.2.3 \
    # Validation
    pydantic==1.10.13 \
    # Science et Données
    && pip3 install --no-cache-dir --break-system-packages \
    numpy==1.26.4 \
    pandas==2.1.4 \
    # Géospatial
    && pip3 install --no-cache-dir --break-system-packages \
    shapely==2.0.2 \
    pyproj==3.6.1 \
    fiona==1.9.5 \
    geopandas==0.14.1 \
    rtree==1.1.0 \
    # Documents et utilitaires
    && pip3 install --no-cache-dir --break-system-packages \
    reportlab==4.0.7 \
    pillow==10.1.0 \
    psycopg2-binary==2.9.9 \
    aiofiles==23.2.1 \
    python-multipart==0.0.6

# Création de la structure de répertoires
RUN mkdir -p \
    /etc/qgis/projects \
    /data/shapefiles \
    /data/csv \
    /data/geojson \
    /data/projects \
    /data/other \
    /data/tiles \
    /data/parcels \
    /data/documents \
    /app \
    /var/log/supervisor \
    /var/log/qgis \
    /var/log/nginx \
    /var/lib/redis \
    && chmod 755 /data/* \
    && chmod 755 /var/log/*

# Projet QGIS avec EPSG:32630
RUN cat > /etc/qgis/projects/project.qgs << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">
<qgis projectname="Projet Principal UTM 30N" version="3.28.0">
  <title>Projet QGIS Server - EPSG:32630 Production</title>
  <spatialrefsys>
    <authid>EPSG:32630</authid>
    <srsid>3452</srsid>
    <srid>32630</srid>
    <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>
    <description>WGS 84 / UTM zone 30N</description>
  </spatialrefsys>
  <layer-tree-group>
    <customproperties/>
  </layer-tree-group>
  <properties>
    <WMSServiceCapabilities type="bool">true</WMSServiceCapabilities>
    <WMSServiceTitle type="QString">QGIS Server Production</WMSServiceTitle>
    <WMSServiceAbstract type="QString">Service WMS sécurisé avec EPSG:32630</WMSServiceAbstract>
    <WFSServiceCapabilities type="bool">true</WFSServiceCapabilities>
  </properties>
</qgis>
EOF

# Configuration Redis optimisée pour production
RUN cat > /etc/redis/redis.conf << 'EOF'
# Réseau
bind 127.0.0.1
port 6379
tcp-backlog 511
timeout 300
tcp-keepalive 300

# Sécurité
protected-mode yes
# requirepass changeme_in_production

# Mémoire
maxmemory 256mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# Logs
loglevel notice
logfile /var/log/redis/redis-server.log

# Performance
databases 16
slowlog-log-slower-than 10000
slowlog-max-len 128
EOF

# Configuration Nginx optimisée
RUN cat > /etc/nginx/nginx.conf << 'EOF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;
error_log /var/log/nginx/error.log warn;

events {
    worker_connections 2048;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 50M;
    client_body_buffer_size 128k;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss 
               application/rss+xml font/truetype font/opentype 
               application/vnd.ms-fontobject image/svg+xml;

    # Sécurité
    server_tokens off;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ogc_limit:10m rate=20r/s;
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    # Upstream API
    upstream api_backend {
        least_conn;
        server 127.0.0.1:10000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    server {
        listen 80 default_server;
        listen [::]:80 default_server;
        server_name _;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;

        # Health check
        location /health {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://api_backend/api/health;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            access_log off;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            limit_conn addr 10;
            
            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache_bypass $http_upgrade;
        }

        # Services OGC
        location ~ ^/api/ogc/(wms|wfs|wcs) {
            limit_req zone=ogc_limit burst=50 nodelay;
            
            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering on;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }

        # Fichiers statiques (si nécessaire)
        location /static/ {
            alias /data/static/;
            expires 1d;
            add_header Cache-Control "public, immutable";
        }

        # Page par défaut
        location / {
            return 301 /api/docs;
        }

        # Gestion des erreurs
        error_page 404 /404.html;
        error_page 500 502 503 504 /50x.html;
        location = /50x.html {
            root /usr/share/nginx/html;
        }
    }
}
EOF

# Configuration Supervisor
RUN cat > /etc/supervisor/conf.d/supervisord.conf << 'EOF'
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor
loglevel=info

[program:redis]
command=/usr/bin/redis-server /etc/redis/redis.conf
autostart=true
autorestart=true
priority=1
stdout_logfile=/var/log/supervisor/redis.log
stderr_logfile=/var/log/supervisor/redis.err
startsecs=5
stopwaitsecs=10

[program:api]
command=/usr/bin/gunicorn \
    --workers %(ENV_WORKERS)s \
    --bind 0.0.0.0:%(ENV_PORT)s \
    --timeout %(ENV_TIMEOUT)s \
    --max-requests %(ENV_MAX_REQUESTS)s \
    --max-requests-jitter %(ENV_MAX_REQUESTS_JITTER)s \
    --worker-class sync \
    --worker-tmp-dir /dev/shm \
    --access-logfile /var/log/supervisor/api-access.log \
    --error-logfile /var/log/supervisor/api-error.log \
    --log-level info \
    --preload \
    api_production:app
directory=/app
autostart=true
autorestart=true
priority=2
stdout_logfile=/var/log/supervisor/api.log
stderr_logfile=/var/log/supervisor/api.err
startsecs=10
stopwaitsecs=30
environment=
    PYTHONUNBUFFERED="1",
    BASE_DIR="%(ENV_BASE_DIR)s",
    QGIS_PROJECT_FILE="%(ENV_QGIS_PROJECT_FILE)s",
    DEFAULT_CRS="%(ENV_DEFAULT_CRS)s",
    REDIS_HOST="127.0.0.1",
    REDIS_PORT="6379"

[program:nginx]
command=/usr/sbin/nginx -g 'daemon off;'
autostart=true
autorestart=true
priority=3
stdout_logfile=/var/log/supervisor/nginx.log
stderr_logfile=/var/log/supervisor/nginx.err
startsecs=5
stopwaitsecs=10
EOF

# Script d'initialisation
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "Démarrage QGIS Server API Production"
echo "=========================================="

# Vérifications
echo "Vérification de l'environnement..."

if [ ! -f "$QGIS_PROJECT_FILE" ]; then
    echo "ERREUR: Projet QGIS non trouvé: $QGIS_PROJECT_FILE"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERREUR: Répertoire de données non trouvé: $BASE_DIR"
    exit 1
fi

# Création des répertoires manquants
for dir in shapefiles csv geojson projects other tiles parcels documents; do
    mkdir -p "$BASE_DIR/$dir"
    chmod 755 "$BASE_DIR/$dir"
done

# Permissions
echo "Configuration des permissions..."
chown -R www-data:www-data /var/log/nginx
chown -R redis:redis /var/lib/redis /var/log/redis
chmod 755 /app/api_production.py

# Génération des secrets si non définis
if [ -z "$SECRET_KEY" ]; then
    export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "ATTENTION: SECRET_KEY généré automatiquement"
fi

if [ -z "$JWT_SECRET" ]; then
    export JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "ATTENTION: JWT_SECRET généré automatiquement"
fi

# Test de connexion Redis
echo "Test connexion Redis..."
timeout 5 redis-cli -h 127.0.0.1 -p 6379 ping > /dev/null 2>&1 || {
    echo "Redis n'est pas encore démarré, attente..."
}

# Validation de l'API
echo "Validation de l'API..."
python3 -c "import api_production; print('API validée')" || {
    echo "ERREUR: Validation de l'API échouée"
    exit 1
}

echo "Configuration terminée"
echo "=========================================="
echo "Démarrage des services..."
echo "  - Redis: 127.0.0.1:6379"
echo "  - API: 0.0.0.0:$PORT"
echo "  - Nginx: 0.0.0.0:80"
echo "  - Workers: $WORKERS"
echo "  - CRS par défaut: $DEFAULT_CRS"
echo "=========================================="

# Démarrage de Supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
EOF

RUN chmod +x /app/start.sh

# Copie de l'API production
COPY api.py /app/api_production.py

# Volumes
VOLUME ["/data", "/var/log"]

# Ports
EXPOSE 80 10000

# Health check amélioré
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Labels pour métadonnées
LABEL org.opencontainers.image.title="QGIS Server API Production"
LABEL org.opencontainers.image.description="API REST sécurisée pour QGIS Server avec EPSG:32630"
LABEL org.opencontainers.image.version="3.0.0"

WORKDIR /app

CMD ["/app/start.sh"]