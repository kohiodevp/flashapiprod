# Dockerfile complet – QGIS Server + API avec EPSG:32630 - SOLUTION FONCTIONNELLE
FROM qgis/qgis-server:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
ENV QGIS_SERVER_LOG_LEVEL=0
ENV QGIS_PROJECT_FILE=/etc/qgis/projects/project.qgs
ENV BASE_DIR=/data
ENV DEFAULT_CRS=EPSG:32630

# ------------------------------------------------------------------
# Installation système
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    redis-server \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Installation séquentielle des dépendances Python AVEC --break-system-packages
# ------------------------------------------------------------------

# 1. Utilisation des versions système de pip et setuptools (installées par Debian)
# Pas de mise à jour nécessaire car les versions système sont compatibles

# 2. Dépendances de base
RUN pip3 install --no-cache-dir --break-system-packages \
    flask==2.3.3 \
    flask-cors==4.0.0 \
    gunicorn==21.2.0

# 3. Dépendances de sécurité et données
RUN pip3 install --no-cache-dir --break-system-packages \
    pyjwt==2.8.0 \
    bcrypt==4.0.1 \
    redis==5.0.1 \
    pydantic==1.10.13

# 4. Dépendances scientifiques (versions compatibles Python 3.12)
RUN pip3 install --no-cache-dir --break-system-packages numpy==1.26.4
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
    pandas==2.1.4 \
    matplotlib==3.8.2

# 5. Dépendances géospatiales (versions compatibles Python 3.12)
RUN pip3 install --no-cache-dir --break-system-packages \
    shapely==2.0.2 \
    pyproj==3.6.1 \
    fiona==1.9.5

# 6. GeoPandas et dépendances restantes (versions compatibles Python 3.12)
RUN pip3 install --no-cache-dir --break-system-packages \
    geopandas==0.14.1 \
    rtree==1.1.0

# 7. Dépendances pour documents (versions compatibles Python 3.12)
RUN pip3 install --no-cache-dir --break-system-packages \
    reportlab==4.0.7 \
    pillow==10.1.0 \
    psycopg2-binary==2.9.9 \
    aiofiles==23.2.1 \
    python-multipart==0.0.6 \
    dicttoxml==1.7.16

# ------------------------------------------------------------------
# Création des répertoires
# ------------------------------------------------------------------
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
    /var/log/supervisor

# ------------------------------------------------------------------
# Projet QGIS avec EPSG:32630
# ------------------------------------------------------------------
RUN printf '%s\n' \
    '<?xml version="1.0" encoding="UTF-8"?>' \
    '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">' \
    '<qgis projectname="Projet Principal UTM 30N" version="3.28.0">' \
    '  <title>Projet QGIS Server - EPSG:32630</title>' \
    '  <spatialrefsys>' \
    '    <authid>EPSG:32630</authid>' \
    '    <srsid>3452</srsid>' \
    '    <srid>32630</srid>' \
    '    <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>' \
    '    <description>WGS 84 / UTM zone 30N</description>' \
    '  </spatialrefsys>' \
    '  <layer-tree-group>' \
    '    <customproperties/>' \
    '  </layer-tree-group>' \
    '</qgis>' > /etc/qgis/projects/project.qgs

# ------------------------------------------------------------------
# Configuration Redis
# ------------------------------------------------------------------
RUN printf '%s\n' \
    'bind 127.0.0.1' \
    'port 6379' \
    'timeout 300' \
    'databases 16' \
    'save 900 1' \
    'save 300 10' \
    'save 60 10000' \
    'rdbcompression yes' \
    'dbfilename dump.rdb' \
    'dir /var/lib/redis' > /etc/redis/redis.conf

# ------------------------------------------------------------------
# Configuration nginx
# ------------------------------------------------------------------
COPY nginx.conf /etc/nginx/nginx.conf

# ------------------------------------------------------------------
# Configuration Supervisor
# ------------------------------------------------------------------
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ------------------------------------------------------------------
# Script d'initialisation
# ------------------------------------------------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ------------------------------------------------------------------
# Copie de l'API
# ------------------------------------------------------------------
COPY api.py /app/api.py

# ------------------------------------------------------------------
# Configuration finale
# ------------------------------------------------------------------
WORKDIR /app

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

CMD ["/app/start.sh"]