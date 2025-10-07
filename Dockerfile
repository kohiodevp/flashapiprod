# ================================================================
# Dockerfile - Image optimisée QGIS + Flask (Version pour Render - Basée sur apt)
# ================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    QGIS_DISABLE_MESSAGE_HOOKS=1 \
    QGIS_NO_OVERRIDE_IMPORT=1 \
    PORT=10000

# --- Dépendances système de base ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    # Installer les bibliothèques C/C++ géospatiales système nécessaires pour QGIS
    gdal-bin \
    libgdal-dev \
    # Installer les bibliothèques Python de base (souvent dépendances de qgis/python3-qgis)
    python3-numpy \
    python3-pandas \
    # Installer les dépendances Qt nécessaires pour QGIS
    libqt5gui5 \
    libqt5core5a \
    libqt5printsupport5 \
    libqt5svg5 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Dépôt QGIS ---
RUN wget -O - https://qgis.org/downloads/qgis-archive-keyring.gpg   | gpg --dearmor | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu   jammy main" > /etc/apt/sources.list.d/qgis.list

# --- Installer QGIS (et ses dépendances Python associées via apt) ---
RUN apt-get update || (sleep 10 && apt-get update) \
    && apt-get install -y \
    qgis \
    qgis-server \
    qgis-plugin-grass \
    python3-qgis \
    qgis-providers \
    # Installer explicitement les paquets Python géospatiaux correspondants via apt
    python3-gdal \
    python3-fiona \
    python3-pyproj \
    python3-shapely \
    python3-geopandas \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Vérifier les versions système installées (utile pour le debug) ---
RUN echo "--- Versions des bibliothèques géospatiales système (après installation QGIS/apt) ---" && \
    python3 -c "import gdal; print('GDAL Python (system):', gdal.__version__)" 2>/dev/null || echo "GDAL Python (system) not found" && \
    python3 -c "import fiona; print('Fiona Python (system):', fiona.__version__)" 2>/dev/null || echo "Fiona Python (system) not found" && \
    python3 -c "import pyproj; print('PyProj Python (system):', pyproj.__version__)" 2>/dev/null || echo "PyProj Python (system) not found" && \
    python3 -c "import shapely; print('Shapely Python (system):', shapely.__version__)" 2>/dev/null || echo "Shapely Python (system) not found" && \
    python3 -c "import geopandas; print('GeoPandas Python (system):', geopandas.__version__)" 2>/dev/null || echo "GeoPandas Python (system) not found" && \
    ogrinfo --version 2>/dev/null || echo "GDAL CLI not found"

# --- Installation des autres dépendances Python via pip ---
# Copier requirements.txt
COPY requirements.txt /tmp/requirements.txt

# --- Filtrer requirements.txt pour enlever les bibliothèques géospatiales et les dépendances de base installées via apt ---
# Ces paquets sont gérés par apt et devraient être compatibles avec QGIS
# On installe donc le reste via pip.
RUN grep -v -E '^(GDAL|fiona|pyproj|shapely|geopandas|numpy|pandas)==.*$' /tmp/requirements.txt > /tmp/requirements_filtered.txt

# Installer les paquets pip restants (sans réinstaller les dépendances de base ou géospatiales)
# Utiliser --no-deps pour être sûr
RUN pip3 install --no-cache-dir --no-deps -r /tmp/requirements_filtered.txt

# Création structure dossiers
RUN mkdir -p /opt/render/project/src/data/{shapefiles,csv,geojson,projects,other,tiles,parcels,documents,cache}

# Copie application
WORKDIR /opt/render/project/src
COPY api.py .
COPY default.qgs data/projects/default.qgs

# Permissions
RUN chmod -R 755 /opt/render/project/src && \
    chmod -R 777 /opt/render/project/src/data

# Exposition port
EXPOSE 10000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:10000/api/health || exit 1

# Démarrage avec Gunicorn
# IMPORTANT : Gunicorn doit attendre que QGIS soit initialisé.
# L'initialisation se fait dans api.py au démarrage du script.
# Si QGIS échoue, api.py lèvera une exception et le worker Gunicorn mourra.
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "1", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api:app"]
