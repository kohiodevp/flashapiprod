# ================================================================
# Dockerfile - Image optimisée QGIS + Flask (Utilisation d'une image QGIS existante)
# ================================================================
# Utiliser une image officielle de QGIS ou une image Ubuntu avec QGIS pré-installé
# Exemple : Utilisation de l'image Ubuntu 22.04 avec dépôt QGIS
# ATTENTION : Il se peut que des images QGIS officielles existent, à vérifier.
# Sinon, on part d'une Ubuntu 22.04 et on installe QGIS *en premier* via apt.
# Cette version tente de minimiser les installations via pip pour les composants QGIS.
# On installe les paquets système QGIS, puis *seulement* les paquets Python non géospatiaux via pip.

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
    # Installer les bibliothèques de base pour python (souvent dépendances de qgis/python3-qgis)
    python3-numpy \
    python3-pandas \
    python3-requests \
    python3-flask \
    python3-flask-cors \
    python3-flask-compress \
    python3-werkzeug \
    python3-pydantic \
    python3-passlib \
    python3-jwt \
    python3-redis \
    python3-gunicorn \
    python3-geopandas \
    python3-shapely \
    python3-pyproj \
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

# --- Installer QGIS (et ses dépendances Python associées) ---
RUN apt-get update || (sleep 10 && apt-get update) \
    && apt-get install -y \
    qgis \
    qgis-server \
    qgis-plugin-grass \
    python3-qgis \
    qgis-providers \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Vérifier les versions système installées (utile pour le debug) ---
RUN echo "--- Versions des bibliothèques géospatiales système (après installation QGIS) ---" && \
    python3 -c "import gdal; print('GDAL Python (system):', gdal.__version__)" || echo "GDAL Python (system) not found" && \
    python3 -c "import fiona; print('Fiona Python (system):', fiona.__version__)" || echo "Fiona Python (system) not found" && \
    python3 -c "import pyproj; print('PyProj Python (system):', pyproj.__version__)" || echo "PyProj Python (system) not found" && \
    python3 -c "import shapely; print('Shapely Python (system):', shapely.__version__)" || echo "Shapely Python (system) not found" && \
    python3 -c "import geopandas; print('GeoPandas Python (system):', geopandas.__version__)" || echo "GeoPandas Python (system) not found" && \
    ogrinfo --version || echo "GDAL CLI not found"

# --- Installation des autres dépendances Python via pip ---
# Copier requirements.txt
COPY requirements.txt /tmp/requirements.txt

# --- Filtrer requirements.txt pour enlever les bibliothèques géospatiales et les dépendances de base déjà installées via apt ---
# On suppose que geopandas, shapely, pyproj, numpy, pandas, gdal, fiona sont gérées par apt/python3-qgis/python3-geopandas
# On installe donc le reste via pip.
# ATTENTION : Si la version installée par apt est trop ancienne, cette méthode échouera.
# Mais elle évite les conflits de version pip/apt.
RUN grep -v -E '^(GDAL|fiona|pyproj|shapely|geopandas|numpy|pandas|requests|flask|flask-cors|flask-compress|werkzeug|pydantic|passlib|PyJWT|redis|gunicorn)==.*$' /tmp/requirements.txt > /tmp/requirements_filtered.txt

# Installer les paquets pip restants (sans réinstaller les dépendances de base ou géospatiales)
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
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "1", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api:app"]
