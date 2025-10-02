#!/bin/bash
set -e

echo "=========================================="
echo "Configuration EPSG:32630 pour QGIS Server"
echo "=========================================="

# Variables d'environnement
export QGIS_PROJECT_FILE=${QGIS_PROJECT_FILE:-/etc/qgis/projects/project.qgs}
export BASE_DIR=${BASE_DIR:-/data}
export DEFAULT_CRS=${DEFAULT_CRS:-EPSG:32630}
export PORT=${PORT:-10000}

# Test de la configuration EPSG:32630
echo "Test de la configuration EPSG:32630..."
cs2cs +init=epsg:4326 +to +init=epsg:32630 <<EOF || echo "⚠️  Attention: Vérifier proj4"
-1.5 12.5
EOF

# Test Python
echo "Test de l'environnement Python..."
python3 -c "
import sys
print(f'✅ Python: {sys.version}')

try:
    import flask
    print(f'✅ Flask: {flask.__version__}')
except ImportError as e:
    print(f'❌ Flask: NON INSTALLÉ - {e}')
    sys.exit(1)

try:
    import geopandas as gpd
    print(f'✅ GeoPandas: {gpd.__version__}')
except ImportError as e:
    print(f'❌ GeoPandas: NON INSTALLÉ - {e}')

try:
    from pyproj import CRS
    crs = CRS.from_epsg(32630)
    print(f'✅ PyProj: {crs.name}')
except Exception as e:
    print(f'❌ PyProj: ERREUR - {e}')

try:
    import redis
    print(f'✅ Redis module: OK')
except ImportError:
    print(f'⚠️  Redis module: NON INSTALLÉ')

try:
    import shapely
    print(f'✅ Shapely: {shapely.__version__}')
except ImportError:
    print(f'❌ Shapely: NON INSTALLÉ')

try:
    from pydantic import BaseModel
    print(f'✅ Pydantic: OK')
except ImportError:
    print(f'❌ Pydantic: NON INSTALLÉ')
"

# Vérifier que Python3 fonctionne
if ! python3 --version; then
    echo "❌ Python3 non trouvé!"
    exit 1
fi

# Créer les répertoires nécessaires
echo "Création des répertoires..."
mkdir -p \
    /data/shapefiles \
    /data/csv \
    /data/geojson \
    /data/projects \
    /data/other \
    /data/tiles \
    /data/parcels \
    /data/documents \
    /var/log/qgis \
    /var/log/supervisor \
    /var/log/nginx \
    /var/log/flask \
    /var/run \
    /tmp/qgis

# Permissions
echo "Configuration des permissions..."
chown -R www-data:www-data /data /var/log/qgis /var/log/flask
chmod -R 755 /data

# Vérifier le projet QGIS
if [ ! -f "$QGIS_PROJECT_FILE" ]; then
    echo "⚠️  ATTENTION: Projet QGIS non trouvé: $QGIS_PROJECT_FILE"
    echo "Création d'un projet par défaut..."
    cat > "$QGIS_PROJECT_FILE" <<'QGIS_EOF'
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="Project EPSG:32630" version="3.28.0">
  <properties>
    <SpatialRefSys>
      <ProjectionsEnabled type="int">1</ProjectionsEnabled>
    </SpatialRefSys>
    <Gui>
      <SelectionColorRedPart type="int">255</SelectionColorRedPart>
      <SelectionColorGreenPart type="int">255</SelectionColorGreenPart>
      <SelectionColorBluePart type="int">0</SelectionColorBluePart>
      <SelectionColorAlphaPart type="int">255</SelectionColorAlphaPart>
    </Gui>
    <Measurement>
      <DistanceUnits type="QString">meters</DistanceUnits>
      <AreaUnits type="QString">m2</AreaUnits>
    </Measurement>
    <WMSServiceCapabilities>
      <WMSOnlineResource type="QString">http://localhost:10000</WMSOnlineResource>
    </WMSServiceCapabilities>
  </properties>
  <projectCrs>
    <spatialrefsys>
      <wkt>PROJCS["WGS 84 / UTM zone 30N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-3],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32630"]]</wkt>
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
  <layerorder/>
</qgis>
QGIS_EOF
    chmod 644 "$QGIS_PROJECT_FILE"
    echo "✅ Projet QGIS créé"
else
    echo "✅ Projet QGIS trouvé: $QGIS_PROJECT_FILE"
fi

# Vérifier api.py
if [ ! -f /app/api.py ]; then
    echo "❌ ERREUR: /app/api.py non trouvé!"
    exit 1
fi

# Test de syntaxe Python
echo "Vérification de la syntaxe de api.py..."
python3 -m py_compile /app/api.py || {
    echo "❌ Erreur de syntaxe dans api.py"
    exit 1
}
echo "✅ Syntaxe api.py OK"

# Configuration Nginx - Vérifier et corriger si nécessaire
echo "Configuration de Nginx..."
nginx -t 2>&1 | head -5 || {
    echo "⚠️  Erreur configuration Nginx, recréation..."
    cat > /etc/nginx/sites-available/default <<'NGINX_EOF'
upstream flask_app {
    server 127.0.0.1:10000;
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 50M;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    location /api/ {
        proxy_pass http://flask_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }

    location / {
        return 200 '{"status":"ok","message":"QGIS API - EPSG:32630","docs":"/api/docs"}';
        add_header Content-Type application/json;
    }
}
NGINX_EOF
    nginx -t || echo "⚠️  Configuration Nginx toujours en erreur"
}

# Affichage des informations
echo "=========================================="
echo "✅ Configuration terminée"
echo "=========================================="
echo "📐 CRS par défaut: $DEFAULT_CRS"
echo "📁 Répertoire données: $BASE_DIR"
echo "🗺️  Projet QGIS: $QGIS_PROJECT_FILE"
echo "🐍 Python: $(python3 --version)"
echo "🔧 API Script: /app/api.py"
echo "🌐 Port: $PORT"
echo "=========================================="

# Démarrage de supervisord
echo "🚀 Démarrage de supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf