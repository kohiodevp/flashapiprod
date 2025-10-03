#!/bin/bash
set -e

ALL_PARCELS="/data/parcels/all_parcels.geojson"
if [ ! -f "$ALL_PARCELS" ]; then
    echo "📝 Création de all_parcels.geojson vide..."
    python3 -c "
import geopandas as gpd
from shapely.geometry import Point
gdf = gpd.GeoDataFrame(
    columns=['id', 'name', 'commune', 'section', 'numero', 'superficie_m2', 'geometry'],
    crs='EPSG:32630'
)
gdf.to_file('$ALL_PARCELS', driver='GeoJSON')
"
    echo "✅ Fichier all_parcels.geojson créé."
else
    echo "📁 Fichier all_parcels.geojson déjà présent."
fi

PROJECT="/data/projects/default.qgs"
if [ ! -f "$PROJECT" ]; then
    echo "⚠️  Projet QGIS manquant ! Création d'un projet minimal..."
    cat > "$PROJECT" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE qgis-project>
<qgis version="3.40.6-Bratislava" projectname="default">
  <layer-tree-group expanded="1" checked="Qt::Checked">
    <layer-tree-layer expanded="1" checked="Qt::Checked" id="parcels" name="parcels"/>
  </layer-tree-group>
  <projectLayers>
    <layer type="vector" id="parcels" enabled="1">
      <source>/data/parcels/all_parcels.geojson</source>
      <provider>ogr</provider>
      <crs>
        <spatialrefsys>
          <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>
          <srsid>3452</srsid>
          <srid>32630</srid>
          <authid>EPSG:32630</authid>
        </spatialrefsys>
      </crs>
    </layer>
  </projectLayers>
  <properties>
    <WMSExtent>
      <value>100000,1300000,400000,1600000</value>
    </WMSExtent>
  </properties>
</qgis>
EOF
    echo "✅ Projet QGIS minimal créé."
fi

echo "🚀 Démarrage de l'API..."
exec "$@"