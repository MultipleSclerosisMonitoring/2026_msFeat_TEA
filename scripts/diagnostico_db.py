import yaml
import os
import urllib3
from influxdb_client import InfluxDBClient

# Silenciar avisos de seguridad
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. Localización del archivo de configuración
base_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_path, '..', 'InfluxDBms', 'config_db.yaml')

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    client = InfluxDBClient(
        url=config['influxdb']['url'],
        token=config['influxdb']['token'],
        org=config['influxdb']['org'],
        verify_ssl=False
    )

    print(f"--- EXPLORANDO TABLA 'Gait' EN EL BUCKET: {config['influxdb']['bucket']} ---")

    # CONSULTA PARA VER LAS ETIQUETAS (TAGS) DISPONIBLES
    # Esto nos dirá si el ID del paciente se guarda en 'p_id', 'codeid', 'patient', etc.
    query_tags = f'import "influxdata/influxdb/schema" schema.measurementTagKeys(bucket: "{config["influxdb"]["bucket"]}", measurement: "Gait")'
    
    tag_keys = client.query_api().query(query_tags)
    
    print("\nEtiquetas encontradas en la medida 'Gait':")
    found_tags = []
    for table in tag_keys:
        for record in table.records:
            tag = record.get_value()
            if tag not in ['_start', '_stop', '_measurement', '_field']:
                found_tags.append(tag)
                print(f" -> {tag}")

    if found_tags:
        print("\n[!] INSTRUCCIÓN PARA TU EXTRACTOR:")
        print(f"En la query de batch_extractor.py, debes usar:")
        print(f'|> filter(fn: (r) => r["_measurement"] == "Gait")')
        print(f'|> filter(fn: (r) => r["{found_tags[0]}"] == "{{p_id}}")') # Usamos la primera etiqueta encontrada
    else:
        print("\n[!] No se han encontrado etiquetas de usuario. Es muy extraño.")

except Exception as e:
    print(f"Error en el diagnóstico: {e}")
finally:
    client.close()