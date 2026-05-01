"""
Centralized multilingual messages for the MS-Feat project.
"""

from typing import Final

MESSAGES: Final = {
    "config_loaded": {
        "en": "Configuration loaded successfully.",
        "es": "Configuración cargada correctamente.",
    },
    "processor_ready": {
        "en": "Gait processor initialized and ready.",
        "es": "Procesador de marcha inicializado correctamente.",
    },
    "cli_ready": {
        "en": "CLI is operational.",
        "es": "CLI operativo.",
    },
    "running_no_plots": {
        "en": "Running without plot generation.",
        "es": "Ejecución sin generación de gráficos.",
    },
    "input_hdf5_path": {
        "en": "Input HDF5 path: {path}",
        "es": "Ruta del HDF5 de entrada: {path}",
    },
    "output_metrics_path": {
        "en": "Output metrics path: {path}",
        "es": "Ruta de métricas de salida: {path}",
    },
    "registry_csv_path": {
        "en": "Registry CSV path: {path}",
        "es": "Ruta del CSV de registro: {path}",
    },
    "selected_test_type": {
        "en": "Selected test type: {test_type}",
        "es": "Tipo de test seleccionado: {test_type}",
    },
    "output_hdf5_path": {
        "en": "Output HDF5 path: {path}",
        "es": "Ruta del HDF5 de salida: {path}",
    },
    "start_batch": {
    "en": "--- Starting batch extraction for test: {test} ---",
    "es": "--- Iniciando extracción para test: {test} ---",
    },
    "records_found": {
        "en": "Identified records: {n}",
        "es": "Registros identificados: {n}",
    },
    "save_ok": {
        "en": "[+] {id} persisted in HDF5 -> {h5_key}",
        "es": "[+] {id} persistido en HDF5 -> {h5_key}",
    },
    "no_data": {
        "en": "[-] {id}: Time range returned no records.",
        "es": "[-] {id}: El rango temporal no devolvió registros.",
    },
    "err_query": {
        "en": "[!] Flux query error for {id}: {e}",
        "es": "[!] Error en consulta Flux para {id}: {e}",
    },
    "err_schema": {
        "en": "[!] {id} rejected: Critical signals missing (S2/_time).",
        "es": "[!] {id} rechazado: Faltan señales críticas (S2/_time).",
    },
    "err_config": {
        "en": "InfluxDB configuration not found: {path}",
        "es": "Configuración de InfluxDB no hallada: {path}",
    },
    "csv_not_found": {
        "en": "Registry CSV not found: {path}",
        "es": "Registro CSV no encontrado: {path}",
    },
}


def get_message(key: str, lang: str = "en", **kwargs) -> str:
    """
    Retrieve a translated message and format it.

    Args:
        key: Message identifier.
        lang: Language code ("en" or "es").
        **kwargs: Formatting arguments.

    Returns:
        Formatted translated message.
    """
    template = MESSAGES.get(key, {}).get(lang) or MESSAGES.get(key, {}).get("en") or key
    return template.format(**kwargs)