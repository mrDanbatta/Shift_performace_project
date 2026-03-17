import yaml
import os

from src.logger import configure_logger
logger = configure_logger()

def load_schema():
    """
    Loads the schema from specified YAML file.
    Returns:
        dict: The loaded schema as a dictionary.
    Raises:
        FileNotFoundError: If the schema file is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'schema.yml')
        )
        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
            logger.info("Schema loaded successfully.")
            return schema
    except Exception as e:
        logger.error(f"Schema file not found at path: {schema_path}")
        raise e
    
load_schema()
    