from __future__ import annotations
from typing import Dict, Any, List
import os
import requests
from dataclasses import dataclass
from tgs_project.logger import logger


# This is for development purposes only, to load environment variables from a .env file.
# In production, you should set these variables in your environment.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

class SolrError(Exception):
    """Custom exception for Solr errors."""
    pass
class SolrFieldError(SolrError):
    """Custom exception for Solr field errors."""
    pass
class SolrCoreError(SolrError):
    """Custom exception for Solr core errors."""
    pass
class SolrCollectionError(SolrError):
    """Custom exception for Solr collection errors."""
    pass
class SolrClientError(SolrError):
    """Custom exception for Solr client errors."""
    pass
class SolrFieldExistsError(SolrFieldError):
    """Custom exception for Solr field already exists errors."""
    pass
class SolrFieldNotFoundError(SolrFieldError):
    """Custom exception for Solr field not found errors."""
    pass
class SolrCoreExistsError(SolrCoreError):
    """Custom exception for Solr core already exists errors."""
    pass
class SolrCoreDocumentIdMissingError(SolrCoreError):
    """Custom exception for Solr core document ID missing errors."""
    pass
class SolrCoreNotFoundError(SolrCoreError):
    """Custom exception for Solr core not found errors."""
    pass

@dataclass
class SolrCore:
    name: str
    client: 'SolrClient'  # forward reference since SolrClient is defined later

    @property
    def base_url(self) -> str:
        return f"{self.client.url}/{self.name}"

    def schema_url(self) -> str:
        return f"{self.base_url}/schema"
    def status_url(self) -> str:
        return f"{self.base_url}/admin/cores?action=STATUS&core={self.name}"
    def custom_field_types_url(self) -> str:
        return f"{self.schema_url()}/fieldtypes"
    def copy_field_url(self) -> str:
        return f"{self.schema_url()}/copyfields"
    def update_url(self, commit: bool = True) -> str:
        return f"{self.base_url}/update{'?commit=true' if commit else ''}"
    def add_documents(
        self,
        docs: list[dict],
        commit: bool = True,
        atomic: bool = False
    ) -> requests.Response:
        url = self.update_url(commit=commit)
        headers = {"Content-Type": "application/json"}
        if not atomic:
            response = requests.post(url, json=docs, headers=headers)
        else:  # For atomic updates, we need to add the "set" to the values
            atomic_docs = []
            for doc in docs:
                if "id" not in doc:
                    msg = f"Document {doc} does not have an 'id' field."
                    logger.error(msg)
                    raise SolrCoreDocumentIdMissingError(msg)
                atomic_doc = {k: {"set": v} for k, v in doc.items() if v is not None and k != "id"}
                atomic_doc["id"] = doc["id"]  # Ensure the ID is included
                atomic_docs.append(atomic_doc)
            response = requests.post(url, json=atomic_docs, headers=headers)
        response.raise_for_status()
        return response
    def delete_all_documents(self) -> requests.Response:
        """Clear all documents in the Solr core."""
        url = self.update_url(commit=True)
        headers = {"Content-Type": "application/json"}
        payload = {"delete": {"query": "*:*"}}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response
class SolrCollection:
    def __init__(self):
        msg = "SolrCollection is not supported in standalone Solr mode."
        logger.error(msg)
        raise ImportError(msg)
    
class SolrSchema:
    def __init__(self, core: SolrCore):
        self.core = core
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.field_types: Dict[str, Dict[str, Any]] = {}
        self.copy_fields: Dict[tuple, Dict[str, Any]] = {}
        self.refresh_fields()

    def create_custom_field_type(
        self,
        custom_type_definition: Dict[str, Any],
        auto_commit: bool = True,
        raise_if_exists: bool = True
        ):
        """Create a custom field type in the Solr schema."""
        if 'name' not in custom_type_definition:
            msg = "Custom field type definition must include a 'name' key."
            logger.error(msg)
            raise ValueError(msg)
        if 'class' not in custom_type_definition:
            msg = "Custom field type definition must include a 'class' key."
            logger.error(msg)
            raise ValueError(msg)
        self.refresh_field_types()
        if custom_type_definition['name'] in self.field_types:
            if raise_if_exists:
                msg = f"Custom field type '{custom_type_definition['name']}' already exists."
                logger.error(msg)
                raise SolrFieldExistsError(msg)
            else:
                logger.info(f"Custom field type '{custom_type_definition['name']}' already exists in the schema.")
                return self.field_types[custom_type_definition['name']]
        
        try:
            if auto_commit:
                # Add the custom field type to the schema
                payload = {"add-field-type": custom_type_definition}
                response = requests.post(self.core.custom_field_types_url(), json=payload)
                response.raise_for_status()
                logger.info(f"Custom field type '{custom_type_definition['name']}' committed to Solr schema.")
            else:
                logger.info(f"Custom field type '{custom_type_definition['name']}' added but not committed.")
        except requests.RequestException as e:
            logger.error(f"Error creating custom field type '{custom_type_definition['name']}': {e}")
            raise SolrFieldError(f"Error creating custom field type: {e}")
        self.field_types[custom_type_definition['name']] = custom_type_definition
        return self.field_types[custom_type_definition['name']]
    def create_copy_fields(
        self,
        copy_field_pairs: list[tuple[str, str]],
        auto_commit: bool = True,
        raise_if_exists: bool = True
    ) -> list[tuple[str, str]]:
        """Create copy fields in the Solr schema from (source, dest) pairs."""
        self.refresh_copy_fields()
        self.refresh_fields()
        existing_pairs = list(self.copy_fields.keys())
        new_pairs = []

        # Make sure the source and destination fields exist
        for source, dest in copy_field_pairs:
            if source not in self.fields:
                msg = f"Source field '{source}' does not exist in the schema."
                logger.error(msg)
                raise SolrFieldNotFoundError(msg)
            if dest not in self.fields:
                msg = f"Destination field '{dest}' does not exist in the schema."
                logger.error(msg)
                raise SolrFieldNotFoundError(msg)
            if (source, dest) in existing_pairs:
                if raise_if_exists:
                    msg = f"Copy field '{source}' -> '{dest}' already exists in the schema."
                    logger.error(msg)
                    raise SolrFieldExistsError(msg)
                else:
                    logger.info(f"Copy field '{source}' -> '{dest}' already exists in the schema.")
            else:
                new_pairs.append((source, dest))

        if not new_pairs:
            if raise_if_exists:
                msg = f"All copy fields already exist: {copy_field_pairs}"
                logger.error(msg)
                raise SolrFieldExistsError(msg)
            else:
                logger.info(f"All copy fields already exist: {copy_field_pairs}")
                return copy_field_pairs

        payload = {
            "add-copy-field": [{"source": src, "dest": dst} for src, dst in new_pairs]
        }

        try:
            if auto_commit:
                response = requests.post(self.core.copy_field_url(), json=payload)
                response.raise_for_status()
                logger.info(f"Committed copy fields: {new_pairs}")
            else:
                logger.info(f"Prepared copy fields (not committed): {new_pairs}")
        except requests.RequestException as e:
            logger.error(f"Error creating copy fields: {e}")
            raise SolrFieldError(f"Error creating copy fields: {e}")

        # Update internal cache
        for pair in new_pairs:
            self.copy_fields[pair] = {"source": pair[0], "dest": pair[1]}

        return new_pairs    
    def commit_all_fields(self) -> Dict[str, Any]:
        for field in self.uncommitted_fields():
            logger.info(f"Committing field '{field['name']}' to Solr schema.")
            self.add_field(field, auto_commit=True)
        logger.info("All fields committed to Solr schema.")
        return self.fields
    
    def uncommitted_fields(self) -> List[Dict[str, Any]]:
        """Return a list of fields that have been added but not yet committed."""
        current_fields: List[Dict[str,str]] = self.list_fields()
        current_fields_set: set[str] = {field["name"] for field in current_fields}
        return [field for field in self.fields.values() if field["name"] not in current_fields_set]
    
    def add_field(
            self,
            field_def: Dict[str, Any],
            auto_commit: bool=True,
            raise_if_exists: bool=True
        ) -> Dict[str, Any]:
        """Add a field to the Solr schema."""
        if 'name' not in field_def:
            msg = "Field definition must include a 'name' key."
            logger.error(msg)
            raise ValueError(msg)
        if field_def['name'] in self.fields:
            if raise_if_exists:
                msg = f"Field '{field_def['name']}' already exists."
                logger.error(msg)
                raise SolrFieldExistsError(msg)
            else:
                logger.info(f"Field '{field_def['name']}' already exists in the schema.")
                return self.fields[field_def['name']]
        self.fields[field_def['name']] = field_def
        if auto_commit:
            # Automatically commit the field addition
            logger.info(f"Committing field '{field_def['name']}' to Solr schema.")
            return self.commit_field(field_def)
        else:
            logger.info(f"Field '{field_def['name']}' added to Solr schema but not committed.")
        return field_def
    
    def commit_field(self, field_def: Dict[str, Any]) -> Dict[str, Any]:
        """Add a field to the Solr schema."""
        payload = {"add-field": field_def}
        response = requests.post(self.core.schema_url(), json=payload)
        try:
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error committing field '{field_def['name']}' to Solr schema: {e}")
            raise

    def delete_field(self, field_name: str) -> Dict[str, Any]:
        """Delete a field from the Solr schema."""
        payload = {"delete-field": {"name": field_name}}
        response = requests.post(self.core.schema_url(), json=payload)
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error deleting field '{field_name}' from Solr schema: {e}")
            raise
        if field_name in self.fields:
            del self.fields[field_name]
        return response.json()

    def list_fields(self) -> List[Dict[str, Any]]:
        """List all fields in the schema."""
        response = requests.get(self.core.schema_url() + "/fields")
        try:
            response.raise_for_status()
            return response.json()['fields']
        except requests.RequestException as e:
            logger.error(f"Error listing fields from Solr schema: {e}")
            raise

    def refresh_fields(self) -> None:
        """Refresh field definitions from Solr."""
        response = requests.get(self.core.schema_url() + "/fields")
        try:
            response.raise_for_status()
            self.fields = {field["name"]: field for field in response.json()["fields"]}
        except requests.RequestException as e:
            logger.error(f"Error refreshing fields from Solr schema: {e}")
            raise
    def refresh_field_types(self) -> None:
        """Refresh field type definitions from Solr."""
        response = requests.get(self.core.custom_field_types_url())
        try:
            response.raise_for_status()
            self.field_types = {ftype["name"]: ftype for ftype in response.json()["fieldTypes"]}
        except requests.RequestException as e:
            logger.error(f"Error refreshing field types from Solr schema: {e}")
            raise
    def refresh_copy_fields(self) -> None:
        """Refresh copy field definitions from Solr."""
        try:
            response = requests.get(self.core.copy_field_url())
            response.raise_for_status()
            self.copy_fields = {
                (cf["source"], cf["dest"]): cf
                for cf in response.json()["copyFields"]
            }
        except requests.RequestException as e:
            logger.error(f"Error refreshing copy fields from Solr schema: {e}")
            raise

    def field_exists(self, name: str) -> bool:
        """Check if a field is already defined."""
        if not self.fields:
            self.refresh_fields()
        return name in self.fields

class SolrDocument:
    """Represents a document in a Solr core."""
    def __init__(self, collection: SolrCollection, id: str):
        self.collection = collection
        self.id = id

    def __repr__(self)-> str:
        """Return a string representation of the SolrDocument."""
        return f"SolrDocument(collection={self.collection}, id={self.id})"
    
    def __str__(self)-> str:
        """Return a user-friendly string representation of the SolrDocument."""
        return f"SolrDocument in {self.collection} with ID {self.id}"
    
class SolrClient:
    def __init__(self, url: str=''):
        """Initialize the SolrClient with a URL."""
        self.url = url if url else os.getenv("SOLR_URL", '')
        if not self.url:
            msg = "Solr url must be passed in or SOLR_URL environment variable is set."
            logger.error(msg)
            raise ValueError(msg)
        self.cores: Dict[str,SolrCore]= {}

        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            msg = "Solr URL must start with 'http://' or 'https://'"
            logger.error(msg)
            raise ValueError(msg)
        if not self.url.endswith("/"):
            self.url += "/"
        if not self.url.endswith("solr/"):
            self.url += "solr/"
        self.url = self.url.rstrip("/")  # Ensure no trailing slash for consistency
        # Check if the Solr server is reachable
        try:
            response = requests.get(self.url)
            response.raise_for_status()
        except requests.RequestException as e:
            msg = f"Could not connect to Solr server at {self.url}: {e}"
            logger.error(msg)
            raise ConnectionError(msg)
        # Initialize the Solr client
        self.refresh_cores()

        logger.info(f"SolrClient initialized with URL: {self.url}")

    def create_core(
        self,
        name: str,
        raise_if_exists: bool=True,
        delete_if_exists: bool=False
    ) -> SolrCore:
        """Create a new Solr core using the Solr REST API."""
        if raise_if_exists and delete_if_exists:
            msg = "Cannot set both raise_if_exists and delete_if_exists to True."
            logger.error(msg)
            raise SolrCoreError(msg)
        core_admin_url = f"{self.url}/admin/cores"
        params = {
            "action": "CREATE",
            "name": name,
            "instanceDir": name
        }
        logger.info(f"Core '{name}' created or already exists.")
        self.refresh_cores()
        if name in self.cores:
            if raise_if_exists:
                msg = f"Core '{name}' already exists."
                logger.error(msg)
                raise SolrCoreExistsError(msg)
            elif delete_if_exists:
                logger.info(f"Core '{name}' already exists, deleting it before creating a new one.")
                self.delete_core(name)
                logger.info(f'Core "{name}" deleted.... recreating it...')
        if name not in self.cores:
            logger.info(f"Core '{name}' created.")
            self.cores[name] = SolrCore(name=name, client=self)
        # Actually create the core again
        response = requests.get(core_admin_url, params=params)
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            msg = f"Error creating core '{name}': {e}"
            logger.error(msg)
            raise SolrCoreError(msg)
        self.cores[name] = SolrCore(name=name, client=self)
        logger.info(f"Core '{name}' deleted and recreated.")
        return self.cores[name]

    def list_cores(self) -> list[str]:
        """Fetches and returns a list of existing Solr core names."""
        response = requests.get(f"{self.url}/admin/cores", params={"action": "STATUS"})
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            msg = f"Error listing cores from Solr: {e}"
            logger.error(msg)
            raise
        data = response.json()
        core_names = list(data["status"].keys())
        
        for name in core_names:
            if name not in self.cores:
                self.cores[name] = SolrCore(name=name, client=self)
        return core_names
    
    def refresh_cores(self) -> None:
        """Refresh the list of cores."""
        response = requests.get(f"{self.url}/admin/cores", params={"action": "STATUS"})
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            msg = f"Error refreshing cores from Solr: {e}"
            logger.error(msg)
            raise
        data = response.json()
        core_names = list(data["status"].keys())
        
        for name in core_names:
            if name not in self.cores:
                self.cores[name] = SolrCore(name=name, client=self)

    def get_or_create_core(self, name: str) -> SolrCore:
        """Get a Solr core by name."""
        if not self.cores:
            self.refresh_cores()
        if name not in self.cores:
            return self.create_core(name)
        return self.cores[name]

    def get_core(self, name: str) -> SolrCore:
        """Get a Solr core by name."""
        if not self.cores:
            self.refresh_cores()
        if name not in self.cores:
            msg = f"Core '{name}' does not exist."
            logger.error(msg)
            raise SolrCoreExistsError(msg)
        return self.cores[name]

    def delete_core(self, name: str) -> None:
        """Delete a Solr core by name."""
        core_admin_url = f"{self.url}/admin/cores"
        params = {
            "action": "UNLOAD",
            "name": name,
            "deleteIndex": "true",
            "deleteDataDir": "true",
            "deleteInstanceDir": "true"
        }
        response = requests.get(core_admin_url, params=params)
        response.raise_for_status()
        if name in self.cores:
            del self.cores[name]
        logger.info(f"Core '{name}' deleted.")

    # The following methods are not supported in standalone Solr mode.
    # They are placeholders to maintain interface consistency.
    def create_collection(self, name: str)-> SolrCollection:
        """Create a new Solr collection if it does not exist."""
        raise NotImplementedError("SolrCollection is not supported in standalone Solr mode.")

    def get_collection(self, name: str) -> SolrCollection:
        raise NotImplementedError("SolrCollection is not supported in standalone Solr mode.")
    
    def delete_collection(self, name: str)-> None:
        """Delete a Solr collection."""
        raise NotImplementedError("SolrCollection is not supported in standalone Solr mode.")
    
