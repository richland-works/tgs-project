from typing import Dict, Any, List
import os
import requests
from dataclasses import dataclass


# This is for development purposes only, to load environment variables from a .env file.
# In production, you should set these variables in your environment.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
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

    def update_url(self, commit: bool = True) -> str:
        return f"{self.base_url}/update{'?commit=true' if commit else ''}"
    def add_documents(self, docs: list[dict], commit: bool = True) -> requests.Response:
        url = self.update_url(commit=commit)
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=docs, headers=headers)
        response.raise_for_status()
        return response
    
class SolrCollection:
    def __init__(self):
        raise ImportError("SolrCollection is not supported in standalone Solr mode.")
    
class SolrSchema:
    def __init__(self, core: SolrCore):
        self.core = core
        self.fields: Dict[str, Dict[str, Any]] = {}

    def commit_all_fields(self) -> Dict[str, Any]:
        for field in self.uncommitted_fields():
            print(f"Committing field '{field['name']}' to Solr schema.")
            self.add_field(field, auto_commit=True)
        print("All fields committed to Solr schema.")
        return self.fields
    
    def uncommitted_fields(self) -> List[Dict[str, Any]]:
        """Return a list of fields that have been added but not yet committed."""
        current_fields: List[Dict[str,str]] = self.list_fields()
        current_fields_set: set[str] = {field["name"] for field in current_fields}
        return [field for field in self.fields.values() if field["name"] not in current_fields_set]
    
    def add_field(self, field_def: Dict[str, Any], auto_commit: bool=True) -> Dict[str, Any]:
        """Add a field to the Solr schema."""
        if 'name' not in field_def:
            raise ValueError("Field definition must include a 'name' key.")
        if field_def['name'] in self.fields:
            raise ValueError(f"Field '{field_def['name']}' already exists in the schema.")
        self.fields[field_def['name']] = field_def
        if auto_commit:
            # Automatically commit the field addition
            print(f"Committing field '{field_def['name']}' to Solr schema.")
            return self.commit_field(field_def)
        else:
            print(f"Field '{field_def['name']}' added to Solr schema but not committed.")
        return field_def
    
    def commit_field(self, field_def: Dict[str, Any]) -> Dict[str, Any]:
        """Add a field to the Solr schema."""
        payload = {"add-field": field_def}
        response = requests.post(self.core.schema_url(), json=payload)
        response.raise_for_status()
        return response.json()

    def delete_field(self, field_name: str) -> Dict[str, Any]:
        """Delete a field from the Solr schema."""
        payload = {"delete-field": {"name": field_name}}
        response = requests.post(self.core.schema_url(), json=payload)
        response.raise_for_status()
        if field_name in self.fields:
            del self.fields[field_name]
        return response.json()

    def list_fields(self) -> List[Dict[str, Any]]:
        """List all fields in the schema."""
        response = requests.get(self.core.schema_url() + "/fields")
        response.raise_for_status()
        return response.json()['fields']
    
    def refresh_fields(self) -> None:
        """Refresh field definitions from Solr."""
        response = requests.get(self.core.schema_url() + "/fields")
        response.raise_for_status()
        self.fields = {field["name"]: field for field in response.json()["fields"]}

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
            raise ValueError("Solr url must be passed in or SOLR_URL environment variable is set.")
        self.cores: Dict[str,SolrCore]= {}

        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            raise ValueError("Solr URL must start with 'http://' or 'https://'")
        if not self.url.endswith("/"):
            self.url += "/"
        if not self.url.endswith("solr/"):
            self.url += "solr/"
        self.url = self.url.rstrip("/")  # Ensure no trailing slash for consistency
        print(f"SolrClient initialized with URL: {self.url}")

    def create_core(self, name: str) -> SolrCore:
        """Create a new Solr core using the Solr REST API."""
        core_admin_url = f"{self.url}/admin/cores"
        params = {
            "action": "CREATE",
            "name": name,
            "instanceDir": name
        }
        response = requests.get(core_admin_url, params=params)
        response.raise_for_status()
        print(f"Core '{name}' created or already exists.")
        self.cores[name] = SolrCore(name=name, client=self)
        return self.cores[name]

    def list_cores(self) -> list[str]:
        """Fetches and returns a list of existing Solr core names."""
        response = requests.get(f"{self.url}/admin/cores", params={"action": "STATUS"})
        response.raise_for_status()
        data = response.json()
        core_names = list(data["status"].keys())
        
        for name in core_names:
            if name not in self.cores:
                self.cores[name] = SolrCore(name=name, client=self)
        return core_names
    
    def get_core(self, name: str) -> SolrCore:
        """Get a Solr core by name."""
        if not self.cores:
            self.list_cores()
        if name not in self.cores:
            raise ValueError(f"Core '{name}' does not exist.")
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
        print(f"Core '{name}' deleted.")

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
    
