#!/usr/bin/env python3
"""
Sistema de Documentación Automática para SEACABAr.
Genera documentación técnica, guías de usuario y documentación de APIs automáticamente.
"""

import os
import ast
import json
import inspect
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import markdown
import jinja2
from jinja2 import Environment, FileSystemLoader, Template
import yaml

# Configurar logging
logger = logging.getLogger(__name__)

class DocumentationType(Enum):
    """Tipos de documentación."""
    API = "api"
    USER_GUIDE = "user_guide"
    TECHNICAL = "technical"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    CHANGELOG = "changelog"

class OutputFormat(Enum):
    """Formatos de salida."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"

@dataclass
class FunctionDoc:
    """Documentación de función."""
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    examples: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0

@dataclass
class ClassDoc:
    """Documentación de clase."""
    name: str
    docstring: str
    methods: List[FunctionDoc]
    attributes: List[Dict[str, Any]]
    inheritance: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0

@dataclass
class ModuleDoc:
    """Documentación de módulo."""
    name: str
    path: str
    docstring: str
    functions: List[FunctionDoc]
    classes: List[ClassDoc]
    imports: List[str] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class APIEndpointDoc:
    """Documentación de endpoint de API."""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class DocumentationConfig:
    """Configuración del sistema de documentación."""
    project_name: str = "SEACABAr"
    version: str = "1.0.0"
    description: str = "Sistema de Análisis de Contenido Audiovisual Basado en IA"
    author: str = "Equipo SEACABAr"
    output_dir: str = "docs/generated"
    template_dir: str = "docs/templates"
    include_private: bool = False
    include_source_links: bool = True
    generate_index: bool = True
    auto_examples: bool = True

class CodeAnalyzer:
    """Analizador de código para extraer documentación."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self._ast_cache: Dict[str, ast.AST] = {}
    
    def analyze_file(self, file_path: Path) -> ModuleDoc:
        """Analizar archivo Python y extraer documentación."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parsear AST
            tree = ast.parse(source)
            self._ast_cache[str(file_path)] = tree
            
            # Extraer información del módulo
            module_doc = ModuleDoc(
                name=file_path.stem,
                path=str(file_path),
                docstring=self._extract_docstring(tree),
                functions=[],
                classes=[],
                imports=self._extract_imports(tree),
                constants=self._extract_constants(tree)
            )
            
            # Analizar nodos del AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_') or self.config.include_private:
                        func_doc = self._analyze_function(node, source, file_path)
                        module_doc.functions.append(func_doc)
                
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith('_') or self.config.include_private:
                        class_doc = self._analyze_class(node, source, file_path)
                        module_doc.classes.append(class_doc)
            
            return module_doc
            
        except Exception as e:
            logger.error(f"Error analizando archivo {file_path}: {e}")
            return ModuleDoc(
                name=file_path.stem,
                path=str(file_path),
                docstring="",
                functions=[],
                classes=[]
            )
    
    def _analyze_function(self, node: ast.FunctionDef, source: str, file_path: Path) -> FunctionDoc:
        """Analizar función y extraer documentación."""
        
        # Extraer signature
        signature = self._extract_signature(node)
        
        # Extraer docstring
        docstring = self._extract_docstring(node)
        
        # Extraer parámetros
        parameters = self._extract_parameters(node, docstring)
        
        # Extraer tipo de retorno
        return_type = self._extract_return_type(node)
        
        # Extraer decoradores
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        # Extraer ejemplos del docstring
        examples = self._extract_examples_from_docstring(docstring)
        
        return FunctionDoc(
            name=node.name,
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            examples=examples,
            decorators=decorators,
            source_file=str(file_path),
            line_number=node.lineno
        )
    
    def _analyze_class(self, node: ast.ClassDef, source: str, file_path: Path) -> ClassDoc:
        """Analizar clase y extraer documentación."""
        
        # Extraer docstring
        docstring = self._extract_docstring(node)
        
        # Extraer métodos
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith('_') or self.config.include_private:
                    method_doc = self._analyze_function(item, source, file_path)
                    methods.append(method_doc)
        
        # Extraer atributos
        attributes = self._extract_class_attributes(node)
        
        # Extraer herencia
        inheritance = [self._get_base_name(base) for base in node.bases]
        
        return ClassDoc(
            name=node.name,
            docstring=docstring,
            methods=methods,
            attributes=attributes,
            inheritance=inheritance,
            source_file=str(file_path),
            line_number=node.lineno
        )
    
    def _extract_docstring(self, node: Union[ast.AST, ast.FunctionDef, ast.ClassDef]) -> str:
        """Extraer docstring de un nodo."""
        
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                return node.body[0].value.value.strip()
        
        return ""
    
    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extraer signature de función."""
        
        args = []
        
        # Argumentos posicionales
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Argumentos con valores por defecto
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                arg_index = len(args) - num_defaults + i
                if arg_index >= 0:
                    args[arg_index] += f" = {ast.unparse(default)}"
        
        # *args
        if node.args.vararg:
            vararg = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg)
        
        # **kwargs
        if node.args.kwarg:
            kwarg = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg)
        
        signature = f"{node.name}({', '.join(args)})"
        
        # Tipo de retorno
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
        
        return signature
    
    def _extract_parameters(self, node: ast.FunctionDef, docstring: str) -> List[Dict[str, Any]]:
        """Extraer información de parámetros."""
        
        parameters = []
        
        # Extraer de la signature
        for arg in node.args.args:
            param = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'description': '',
                'required': True,
                'default': None
            }
            parameters.append(param)
        
        # Agregar información de defaults
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                param_index = len(parameters) - num_defaults + i
                if param_index >= 0:
                    parameters[param_index]['required'] = False
                    parameters[param_index]['default'] = ast.unparse(default)
        
        # Intentar extraer descripciones del docstring
        # Implementación simplificada - en producción usar bibliotecas como sphinx
        lines = docstring.split('\n')
        for line in lines:
            line = line.strip()
            if ':param' in line or 'Args:' in line:
                # Parsear descripción de parámetros
                pass
        
        return parameters
    
    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extraer tipo de retorno."""
        
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extraer ejemplos del docstring."""
        
        examples = []
        lines = docstring.split('\n')
        in_example = False
        current_example = []
        
        for line in lines:
            line = line.strip()
            
            if 'Example' in line or 'Examples' in line or '>>>' in line:
                in_example = True
                if current_example:
                    examples.append('\n'.join(current_example))
                    current_example = []
            
            if in_example:
                if line.startswith('>>>') or line.startswith('...'):
                    current_example.append(line)
                elif line and not line.startswith(' '):
                    in_example = False
                    if current_example:
                        examples.append('\n'.join(current_example))
                        current_example = []
        
        if current_example:
            examples.append('\n'.join(current_example))
        
        return examples
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extraer imports del módulo."""
        
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extraer constantes del módulo."""
        
        constants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': ast.unparse(node.value),
                            'type': type(node.value).__name__
                        })
        
        return constants
    
    def _extract_class_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extraer atributos de clase."""
        
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': None,
                            'description': '',
                            'default': ast.unparse(item.value)
                        })
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attributes.append({
                    'name': item.target.id,
                    'type': ast.unparse(item.annotation),
                    'description': '',
                    'default': ast.unparse(item.value) if item.value else None
                })
        
        return attributes
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Obtener nombre del decorador."""
        
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            return ast.unparse(decorator.func)
        else:
            return ast.unparse(decorator)
    
    def _get_base_name(self, base: ast.expr) -> str:
        """Obtener nombre de clase base."""
        
        if isinstance(base, ast.Name):
            return base.id
        else:
            return ast.unparse(base)

class DocumentationGenerator:
    """Generador de documentación."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.analyzer = CodeAnalyzer(config)
        
        # Configurar Jinja2
        self.template_dir = Path(config.template_dir)
        self.output_dir = Path(config.output_dir)
        
        # Crear directorios si no existen
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar entorno de templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Crear templates por defecto si no existen
        self._create_default_templates()
    
    def generate_project_documentation(self, source_dir: Path) -> Dict[str, Any]:
        """Generar documentación completa del proyecto."""
        
        logger.info(f"Generando documentación para {source_dir}")
        
        # Analizar todos los archivos Python
        modules = []
        python_files = list(source_dir.rglob("*.py"))
        
        for file_path in python_files:
            if not any(part.startswith('.') for part in file_path.parts):
                module_doc = self.analyzer.analyze_file(file_path)
                modules.append(module_doc)
        
        # Generar documentación por tipo
        results = {}
        
        # Documentación técnica
        results['technical'] = self._generate_technical_docs(modules)
        
        # Documentación de API
        results['api'] = self._generate_api_docs(modules)
        
        # Guía de usuario
        results['user_guide'] = self._generate_user_guide(modules)
        
        # Documentación de referencia
        results['reference'] = self._generate_reference_docs(modules)
        
        # Índice general
        if self.config.generate_index:
            results['index'] = self._generate_index(modules, results)
        
        logger.info(f"Documentación generada en {self.output_dir}")
        return results
    
    def _generate_technical_docs(self, modules: List[ModuleDoc]) -> Dict[str, str]:
        """Generar documentación técnica."""
        
        template = self.jinja_env.get_template('technical.md.j2')
        
        docs = {}
        
        for module in modules:
            content = template.render(
                module=module,
                config=self.config,
                timestamp=datetime.now()
            )
            
            output_file = self.output_dir / f"technical_{module.name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            docs[module.name] = str(output_file)
        
        return docs
    
    def _generate_api_docs(self, modules: List[ModuleDoc]) -> Dict[str, str]:
        """Generar documentación de API."""
        
        template = self.jinja_env.get_template('api.md.j2')
        
        docs = {}
        
        # Filtrar módulos que contienen APIs
        api_modules = [
            module for module in modules
            if 'api' in module.name.lower() or any(
                'api' in func.name.lower() or any(
                    'route' in dec or 'endpoint' in dec
                    for dec in func.decorators
                ) for func in module.functions
            )
        ]
        
        for module in api_modules:
            content = template.render(
                module=module,
                config=self.config,
                timestamp=datetime.now()
            )
            
            output_file = self.output_dir / f"api_{module.name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            docs[module.name] = str(output_file)
        
        return docs
    
    def _generate_user_guide(self, modules: List[ModuleDoc]) -> str:
        """Generar guía de usuario."""
        
        template = self.jinja_env.get_template('user_guide.md.j2')
        
        # Extraer funciones principales y ejemplos
        main_functions = []
        for module in modules:
            for func in module.functions:
                if (not func.name.startswith('_') and 
                    func.examples and 
                    'main' in module.name.lower()):
                    main_functions.append(func)
        
        content = template.render(
            modules=modules,
            main_functions=main_functions,
            config=self.config,
            timestamp=datetime.now()
        )
        
        output_file = self.output_dir / "user_guide.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    def _generate_reference_docs(self, modules: List[ModuleDoc]) -> str:
        """Generar documentación de referencia."""
        
        template = self.jinja_env.get_template('reference.md.j2')
        
        content = template.render(
            modules=modules,
            config=self.config,
            timestamp=datetime.now()
        )
        
        output_file = self.output_dir / "reference.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    def _generate_index(self, modules: List[ModuleDoc], results: Dict[str, Any]) -> str:
        """Generar índice general."""
        
        template = self.jinja_env.get_template('index.md.j2')
        
        content = template.render(
            modules=modules,
            results=results,
            config=self.config,
            timestamp=datetime.now()
        )
        
        output_file = self.output_dir / "index.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    def _create_default_templates(self):
        """Crear templates por defecto."""
        
        templates = {
            'technical.md.j2': '''# {{ module.name }} - Documentación Técnica

{{ module.docstring }}

**Archivo:** `{{ module.path }}`  
**Generado:** {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Imports

{% for import in module.imports %}
- `{{ import }}`
{% endfor %}

## Constantes

{% for const in module.constants %}
### {{ const.name }}

**Tipo:** {{ const.type }}  
**Valor:** `{{ const.value }}`

{% endfor %}

## Funciones

{% for func in module.functions %}
### {{ func.name }}

```python
{{ func.signature }}
```

{{ func.docstring }}

{% if func.parameters %}
**Parámetros:**

{% for param in func.parameters %}
- `{{ param.name }}` ({{ param.type or 'Any' }}): {{ param.description }}
  {% if not param.required %}*Opcional, por defecto: `{{ param.default }}`*{% endif %}
{% endfor %}
{% endif %}

{% if func.return_type %}
**Retorna:** {{ func.return_type }}
{% endif %}

{% if func.examples %}
**Ejemplos:**

{% for example in func.examples %}
```python
{{ example }}
```
{% endfor %}
{% endif %}

{% if func.decorators %}
**Decoradores:** {{ func.decorators | join(', ') }}
{% endif %}

---

{% endfor %}

## Clases

{% for class in module.classes %}
### {{ class.name }}

{{ class.docstring }}

{% if class.inheritance %}
**Hereda de:** {{ class.inheritance | join(', ') }}
{% endif %}

{% if class.attributes %}
**Atributos:**

{% for attr in class.attributes %}
- `{{ attr.name }}` ({{ attr.type or 'Any' }}): {{ attr.description }}
  {% if attr.default %}*Por defecto: `{{ attr.default }}`*{% endif %}
{% endfor %}
{% endif %}

**Métodos:**

{% for method in class.methods %}
#### {{ method.name }}

```python
{{ method.signature }}
```

{{ method.docstring }}

{% if method.parameters %}
**Parámetros:**

{% for param in method.parameters %}
- `{{ param.name }}` ({{ param.type or 'Any' }}): {{ param.description }}
  {% if not param.required %}*Opcional, por defecto: `{{ param.default }}`*{% endif %}
{% endfor %}
{% endif %}

{% if method.examples %}
**Ejemplos:**

{% for example in method.examples %}
```python
{{ example }}
```
{% endfor %}
{% endif %}

---

{% endfor %}

{% endfor %}
''',
            
            'api.md.j2': '''# {{ module.name }} - Documentación de API

{{ module.docstring }}

**Archivo:** `{{ module.path }}`  
**Generado:** {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Endpoints

{% for func in module.functions %}
{% if func.decorators and ('route' in func.decorators | join(' ') or 'endpoint' in func.decorators | join(' ')) %}
### {{ func.name }}

```python
{{ func.signature }}
```

{{ func.docstring }}

**Decoradores:** {{ func.decorators | join(', ') }}

{% if func.parameters %}
**Parámetros:**

{% for param in func.parameters %}
- `{{ param.name }}` ({{ param.type or 'Any' }}): {{ param.description }}
  {% if not param.required %}*Opcional, por defecto: `{{ param.default }}`*{% endif %}
{% endfor %}
{% endif %}

{% if func.examples %}
**Ejemplos de uso:**

{% for example in func.examples %}
```python
{{ example }}
```
{% endfor %}
{% endif %}

---

{% endif %}
{% endfor %}
''',
            
            'user_guide.md.j2': '''# {{ config.project_name }} - Guía de Usuario

{{ config.description }}

**Versión:** {{ config.version }}  
**Autor:** {{ config.author }}  
**Generado:** {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Introducción

Esta guía te ayudará a comenzar a usar {{ config.project_name }}.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Básico

{% for func in main_functions %}
### {{ func.name }}

{{ func.docstring }}

{% if func.examples %}
**Ejemplo:**

{% for example in func.examples %}
```python
{{ example }}
```
{% endfor %}
{% endif %}

{% endfor %}

## Módulos Disponibles

{% for module in modules %}
### {{ module.name }}

{{ module.docstring }}

**Funciones principales:**
{% for func in module.functions[:3] %}
- `{{ func.name }}`: {{ func.docstring.split('.')[0] if func.docstring else 'Sin descripción' }}
{% endfor %}

{% endfor %}
''',
            
            'reference.md.j2': '''# {{ config.project_name }} - Referencia Completa

{{ config.description }}

**Versión:** {{ config.version }}  
**Generado:** {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Índice de Módulos

{% for module in modules %}
- [{{ module.name }}](#{{ module.name.lower().replace('_', '-') }})
{% endfor %}

{% for module in modules %}
## {{ module.name }}

{{ module.docstring }}

**Archivo:** `{{ module.path }}`

### Funciones

{% for func in module.functions %}
- [`{{ func.name }}`](#{{ func.name.lower().replace('_', '-') }}): {{ func.docstring.split('.')[0] if func.docstring else 'Sin descripción' }}
{% endfor %}

### Clases

{% for class in module.classes %}
- [`{{ class.name }}`](#{{ class.name.lower().replace('_', '-') }}): {{ class.docstring.split('.')[0] if class.docstring else 'Sin descripción' }}
{% endfor %}

---

{% endfor %}
''',
            
            'index.md.j2': '''# {{ config.project_name }} - Documentación

{{ config.description }}

**Versión:** {{ config.version }}  
**Autor:** {{ config.author }}  
**Generado:** {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}

## Documentación Disponible

### Guías

- [Guía de Usuario](user_guide.md) - Introducción y uso básico
- [Referencia Completa](reference.md) - Documentación completa de todos los módulos

### Documentación Técnica

{% for module_name, doc_path in results.technical.items() %}
- [{{ module_name }}]({{ doc_path | basename }}) - Documentación técnica detallada
{% endfor %}

### APIs

{% for module_name, doc_path in results.api.items() %}
- [{{ module_name }}]({{ doc_path | basename }}) - Documentación de API
{% endfor %}

## Estadísticas del Proyecto

- **Módulos:** {{ modules | length }}
- **Funciones totales:** {{ modules | sum(attribute='functions') | length }}
- **Clases totales:** {{ modules | sum(attribute='classes') | length }}

## Módulos Principales

{% for module in modules[:5] %}
### {{ module.name }}

{{ module.docstring.split('.')[0] if module.docstring else 'Sin descripción' }}

- **Funciones:** {{ module.functions | length }}
- **Clases:** {{ module.classes | length }}

{% endfor %}
'''
        }
        
        # Crear archivos de template
        for template_name, content in templates.items():
            template_file = self.template_dir / template_name
            if not template_file.exists():
                with open(template_file, 'w', encoding='utf-8') as f:
                    f.write(content)
    
    def generate_changelog(self, git_log: Optional[str] = None) -> str:
        """Generar changelog automático."""
        
        changelog_content = f"""# Changelog - {self.config.project_name}

Todos los cambios notables de este proyecto serán documentados en este archivo.

## [Unreleased]

### Agregado
- Sistema de documentación automática
- Generación de guías de usuario
- Documentación técnica detallada

### Cambiado
- Mejorada la estructura de documentación

### Corregido
- Correcciones menores en la documentación

---

*Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        output_file = self.output_dir / "CHANGELOG.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(changelog_content)
        
        return str(output_file)
    
    def export_to_json(self, modules: List[ModuleDoc]) -> str:
        """Exportar documentación a JSON."""
        
        data = {
            'project': {
                'name': self.config.project_name,
                'version': self.config.version,
                'description': self.config.description,
                'author': self.config.author,
                'generated_at': datetime.now().isoformat()
            },
            'modules': []
        }
        
        for module in modules:
            module_data = {
                'name': module.name,
                'path': module.path,
                'docstring': module.docstring,
                'imports': module.imports,
                'constants': module.constants,
                'functions': [
                    {
                        'name': func.name,
                        'signature': func.signature,
                        'docstring': func.docstring,
                        'parameters': func.parameters,
                        'return_type': func.return_type,
                        'examples': func.examples,
                        'decorators': func.decorators,
                        'line_number': func.line_number
                    }
                    for func in module.functions
                ],
                'classes': [
                    {
                        'name': cls.name,
                        'docstring': cls.docstring,
                        'attributes': cls.attributes,
                        'inheritance': cls.inheritance,
                        'methods': [
                            {
                                'name': method.name,
                                'signature': method.signature,
                                'docstring': method.docstring,
                                'parameters': method.parameters,
                                'return_type': method.return_type,
                                'examples': method.examples,
                                'decorators': method.decorators
                            }
                            for method in cls.methods
                        ],
                        'line_number': cls.line_number
                    }
                    for cls in module.classes
                ]
            }
            data['modules'].append(module_data)
        
        output_file = self.output_dir / "documentation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(output_file)

class DocumentationSystem:
    """Sistema principal de documentación."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializar sistema de documentación."""
        
        config_dict = config or {}
        self.config = DocumentationConfig(**config_dict)
        self.generator = DocumentationGenerator(self.config)
        
        logger.info("Sistema de documentación inicializado")
    
    def generate_full_documentation(self, source_dir: str = ".") -> Dict[str, Any]:
        """Generar documentación completa."""
        
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise ValueError(f"Directorio fuente no existe: {source_dir}")
        
        logger.info(f"Iniciando generación de documentación para {source_path}")
        
        # Generar documentación
        results = self.generator.generate_project_documentation(source_path)
        
        # Generar changelog
        changelog_path = self.generator.generate_changelog()
        results['changelog'] = changelog_path
        
        # Exportar a JSON
        modules = []
        for file_path in source_path.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                module_doc = self.generator.analyzer.analyze_file(file_path)
                modules.append(module_doc)
        
        json_path = self.generator.export_to_json(modules)
        results['json_export'] = json_path
        
        logger.info("Documentación generada exitosamente")
        return results
    
    def update_documentation(self, source_dir: str = ".") -> Dict[str, Any]:
        """Actualizar documentación existente."""
        
        return self.generate_full_documentation(source_dir)
    
    def get_documentation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de documentación."""
        
        output_dir = Path(self.config.output_dir)
        
        if not output_dir.exists():
            return {'error': 'No se ha generado documentación aún'}
        
        # Contar archivos generados
        md_files = list(output_dir.glob("*.md"))
        json_files = list(output_dir.glob("*.json"))
        
        # Calcular tamaños
        total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
        
        return {
            'output_directory': str(output_dir),
            'markdown_files': len(md_files),
            'json_files': len(json_files),
            'total_files': len(list(output_dir.iterdir())),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'last_generated': datetime.fromtimestamp(
                max(f.stat().st_mtime for f in output_dir.iterdir() if f.is_file())
            ).isoformat() if output_dir.iterdir() else None
        }

# Instancia global
_documentation_system: Optional[DocumentationSystem] = None

def get_documentation_system() -> DocumentationSystem:
    """Obtener instancia global del sistema de documentación."""
    global _documentation_system
    if _documentation_system is None:
        _documentation_system = DocumentationSystem()
    return _documentation_system

def initialize_documentation_system(config: Dict[str, Any] = None) -> DocumentationSystem:
    """Inicializar sistema de documentación."""
    global _documentation_system
    _documentation_system = DocumentationSystem(config)
    return _documentation_system

def generate_docs(source_dir: str = ".", output_dir: str = "docs/generated") -> Dict[str, Any]:
    """Función de conveniencia para generar documentación."""
    
    config = {
        'output_dir': output_dir,
        'include_private': False,
        'generate_index': True,
        'auto_examples': True
    }
    
    doc_system = initialize_documentation_system(config)
    return doc_system.generate_full_documentation(source_dir)

if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar documentación automática")
    parser.add_argument("--source", "-s", default=".", help="Directorio fuente")
    parser.add_argument("--output", "-o", default="docs/generated", help="Directorio de salida")
    parser.add_argument("--project", "-p", default="SEACABAr", help="Nombre del proyecto")
    parser.add_argument("--version", "-v", default="1.0.0", help="Versión del proyecto")
    parser.add_argument("--private", action="store_true", help="Incluir elementos privados")
    
    args = parser.parse_args()
    
    # Configurar sistema
    config = {
        'project_name': args.project,
        'version': args.version,
        'output_dir': args.output,
        'include_private': args.private,
        'generate_index': True,
        'auto_examples': True
    }
    
    # Generar documentación
    doc_system = initialize_documentation_system(config)
    results = doc_system.generate_full_documentation(args.source)
    
    print(f"Documentación generada en: {args.output}")
    print(f"Archivos generados: {len(results)}")
    
    # Mostrar estadísticas
    stats = doc_system.get_documentation_stats()
    print(f"Estadísticas: {json.dumps(stats, indent=2, default=str)}")