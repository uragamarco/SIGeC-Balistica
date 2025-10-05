#!/usr/bin/env python3
"""
Sistema de CI/CD y Despliegue Automatizado para SEACABAr.
Proporciona pipelines automatizados, contenedorización y despliegue continuo.
"""

import os
import json
import yaml
import subprocess
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import asyncio
import aiofiles
import docker
import git
from jinja2 import Environment, FileSystemLoader, Template

# Configurar logging
logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Etapas del pipeline."""
    CHECKOUT = "checkout"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    PACKAGE = "package"
    DEPLOY_STAGING = "deploy_staging"
    INTEGRATION_TEST = "integration_test"
    DEPLOY_PRODUCTION = "deploy_production"
    CLEANUP = "cleanup"

class DeploymentEnvironment(Enum):
    """Entornos de despliegue."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class PipelineStatus(Enum):
    """Estados del pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

class ContainerPlatform(Enum):
    """Plataformas de contenedores."""
    DOCKER = "docker"
    PODMAN = "podman"
    KUBERNETES = "kubernetes"

@dataclass
class PipelineStep:
    """Paso del pipeline."""
    name: str
    stage: PipelineStage
    command: str
    working_dir: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300  # segundos
    retry_count: int = 0
    allow_failure: bool = False
    condition: Optional[str] = None  # Condición para ejecutar
    artifacts: List[str] = field(default_factory=list)

@dataclass
class PipelineResult:
    """Resultado de ejecución del pipeline."""
    step_name: str
    stage: PipelineStage
    status: PipelineStatus
    start_time: float
    end_time: float
    duration: float
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    artifacts: List[str] = field(default_factory=list)
    error_message: str = ""

@dataclass
class DeploymentConfig:
    """Configuración de despliegue."""
    environment: DeploymentEnvironment
    target_hosts: List[str]
    container_registry: str = ""
    image_name: str = ""
    image_tag: str = "latest"
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    health_check: Dict[str, Any] = field(default_factory=dict)
    rollback_enabled: bool = True

@dataclass
class CICDConfig:
    """Configuración del sistema CI/CD."""
    project_name: str = "SEACABAr"
    repository_url: str = ""
    branch: str = "main"
    workspace_dir: str = "/tmp/cicd_workspace"
    artifacts_dir: str = "artifacts"
    container_platform: ContainerPlatform = ContainerPlatform.DOCKER
    registry_url: str = ""
    registry_username: str = ""
    registry_password: str = ""
    notification_webhook: str = ""
    parallel_jobs: int = 2
    cleanup_after_days: int = 7

class CommandExecutor:
    """Ejecutor de comandos."""
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir)
    
    async def execute(self, command: str, timeout: int = 300,
                     environment: Dict[str, str] = None) -> Tuple[int, str, str]:
        """Ejecutar comando de forma asíncrona."""
        
        env = os.environ.copy()
        if environment:
            env.update(environment)
        
        logger.info(f"Ejecutando: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir),
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            return (
                process.returncode,
                stdout.decode('utf-8', errors='ignore'),
                stderr.decode('utf-8', errors='ignore')
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Comando timeout después de {timeout}s: {command}")
            return -1, "", f"Timeout después de {timeout} segundos"
        
        except Exception as e:
            logger.error(f"Error ejecutando comando: {e}")
            return -1, "", str(e)

class ContainerManager:
    """Gestor de contenedores."""
    
    def __init__(self, platform: ContainerPlatform = ContainerPlatform.DOCKER):
        self.platform = platform
        
        if platform == ContainerPlatform.DOCKER:
            try:
                self.client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker no disponible: {e}")
                self.client = None
    
    async def build_image(self, dockerfile_path: str, image_name: str,
                         image_tag: str = "latest", build_args: Dict[str, str] = None) -> bool:
        """Construir imagen de contenedor."""
        
        if self.platform == ContainerPlatform.DOCKER:
            return await self._build_docker_image(
                dockerfile_path, image_name, image_tag, build_args
            )
        
        return False
    
    async def _build_docker_image(self, dockerfile_path: str, image_name: str,
                                 image_tag: str, build_args: Dict[str, str] = None) -> bool:
        """Construir imagen Docker."""
        
        if not self.client:
            logger.error("Cliente Docker no disponible")
            return False
        
        try:
            dockerfile_dir = Path(dockerfile_path).parent
            full_image_name = f"{image_name}:{image_tag}"
            
            logger.info(f"Construyendo imagen Docker: {full_image_name}")
            
            # Construir imagen
            image, build_logs = self.client.images.build(
                path=str(dockerfile_dir),
                dockerfile=Path(dockerfile_path).name,
                tag=full_image_name,
                buildargs=build_args or {},
                rm=True,
                forcerm=True
            )
            
            # Log de construcción
            for log in build_logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            
            logger.info(f"Imagen construida exitosamente: {full_image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error construyendo imagen Docker: {e}")
            return False
    
    async def push_image(self, image_name: str, image_tag: str = "latest",
                        registry_url: str = "", username: str = "", 
                        password: str = "") -> bool:
        """Subir imagen al registry."""
        
        if self.platform == ContainerPlatform.DOCKER:
            return await self._push_docker_image(
                image_name, image_tag, registry_url, username, password
            )
        
        return False
    
    async def _push_docker_image(self, image_name: str, image_tag: str,
                                registry_url: str, username: str, password: str) -> bool:
        """Subir imagen Docker."""
        
        if not self.client:
            logger.error("Cliente Docker no disponible")
            return False
        
        try:
            # Autenticación si se proporciona
            if username and password:
                self.client.login(username=username, password=password, registry=registry_url)
            
            # Nombre completo de la imagen
            if registry_url:
                full_image_name = f"{registry_url}/{image_name}:{image_tag}"
            else:
                full_image_name = f"{image_name}:{image_tag}"
            
            logger.info(f"Subiendo imagen: {full_image_name}")
            
            # Subir imagen
            push_logs = self.client.images.push(
                repository=full_image_name,
                stream=True,
                decode=True
            )
            
            # Log de subida
            for log in push_logs:
                if 'status' in log:
                    logger.info(f"Push: {log['status']}")
            
            logger.info(f"Imagen subida exitosamente: {full_image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error subiendo imagen Docker: {e}")
            return False
    
    def create_dockerfile(self, output_path: str, base_image: str = "python:3.9-slim",
                         requirements_file: str = "requirements.txt") -> str:
        """Crear Dockerfile automático."""
        
        dockerfile_content = f'''# Dockerfile generado automáticamente para SEACABAr
FROM {base_image}

# Configurar directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY {requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Copiar código de la aplicación
COPY . .

# Crear usuario no-root
RUN useradd -m -u 1000 seacabar && chown -R seacabar:seacabar /app
USER seacabar

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Dockerfile creado: {output_path}")
        return output_path

class PipelineEngine:
    """Motor de ejecución de pipelines."""
    
    def __init__(self, config: CICDConfig):
        self.config = config
        self.executor = CommandExecutor()
        self.container_manager = ContainerManager(config.container_platform)
        self.results: List[PipelineResult] = []
        self.workspace = Path(config.workspace_dir)
        
        # Crear workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    async def execute_pipeline(self, steps: List[PipelineStep]) -> List[PipelineResult]:
        """Ejecutar pipeline completo."""
        
        logger.info(f"Iniciando pipeline con {len(steps)} pasos")
        self.results = []
        
        for step in steps:
            # Verificar condición
            if step.condition and not await self._evaluate_condition(step.condition):
                logger.info(f"Paso saltado por condición: {step.name}")
                continue
            
            # Ejecutar paso
            result = await self._execute_step(step)
            self.results.append(result)
            
            # Verificar si falló y no permite fallos
            if result.status == PipelineStatus.FAILED and not step.allow_failure:
                logger.error(f"Pipeline falló en paso: {step.name}")
                break
        
        # Resumen de resultados
        self._log_pipeline_summary()
        
        return self.results
    
    async def _execute_step(self, step: PipelineStep) -> PipelineResult:
        """Ejecutar paso individual."""
        
        logger.info(f"Ejecutando paso: {step.name} ({step.stage.value})")
        
        start_time = datetime.now().timestamp()
        
        # Configurar directorio de trabajo
        working_dir = self.workspace / step.working_dir
        working_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor.working_dir = working_dir
        
        # Intentos con retry
        for attempt in range(step.retry_count + 1):
            if attempt > 0:
                logger.info(f"Reintento {attempt} para paso: {step.name}")
            
            try:
                # Ejecutar comando
                exit_code, stdout, stderr = await self.executor.execute(
                    step.command, step.timeout, step.environment
                )
                
                end_time = datetime.now().timestamp()
                duration = end_time - start_time
                
                # Determinar estado
                status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.FAILED
                
                # Procesar artefactos
                artifacts = await self._collect_artifacts(step.artifacts, working_dir)
                
                result = PipelineResult(
                    step_name=step.name,
                    stage=step.stage,
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    artifacts=artifacts
                )
                
                if status == PipelineStatus.SUCCESS or attempt == step.retry_count:
                    return result
                
            except Exception as e:
                logger.error(f"Error ejecutando paso {step.name}: {e}")
                
                end_time = datetime.now().timestamp()
                duration = end_time - start_time
                
                return PipelineResult(
                    step_name=step.name,
                    stage=step.stage,
                    status=PipelineStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    error_message=str(e)
                )
        
        # No debería llegar aquí
        return PipelineResult(
            step_name=step.name,
            stage=step.stage,
            status=PipelineStatus.FAILED,
            start_time=start_time,
            end_time=datetime.now().timestamp(),
            duration=0,
            error_message="Error desconocido"
        )
    
    async def _evaluate_condition(self, condition: str) -> bool:
        """Evaluar condición para ejecutar paso."""
        
        # Implementación simplificada - en producción usar parser más robusto
        try:
            # Condiciones básicas soportadas
            if condition == "always":
                return True
            elif condition == "never":
                return False
            elif condition.startswith("branch=="):
                target_branch = condition.split("==")[1].strip('"\'')
                return self.config.branch == target_branch
            elif condition.startswith("env."):
                env_var = condition.split(".")[1]
                return env_var in os.environ
            
            return True
            
        except Exception as e:
            logger.warning(f"Error evaluando condición '{condition}': {e}")
            return True
    
    async def _collect_artifacts(self, artifact_patterns: List[str], 
                                working_dir: Path) -> List[str]:
        """Recolectar artefactos."""
        
        artifacts = []
        artifacts_dir = self.workspace / self.config.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        for pattern in artifact_patterns:
            try:
                # Buscar archivos que coincidan con el patrón
                matching_files = list(working_dir.glob(pattern))
                
                for file_path in matching_files:
                    if file_path.is_file():
                        # Copiar a directorio de artefactos
                        artifact_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
                        artifact_path = artifacts_dir / artifact_name
                        
                        shutil.copy2(file_path, artifact_path)
                        artifacts.append(str(artifact_path))
                        
                        logger.info(f"Artefacto recolectado: {artifact_path}")
                
            except Exception as e:
                logger.warning(f"Error recolectando artefactos '{pattern}': {e}")
        
        return artifacts
    
    def _log_pipeline_summary(self):
        """Log resumen del pipeline."""
        
        total_steps = len(self.results)
        successful_steps = sum(1 for r in self.results if r.status == PipelineStatus.SUCCESS)
        failed_steps = sum(1 for r in self.results if r.status == PipelineStatus.FAILED)
        total_duration = sum(r.duration for r in self.results)
        
        logger.info(f"Pipeline completado:")
        logger.info(f"  Total pasos: {total_steps}")
        logger.info(f"  Exitosos: {successful_steps}")
        logger.info(f"  Fallidos: {failed_steps}")
        logger.info(f"  Duración total: {total_duration:.2f}s")
        
        # Log detalles de pasos fallidos
        for result in self.results:
            if result.status == PipelineStatus.FAILED:
                logger.error(f"  FALLO - {result.step_name}: {result.error_message}")

class DeploymentManager:
    """Gestor de despliegues."""
    
    def __init__(self, config: CICDConfig):
        self.config = config
        self.container_manager = ContainerManager(config.container_platform)
        self.executor = CommandExecutor()
    
    async def deploy(self, deployment_config: DeploymentConfig) -> bool:
        """Realizar despliegue."""
        
        logger.info(f"Iniciando despliegue en {deployment_config.environment.value}")
        
        try:
            # Preparar configuración de despliegue
            await self._prepare_deployment_config(deployment_config)
            
            # Desplegar según el entorno
            if deployment_config.environment == DeploymentEnvironment.PRODUCTION:
                return await self._deploy_production(deployment_config)
            elif deployment_config.environment == DeploymentEnvironment.STAGING:
                return await self._deploy_staging(deployment_config)
            else:
                return await self._deploy_development(deployment_config)
                
        except Exception as e:
            logger.error(f"Error en despliegue: {e}")
            return False
    
    async def _prepare_deployment_config(self, deployment_config: DeploymentConfig):
        """Preparar configuración de despliegue."""
        
        # Crear archivos de configuración
        config_dir = self.config.workspace_dir / "deployment_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Docker Compose
        await self._create_docker_compose(deployment_config, config_dir)
        
        # Kubernetes manifests
        await self._create_kubernetes_manifests(deployment_config, config_dir)
        
        # Scripts de despliegue
        await self._create_deployment_scripts(deployment_config, config_dir)
    
    async def _create_docker_compose(self, deployment_config: DeploymentConfig, 
                                   config_dir: Path):
        """Crear archivo docker-compose.yml."""
        
        compose_config = {
            'version': '3.8',
            'services': {
                'seacabar': {
                    'image': f"{deployment_config.image_name}:{deployment_config.image_tag}",
                    'ports': ['8000:8000'],
                    'environment': deployment_config.environment_variables,
                    'volumes': [
                        f"{vol['host']}:{vol['container']}" 
                        for vol in deployment_config.volumes
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': deployment_config.health_check.get(
                            'test', ['CMD', 'curl', '-f', 'http://localhost:8000/health']
                        ),
                        'interval': deployment_config.health_check.get('interval', '30s'),
                        'timeout': deployment_config.health_check.get('timeout', '10s'),
                        'retries': deployment_config.health_check.get('retries', 3)
                    }
                }
            }
        }
        
        # Agregar recursos si se especifican
        if deployment_config.resources:
            compose_config['services']['seacabar']['deploy'] = {
                'resources': deployment_config.resources
            }
        
        compose_file = config_dir / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info(f"Docker Compose creado: {compose_file}")
    
    async def _create_kubernetes_manifests(self, deployment_config: DeploymentConfig,
                                         config_dir: Path):
        """Crear manifests de Kubernetes."""
        
        # Deployment
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'seacabar',
                'labels': {'app': 'seacabar'}
            },
            'spec': {
                'replicas': deployment_config.replicas,
                'selector': {'matchLabels': {'app': 'seacabar'}},
                'template': {
                    'metadata': {'labels': {'app': 'seacabar'}},
                    'spec': {
                        'containers': [{
                            'name': 'seacabar',
                            'image': f"{deployment_config.image_name}:{deployment_config.image_tag}",
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': k, 'value': v} 
                                for k, v in deployment_config.environment_variables.items()
                            ],
                            'resources': deployment_config.resources,
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/health', 'port': 8000},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'seacabar-service',
                'labels': {'app': 'seacabar'}
            },
            'spec': {
                'selector': {'app': 'seacabar'},
                'ports': [{'port': 80, 'targetPort': 8000}],
                'type': 'LoadBalancer'
            }
        }
        
        # Guardar manifests
        deployment_file = config_dir / "k8s-deployment.yaml"
        service_file = config_dir / "k8s-service.yaml"
        
        with open(deployment_file, 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        
        with open(service_file, 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        
        logger.info(f"Manifests K8s creados: {deployment_file}, {service_file}")
    
    async def _create_deployment_scripts(self, deployment_config: DeploymentConfig,
                                       config_dir: Path):
        """Crear scripts de despliegue."""
        
        # Script de despliegue Docker
        docker_script = f'''#!/bin/bash
set -e

echo "Iniciando despliegue Docker..."

# Detener contenedores existentes
docker-compose -f docker-compose.yml down || true

# Actualizar imagen
docker pull {deployment_config.image_name}:{deployment_config.image_tag}

# Iniciar servicios
docker-compose -f docker-compose.yml up -d

# Verificar salud
sleep 10
curl -f http://localhost:8000/health || exit 1

echo "Despliegue Docker completado exitosamente"
'''
        
        # Script de despliegue Kubernetes
        k8s_script = f'''#!/bin/bash
set -e

echo "Iniciando despliegue Kubernetes..."

# Aplicar manifests
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml

# Esperar rollout
kubectl rollout status deployment/seacabar

# Verificar pods
kubectl get pods -l app=seacabar

echo "Despliegue Kubernetes completado exitosamente"
'''
        
        # Guardar scripts
        docker_script_file = config_dir / "deploy-docker.sh"
        k8s_script_file = config_dir / "deploy-k8s.sh"
        
        with open(docker_script_file, 'w') as f:
            f.write(docker_script)
        
        with open(k8s_script_file, 'w') as f:
            f.write(k8s_script)
        
        # Hacer ejecutables
        os.chmod(docker_script_file, 0o755)
        os.chmod(k8s_script_file, 0o755)
        
        logger.info(f"Scripts de despliegue creados: {docker_script_file}, {k8s_script_file}")
    
    async def _deploy_production(self, deployment_config: DeploymentConfig) -> bool:
        """Despliegue en producción."""
        
        logger.info("Ejecutando despliegue en producción")
        
        try:
            # Validaciones adicionales para producción
            if not await self._validate_production_deployment(deployment_config):
                return False
            
            # Backup antes del despliegue
            await self._create_backup()
            
            # Despliegue con rolling update
            success = await self._rolling_deployment(deployment_config)
            
            if not success and deployment_config.rollback_enabled:
                logger.warning("Despliegue falló, iniciando rollback")
                await self._rollback_deployment()
            
            return success
            
        except Exception as e:
            logger.error(f"Error en despliegue de producción: {e}")
            return False
    
    async def _deploy_staging(self, deployment_config: DeploymentConfig) -> bool:
        """Despliegue en staging."""
        
        logger.info("Ejecutando despliegue en staging")
        
        try:
            config_dir = Path(self.config.workspace_dir) / "deployment_configs"
            
            # Usar Docker Compose para staging
            exit_code, stdout, stderr = await self.executor.execute(
                f"cd {config_dir} && ./deploy-docker.sh",
                timeout=600
            )
            
            if exit_code == 0:
                logger.info("Despliegue en staging exitoso")
                return True
            else:
                logger.error(f"Error en despliegue staging: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error en despliegue de staging: {e}")
            return False
    
    async def _deploy_development(self, deployment_config: DeploymentConfig) -> bool:
        """Despliegue en desarrollo."""
        
        logger.info("Ejecutando despliegue en desarrollo")
        
        try:
            # Despliegue simple para desarrollo
            exit_code, stdout, stderr = await self.executor.execute(
                f"docker run -d -p 8000:8000 --name seacabar-dev "
                f"{deployment_config.image_name}:{deployment_config.image_tag}",
                timeout=300
            )
            
            if exit_code == 0:
                logger.info("Despliegue en desarrollo exitoso")
                return True
            else:
                logger.error(f"Error en despliegue desarrollo: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error en despliegue de desarrollo: {e}")
            return False
    
    async def _validate_production_deployment(self, deployment_config: DeploymentConfig) -> bool:
        """Validar despliegue de producción."""
        
        # Verificar que la imagen existe
        if not await self._image_exists(deployment_config.image_name, deployment_config.image_tag):
            logger.error("Imagen no encontrada en registry")
            return False
        
        # Verificar configuración de recursos
        if not deployment_config.resources:
            logger.warning("No se especificaron recursos para producción")
        
        # Verificar health check
        if not deployment_config.health_check:
            logger.warning("No se especificó health check para producción")
        
        return True
    
    async def _image_exists(self, image_name: str, image_tag: str) -> bool:
        """Verificar si la imagen existe."""
        
        try:
            if self.container_manager.client:
                self.container_manager.client.images.get(f"{image_name}:{image_tag}")
                return True
        except Exception:
            pass
        
        return False
    
    async def _create_backup(self):
        """Crear backup antes del despliegue."""
        
        logger.info("Creando backup pre-despliegue")
        # Implementar lógica de backup específica
        pass
    
    async def _rolling_deployment(self, deployment_config: DeploymentConfig) -> bool:
        """Realizar rolling deployment."""
        
        logger.info("Ejecutando rolling deployment")
        
        try:
            config_dir = Path(self.config.workspace_dir) / "deployment_configs"
            
            # Usar Kubernetes para rolling deployment
            exit_code, stdout, stderr = await self.executor.execute(
                f"cd {config_dir} && ./deploy-k8s.sh",
                timeout=900
            )
            
            return exit_code == 0
            
        except Exception as e:
            logger.error(f"Error en rolling deployment: {e}")
            return False
    
    async def _rollback_deployment(self):
        """Rollback del despliegue."""
        
        logger.info("Ejecutando rollback")
        
        try:
            # Rollback usando Kubernetes
            exit_code, stdout, stderr = await self.executor.execute(
                "kubectl rollout undo deployment/seacabar",
                timeout=300
            )
            
            if exit_code == 0:
                logger.info("Rollback completado exitosamente")
            else:
                logger.error(f"Error en rollback: {stderr}")
                
        except Exception as e:
            logger.error(f"Error ejecutando rollback: {e}")

class CICDSystem:
    """Sistema principal de CI/CD."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializar sistema CI/CD."""
        
        config_dict = config or {}
        self.config = CICDConfig(**config_dict)
        self.pipeline_engine = PipelineEngine(self.config)
        self.deployment_manager = DeploymentManager(self.config)
        self.container_manager = ContainerManager(self.config.container_platform)
        
        logger.info("Sistema CI/CD inicializado")
    
    def create_default_pipeline(self) -> List[PipelineStep]:
        """Crear pipeline por defecto."""
        
        steps = [
            # Checkout
            PipelineStep(
                name="checkout",
                stage=PipelineStage.CHECKOUT,
                command=f"git clone {self.config.repository_url} . && git checkout {self.config.branch}",
                timeout=300
            ),
            
            # Instalar dependencias
            PipelineStep(
                name="install_dependencies",
                stage=PipelineStage.BUILD,
                command="pip install -r requirements.txt",
                timeout=600
            ),
            
            # Ejecutar tests
            PipelineStep(
                name="run_tests",
                stage=PipelineStage.TEST,
                command="python -m pytest tests/ -v --tb=short",
                timeout=900,
                artifacts=["test-results.xml", "coverage.xml"]
            ),
            
            # Análisis de seguridad
            PipelineStep(
                name="security_scan",
                stage=PipelineStage.SECURITY_SCAN,
                command="bandit -r . -f json -o security-report.json || true",
                timeout=300,
                artifacts=["security-report.json"],
                allow_failure=True
            ),
            
            # Construir imagen Docker
            PipelineStep(
                name="build_image",
                stage=PipelineStage.PACKAGE,
                command=f"docker build -t {self.config.project_name.lower()}:latest .",
                timeout=1200
            ),
            
            # Subir imagen
            PipelineStep(
                name="push_image",
                stage=PipelineStage.PACKAGE,
                command=f"docker push {self.config.registry_url}/{self.config.project_name.lower()}:latest",
                timeout=600,
                condition='branch=="main"'
            )
        ]
        
        return steps
    
    async def run_pipeline(self, steps: List[PipelineStep] = None) -> List[PipelineResult]:
        """Ejecutar pipeline."""
        
        if steps is None:
            steps = self.create_default_pipeline()
        
        logger.info("Iniciando ejecución de pipeline CI/CD")
        
        # Preparar workspace
        await self._prepare_workspace()
        
        # Crear Dockerfile si no existe
        dockerfile_path = Path(self.config.workspace_dir) / "Dockerfile"
        if not dockerfile_path.exists():
            self.container_manager.create_dockerfile(str(dockerfile_path))
        
        # Ejecutar pipeline
        results = await self.pipeline_engine.execute_pipeline(steps)
        
        # Generar reporte
        await self._generate_pipeline_report(results)
        
        return results
    
    async def deploy_application(self, environment: DeploymentEnvironment,
                               image_tag: str = "latest") -> bool:
        """Desplegar aplicación."""
        
        deployment_config = DeploymentConfig(
            environment=environment,
            target_hosts=["localhost"],
            container_registry=self.config.registry_url,
            image_name=f"{self.config.registry_url}/{self.config.project_name.lower()}",
            image_tag=image_tag,
            replicas=1 if environment == DeploymentEnvironment.DEVELOPMENT else 3,
            environment_variables={
                "ENV": environment.value,
                "PROJECT_NAME": self.config.project_name
            },
            health_check={
                "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            },
            rollback_enabled=environment == DeploymentEnvironment.PRODUCTION
        )
        
        return await self.deployment_manager.deploy(deployment_config)
    
    async def _prepare_workspace(self):
        """Preparar workspace."""
        
        workspace = Path(self.config.workspace_dir)
        
        # Limpiar workspace anterior si existe
        if workspace.exists():
            shutil.rmtree(workspace)
        
        # Crear nuevo workspace
        workspace.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Workspace preparado: {workspace}")
    
    async def _generate_pipeline_report(self, results: List[PipelineResult]):
        """Generar reporte del pipeline."""
        
        report_data = {
            "pipeline_execution": {
                "timestamp": datetime.now().isoformat(),
                "project": self.config.project_name,
                "branch": self.config.branch,
                "total_steps": len(results),
                "successful_steps": sum(1 for r in results if r.status == PipelineStatus.SUCCESS),
                "failed_steps": sum(1 for r in results if r.status == PipelineStatus.FAILED),
                "total_duration": sum(r.duration for r in results)
            },
            "steps": [
                {
                    "name": r.step_name,
                    "stage": r.stage.value,
                    "status": r.status.value,
                    "duration": r.duration,
                    "exit_code": r.exit_code,
                    "artifacts": r.artifacts,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }
        
        # Guardar reporte
        report_file = Path(self.config.workspace_dir) / "pipeline-report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Reporte de pipeline generado: {report_file}")
    
    def create_github_actions_workflow(self) -> str:
        """Crear workflow de GitHub Actions."""
        
        workflow_content = f'''name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --tb=short
    
    - name: Security scan
      run: |
        pip install bandit
        bandit -r . -f json -o security-report.json || true
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: |
          test-results.xml
          coverage.xml
          security-report.json

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{{{ secrets.REGISTRY_URL }}}}
        username: ${{{{ secrets.REGISTRY_USERNAME }}}}
        password: ${{{{ secrets.REGISTRY_PASSWORD }}}}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{{{ secrets.REGISTRY_URL }}}}/{self.config.project_name.lower()}:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Agregar comandos de despliegue específicos
'''
        
        workflow_file = Path(self.config.workspace_dir) / ".github/workflows/cicd.yml"
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(workflow_file, 'w') as f:
            f.write(workflow_content)
        
        logger.info(f"Workflow de GitHub Actions creado: {workflow_file}")
        return str(workflow_file)
    
    def create_gitlab_ci_config(self) -> str:
        """Crear configuración de GitLab CI."""
        
        gitlab_ci_content = f'''stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

before_script:
  - python -m pip install --upgrade pip

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - python -m pytest tests/ -v --tb=short
    - bandit -r . -f json -o security-report.json || true
  artifacts:
    reports:
      junit: test-results.xml
    paths:
      - security-report.json
      - coverage.xml
    expire_in: 1 week

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $REGISTRY_USERNAME -p $REGISTRY_PASSWORD $REGISTRY_URL
    - docker build -t $REGISTRY_URL/{self.config.project_name.lower()}:latest .
    - docker push $REGISTRY_URL/{self.config.project_name.lower()}:latest
  only:
    - main

deploy_staging:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to staging"
    - apk add --no-cache curl
    - curl -f http://staging.example.com/health || exit 1
  environment:
    name: staging
    url: http://staging.example.com
  only:
    - main

deploy_production:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to production"
    - apk add --no-cache curl
    - curl -f http://production.example.com/health || exit 1
  environment:
    name: production
    url: http://production.example.com
  when: manual
  only:
    - main
'''
        
        gitlab_ci_file = Path(self.config.workspace_dir) / ".gitlab-ci.yml"
        
        with open(gitlab_ci_file, 'w') as f:
            f.write(gitlab_ci_content)
        
        logger.info(f"Configuración de GitLab CI creada: {gitlab_ci_file}")
        return str(gitlab_ci_file)

# Instancia global
_cicd_system: Optional[CICDSystem] = None

def get_cicd_system() -> CICDSystem:
    """Obtener instancia global del sistema CI/CD."""
    global _cicd_system
    if _cicd_system is None:
        _cicd_system = CICDSystem()
    return _cicd_system

def initialize_cicd_system(config: Dict[str, Any] = None) -> CICDSystem:
    """Inicializar sistema CI/CD."""
    global _cicd_system
    _cicd_system = CICDSystem(config)
    return _cicd_system

async def run_cicd_pipeline(repository_url: str = "", branch: str = "main",
                           workspace_dir: str = "/tmp/cicd_workspace") -> List[PipelineResult]:
    """Función de conveniencia para ejecutar pipeline CI/CD."""
    
    config = {
        'repository_url': repository_url,
        'branch': branch,
        'workspace_dir': workspace_dir,
        'project_name': 'SEACABAr',
        'container_platform': ContainerPlatform.DOCKER
    }
    
    cicd_system = initialize_cicd_system(config)
    return await cicd_system.run_pipeline()

if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema CI/CD SEACABAr")
    parser.add_argument("--action", choices=["pipeline", "deploy", "github", "gitlab"], 
                       default="pipeline", help="Acción a ejecutar")
    parser.add_argument("--repo", help="URL del repositorio")
    parser.add_argument("--branch", default="main", help="Rama a usar")
    parser.add_argument("--env", choices=["development", "staging", "production"], 
                       default="development", help="Entorno de despliegue")
    parser.add_argument("--workspace", default="/tmp/cicd_workspace", help="Directorio de trabajo")
    
    args = parser.parse_args()
    
    # Configurar sistema
    config = {
        'repository_url': args.repo or "",
        'branch': args.branch,
        'workspace_dir': args.workspace,
        'project_name': 'SEACABAr',
        'container_platform': ContainerPlatform.DOCKER,
        'registry_url': 'localhost:5000'
    }
    
    cicd_system = initialize_cicd_system(config)
    
    async def main():
        if args.action == "pipeline":
            print("Ejecutando pipeline CI/CD...")
            results = await cicd_system.run_pipeline()
            
            print(f"\nResultados del pipeline:")
            for result in results:
                status_icon = "✅" if result.status == PipelineStatus.SUCCESS else "❌"
                print(f"{status_icon} {result.step_name}: {result.status.value} ({result.duration:.2f}s)")
        
        elif args.action == "deploy":
            print(f"Desplegando en {args.env}...")
            env = DeploymentEnvironment(args.env)
            success = await cicd_system.deploy_application(env)
            
            if success:
                print("✅ Despliegue exitoso")
            else:
                print("❌ Despliegue falló")
        
        elif args.action == "github":
            print("Creando workflow de GitHub Actions...")
            workflow_file = cicd_system.create_github_actions_workflow()
            print(f"Workflow creado: {workflow_file}")
        
        elif args.action == "gitlab":
            print("Creando configuración de GitLab CI...")
            ci_file = cicd_system.create_gitlab_ci_config()
            print(f"Configuración creada: {ci_file}")
    
    # Ejecutar
    asyncio.run(main())