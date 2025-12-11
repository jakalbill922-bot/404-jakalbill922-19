import os
from io import BytesIO
import torch
from pathlib import Path
import sys
from pathlib import Path
from time import time
import logging
import asyncio
import gc
import base64
import yaml

from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State
from fastapi import Depends
from fastapi.responses import Response
from contextlib import asynccontextmanager
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from trellis.pipelines import TrellisImageTo3DPipeline

import math
from openai import OpenAI
from plyfile import PlyData, PlyElement

import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np
import open3d as o3d
import cv2
from rembg import remove
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    logger.warning(f"Configuration loaded from {config_path}")
except Exception as e:
    logger.error(f"Failed to load config: {e}. Using defaults.")
    # Fallback defaults
    CONFIG = {
        'image_analysis': {'edge_density_threshold': 0.08, 'entropy_threshold': 5.0},
        'complex_image': {
            'sparse_structure': {'steps': 25, 'cfg_strength': 7.5},
            'slat': {'steps': 22, 'cfg_strength': 3.0}
        },
        'simple_object': {
            'sparse_structure': {'steps': 15, 'cfg_strength': 6.0},
            'slat': {'steps': 17, 'cfg_strength': 2.0}
        },
        'refinement': {'enabled': True, 'min_points': 20, 'nb_neighbors': 20, 'std_ratio': 2.0},
        'generation': {'seed': 42, 'formats': ['gaussian'], 'timeout_seconds': 30},
        'performance': {'max_concurrent': 2}
    }

trellis_structure_step = 25
trellis_slat_step = 25

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.warning("Loading models...")
    
    # Store models in app state
    app.state.models = {}
    
    # Add request semaphore for concurrent request limiting
    max_concurrent = CONFIG.get('performance', {}).get('max_concurrent', 2)
    app.state.processing_semaphore = asyncio.Semaphore(max_concurrent)
    logger.warning(f"Max concurrent generations: {max_concurrent}")
    
    # Set device for all models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download LoRA checkpoint from Hugging Face
    logger.warning("Downloading LoRA checkpoint from Hugging Face...")
    try:
        # Using huggingface_hub to download from rayli/DSO-finetuned-TRELLIS
        ckpt_path = hf_hub_download(
            repo_id="rayli/DSO-finetuned-TRELLIS",
            filename="dpo-8000iters.safetensors",
            cache_dir="./model_cache"  # Local cache directory
        )
        logger.warning(f"LoRA checkpoint downloaded to: {ckpt_path}")
        
    except Exception as e:
        logger.error(f"Failed to download from Hugging Face: {e}")
        # Fallback to local path if download fails
        ckpt_path = "./DSO-finetuned-TRELLIS/dpo-8000iters.safetensors"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"LoRA checkpoint not found at {ckpt_path} and download failed")
        logger.warning(f"Using local checkpoint: {ckpt_path}")
    
    # Load Trellis pipeline
    logger.warning("Loading Trellis pipeline...")
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis_pipeline.to(device)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
    )
    trellis_pipeline.models["sparse_structure_flow_model"] = get_peft_model(trellis_pipeline.models["sparse_structure_flow_model"], peft_config)
    trellis_pipeline.models["sparse_structure_flow_model"].load_state_dict(load_file(ckpt_path))
    app.state.models["trellis_pipeline"] = trellis_pipeline
 
    logger.warning("Models loaded successfully!")
    
    yield
    
    # Shutdown
    logger.warning("Shutting down...")

# -----------------------------
# Environment and device
# -----------------------------

# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
device = "cuda" if torch.cuda.is_available() else "cpu"

using_qwen = False

app = FastAPI(
    title="3D Generation API", 
    description="API for generating 3D models from text prompts",
    lifespan=lifespan
)

# -----------------------------
# Helper Functions
# -----------------------------

async def cleanup_gpu_memory():
    """Clean up GPU memory after generation."""
    if torch.cuda.is_available():
        # Get memory info before cleanup
        before_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        before_cached = torch.cuda.memory_reserved() / 1024**3      # GB
        
        # Cleanup
        torch.cuda.empty_cache()  # Releases cached memory back to GPU
        gc.collect()              # Python garbage collection for CPU
        
        # Log the improvement
        after_allocated = torch.cuda.memory_allocated() / 1024**3   # GB  
        after_cached = torch.cuda.memory_reserved() / 1024**3       # GB
        
        freed_cache = before_cached - after_cached
        if freed_cache > 0.1:  # Only log if significant cleanup (>100MB)
            logger.warning(f"GPU cleanup freed {freed_cache:.2f}GB cached memory")

def decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 image data and return PIL Image."""
    try:
        logger.warning(f"Decoding base64 image data (length: {len(image_data)} bytes)")
        
        # Decode base64 data
        decoded_data = base64.b64decode(image_data)
        
        # Create PIL Image from decoded data
        image = Image.open(BytesIO(decoded_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.warning(f"Successfully decoded base64 image: {image.size}")
        return image
        
    except Exception as e:
        logger.warning(f"Error decoding base64 image data: {e}")
        raise

async def process_error(prompt) -> BytesIO:
    """Load and return empty.ply file as BytesIO buffer."""
    logger.warning("Could not find empty.ply, returning empty buffer")
    buffer = BytesIO()
    buffer.write(b"")
    buffer.seek(0)
    return buffer

def add_white_background(image: Image.Image) -> Image.Image:
    # Create white background with same size
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))

    # Composite the image onto white background
    return Image.alpha_composite(white_bg, image.convert("RGBA"))

def extract_simple_features(image):
    img_array = np.array(image)
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # brightness & contrast
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.astype(bool).mean()

    # entropy (rough texture measure)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    return dict(
        brightness=brightness,
        contrast=contrast,
        edge_density=edge_density,
        entropy=entropy,
    )

def choose_sampler_params(features):
    """Choose sampler parameters based on image complexity using config."""
    ed = features["edge_density"]
    ent = features["entropy"]
    
    # Get thresholds from config
    ed_threshold = CONFIG['image_analysis']['edge_density_threshold']
    ent_threshold = CONFIG['image_analysis']['entropy_threshold']
    
    # Determine if image is complex
    is_complex = ed > ed_threshold or ent > ent_threshold
    
    # Select appropriate config section
    if is_complex:
        config_section = CONFIG['complex_image']
        logger.warning(f"Complex image detected (edge_density={ed:.4f}, entropy={ent:.2f})")
    else:
        config_section = CONFIG['simple_object']
        logger.warning(f"Simple object detected (edge_density={ed:.4f}, entropy={ent:.2f})")
    
    # Extract parameters from config
    sparse_cfg = config_section['sparse_structure']
    slat_cfg = config_section['slat']
    
    return dict(
        sparse_structure_sampler_params={
            "steps": sparse_cfg['steps'],
            "cfg_strength": sparse_cfg['cfg_strength'],
        },
        slat_sampler_params={
            "steps": slat_cfg['steps'],
            "cfg_strength": slat_cfg['cfg_strength'],
        },
    )

def run_trellis(models, imageOf3D):
    """Helper function to run trellis pipeline and scoring."""
    trellis_pipeline = models["trellis_pipeline"]

    features = extract_simple_features(imageOf3D)
    sampler_cfg = choose_sampler_params(features)
    
    # Get seed from config
    seed = CONFIG['generation']['seed']
    formats = 'gaussian'
    
    # Enhance image contrast
    # imageOf3D = ImageEnhance.Contrast(imageOf3D).enhance(1.2)
    
    trellis_outputs = trellis_pipeline.run(
        imageOf3D,
        seed=seed,
        formats=formats,
        # # Higher steps = better quality & geometry consistency
        # sparse_structure_sampler_params={
        #     "steps": 50,
        #     "cfg_strength": 3.0  # Controls structure guidance (7.5 is good baseline)
        # },
        # slat_sampler_params={
        #     "steps": 6,
        #     "cfg_strength": 3.0  # Increased from 3.0 for stronger detail guidance
        # },
        sparse_structure_sampler_params=sampler_cfg["sparse_structure_sampler_params"],
        slat_sampler_params=sampler_cfg["slat_sampler_params"],
    )
    # ply_dalle_path = "temp.ply"
    gaussian = trellis_outputs['gaussian'][0]

    ed = features["edge_density"]
    ent = features["entropy"]

    if ed > 0.045 or ent > 3.5:
        logger.warning(f"High complexity image detected (edge_density={ed:.4f}, entropy={ent:.2f}), skipping refinement")
        return gaussian
    # gaussian.save_ply(ply_dalle_path)
    try:
        # Save temporary PLY file
        temp_ply = "temp_before_refine.ply"
        gaussian.save_ply(temp_ply)
        
        # Load point cloud with Open3D
        pcd = o3d.io.read_point_cloud(temp_ply)
        num_points_before = len(pcd.points)
        
        # Apply statistical outlier removal if we have enough points
        refinement_cfg = CONFIG['refinement']
        if refinement_cfg['enabled'] and num_points_before > refinement_cfg['min_points']:
            logger.warning(f"Refining PLY file with Open3D: {temp_ply}")
            
            # Statistical outlier removal - parameters from config
            # nb_neighbors: number of neighbors to consider (higher = smoother)
            # std_ratio: threshold (higher = keeps more points, preserves geometry)
            pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=refinement_cfg['nb_neighbors'],
                std_ratio=refinement_cfg['std_ratio']
            )
            
            num_points_after = len(pcd_filtered.points)
            removed_points = num_points_before - num_points_after
            
            logger.warning(f"Outlier removal: {removed_points} points removed ({removed_points/num_points_before*100:.1f}%)")
            
            # Convert inlier_indices to numpy array for filtering
            inlier_mask = np.array(inlier_indices)
            
            # Load original PLY with plyfile to preserve all Gaussian Splatting properties
            plydata = PlyData.read(temp_ply)
            vertex = plydata['vertex']
            
            # Filter vertices using inlier mask
            filtered_vertex = vertex[inlier_mask]
            
            # Create new PLY with filtered data
            refined_ply = temp_ply.replace('.ply', '_refined.ply')
            new_vertex = PlyElement.describe(filtered_vertex, 'vertex')
            PlyData([new_vertex], text=False).write(refined_ply)
            
            logger.warning(f"Refined PLY saved with {num_points_after} points (preserved all Gaussian Splatting properties)")
            
            gaussian.load_ply(refined_ply)
            logger.warning(f"Refined PLY loaded back into Gaussian object")
            
            # Clean up temporary files
            if os.path.exists(temp_ply):
                os.remove(temp_ply)
            if os.path.exists(refined_ply):
                os.remove(refined_ply)
        else:
            logger.warning(f"Skipping refinement (only {num_points_before} points)")
            if os.path.exists(temp_ply):
                os.remove(temp_ply)
            
    except Exception as e:
        logger.warning(f"PLY refinement with Open3D failed, using original: {e}")
        # Clean up if temp file exists
        if os.path.exists("temp_before_refine.ply"):
            try:
                os.remove("temp_before_refine.ply")
            except:
                pass
        if os.path.exists("temp_before_refine_refined.ply"):
            try:
                os.remove("temp_before_refine_refined.ply")
            except:
                pass

    return gaussian

async def run_trellis_with_timeout(models, imageOf3D, timeout_seconds=None):
    """Async wrapper for run_trellis with timeout."""
    if timeout_seconds is None:
        timeout_seconds = CONFIG['generation']['timeout_seconds']
    try:
        # Run the synchronous function in a thread pool to make it cancellable
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, run_trellis, models, imageOf3D)
        
        # Wait for completion or timeout
        return await asyncio.wait_for(task, timeout=timeout_seconds)
        
    except asyncio.TimeoutError:
        logger.warning(f"3D generation timed out after {timeout_seconds} seconds")
        raise

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
async def root():
    return {"message": "3D Generation API is running"}

@app.get("/memory/status")
async def memory_status():
    """Get current GPU memory status."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_info[f"gpu_{i}"] = {
            "allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
            "cached_gb": torch.cuda.memory_reserved(i) / 1024**3,
            "total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
            "utilization": (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100
        }
    return memory_info

@app.post("/memory/cleanup")  
async def manual_cleanup():
    """Manually trigger memory cleanup."""
    before_info = await memory_status()
    await cleanup_gpu_memory()
    after_info = await memory_status()
    
    return {
        "message": "Memory cleanup completed",
        "before": before_info,
        "after": after_info
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with memory info."""
    health_status = {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": list(app.state.models.keys()) if hasattr(app.state, 'models') else [],
        "gpu_memory": await memory_status() if torch.cuda.is_available() else None
    }
    return health_status

@app.post("/generate")
async def generate(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    """Generate 3D model from text/image data with 25-second timeout"""
    # Use semaphore to prevent concurrent GPU interference
    async with app.state.processing_semaphore:
        t0 = time()
        logger.warning(f"Starting generation...")
        models = app.state.models
    try:
        # Clean up GPU memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        contents = await prompt_image_file.read()
        imageOf3D = Image.open(BytesIO(contents))
        # Set seeds for reproducibility without forcing deterministic algorithms
        # This provides consistency while maintaining performance
        seed = CONFIG['generation']['seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Set numpy seed for any numpy operations
        import numpy as np
        np.random.seed(seed)
        
        # Handle different task types
        # Image task - use base64 image data directly
        logger.warning(f"Processing image task...")
        try:
            elapsed_0 = time() - t0
            timeout_seconds = CONFIG['generation']['timeout_seconds']
            remaining_timeout = max(1, timeout_seconds - elapsed_0)  # At least 1 second remaining
            
            prompt_image = imageOf3D
            if len(prompt_image.getbands()) == 4:
                prompt_image = add_white_background(prompt_image)
            prompt_image = prompt_image.convert("RGB")
            np_img = np.asarray(prompt_image)
            torch_prompt_image = torch.from_numpy(np_img)
            prompt = torch_prompt_image
            

            gaussian = await run_trellis_with_timeout(models, imageOf3D)

            buffer = BytesIO()
            gaussian.save_ply_cuda(buffer)
            buffer.seek(0)
            
        except Exception as e:
            logger.warning(f"Error processing image task: {e}")
            buffer = BytesIO()
            buffer.write(b"")
            buffer.seek(0)
            return Response(buffer.getbuffer(), media_type="application/octet-stream")
        
        t1 = time()
        logger.warning(f"Generation took: {(t1 - t0):.2f} seconds")
        
        return Response(buffer.getbuffer(), media_type="application/octet-stream")
        
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU out of memory during generation")
        await cleanup_gpu_memory()
        buffer = await process_error(prompt)
        return Response(buffer.getbuffer(), media_type="application/octet-stream")
    except Exception as e:
        logger.warning(f"Error during generation: {str(e)}", exc_info=True)
        # Return empty PLY file on error
        buffer = await process_error(prompt)
        return Response(buffer.getbuffer(), media_type="application/octet-stream")
    finally:
        # ALWAYS cleanup after each request - this is the key benefit!
        await cleanup_gpu_memory()
        t1 = time()
        logger.warning(f"Generation completed in {(t1 - t0):.2f} seconds")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10006)
        