from datetime import datetime
from uuid import uuid4
import time

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict

from backend.app.core.jobs import EvaluationJob
from backend.app.core.state import job_registry
from evaluations.evaluator import CustomEvaluator
from utils import get_device

router = APIRouter(
    prefix="/api/evaluations",
    tags=["evaluations"],
    responses={404: {"description": "Not found"}},
)

class EvalRequest(BaseModel):
    checkpoint_id: str
    tasks: List[str]
    limit: Optional[float] = None
    batch_size: int = 8

class EvalJobStatus(BaseModel):
    job_id: str
    status: str # "pending", "running", "completed", "failed", "error"
    tasks: List[str]
    created_at: float
    progress: float = 0.0
    error: Optional[str] = None
    results: Optional[Dict] = None

# Common tasks supported by lm-eval
# Task metadata with educational descriptions
TASK_METADATA = {
    "hellaswag": {
        "name": "HellaSwag",
        "description": "Tests commonsense reasoning by asking the model to complete a sentence describing a daily life situation.",
        "metrics": ["acc", "acc_norm"]
    },
    "mmlu": {
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding. Covers 57 subjects across STEM, the humanities, the social sciences, and more.",
        "metrics": ["acc", "acc_norm"]
    },
    "winogrande": {
        "name": "WinoGrande",
        "description": "Tests commonsense reasoning using pronoun resolution problems that require world knowledge to solve.",
        "metrics": ["acc"]
    },
    "arc_challenge": {
        "name": "ARC (Challenge)",
        "description": "Grade-school science questions that are hard to answer with simple retrieval or correlation.",
        "metrics": ["acc", "acc_norm"]
    },
    "arc_easy": {
        "name": "ARC (Easy)",
        "description": "Grade-school science questions that are generally answerable with simple retrieval.",
        "metrics": ["acc", "acc_norm"]
    },
    "lambada": {
        "name": "LAMBADA",
        "description": "Tests the ability to predict the last word of a sentence where the word is only predictable given the broader context.",
        "metrics": ["acc", "perplexity"]
    },
    "piqa": {
        "name": "PIQA",
        "description": "Physical Interaction QA. Tests knowledge about physical mechanics and how to interact with everyday objects.",
        "metrics": ["acc", "acc_norm"]
    },
    "openbookqa": {
        "name": "OpenBookQA",
        "description": "QA tasks that require multiple steps of reasoning and use of common knowledge.",
        "metrics": ["acc", "acc_norm"]
    },
    "boolq": {
        "name": "BoolQ",
        "description": "Yes/no question answering task based on text passages.",
        "metrics": ["acc"]
    },
    "sciq": {
        "name": "SciQ",
        "description": "Science exam questions about Physics, Chemistry and Biology.",
        "metrics": ["acc", "acc_norm"]
    },
    "wikitext": {
        "name": "WikiText",
        "description": "Language modeling benchmark using Wikipedia articles. Good for testing general text generation capability.",
        "metrics": ["perplexity"]
    },
}

@router.get("/tasks")
async def get_tasks():
    """Get list of supported evaluation tasks with metadata."""
    return {"tasks": TASK_METADATA}

@router.post("/run", response_model=EvalJobStatus)
async def run_evaluation(request: EvalRequest):
    """Start a new evaluation job."""
    
    # Resolve checkpoint path
    from pathlib import Path
    checkpoint_dir = Path("checkpoints") / request.checkpoint_id
    
    if not checkpoint_dir.exists():
        if Path(request.checkpoint_id).exists():
           checkpoint_path = Path(request.checkpoint_id)
        else:
           raise HTTPException(status_code=404, detail=f"Checkpoint {request.checkpoint_id} not found")
    else:
        pt_files = list(checkpoint_dir.glob("*.pt"))
        if not pt_files:
             raise HTTPException(status_code=404, detail=f"No .pt files found in {checkpoint_dir}")
        checkpoint_path = sorted(pt_files)[-1]
        if (checkpoint_dir / "final_model.pt").exists():
            checkpoint_path = checkpoint_dir / "final_model.pt"

    job_id = str(uuid4())
    device = get_device()
    
    try:
        evaluator = CustomEvaluator(
            checkpoint_path=str(checkpoint_path),
            device=device,
            batch_size=request.batch_size
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize evaluator: {str(e)}")

    job = EvaluationJob(
        job_id=job_id,
        evaluator=evaluator,
        tasks=request.tasks,
        created_at=time.time()
    )
    
    job_registry.add(job)
    job.start()
    
    return EvalJobStatus(
        job_id=job.job_id,
        status=job.status,
        tasks=job.tasks,
        created_at=job.created_at
    )

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an evaluation job."""
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check type if needed, but registry handles union
    if not isinstance(job, EvaluationJob):
         # Could define a way to inspect generic jobs, but for now specific to EvalJob
         raise HTTPException(status_code=404, detail="Job is not an evaluation job")

    payload = job._status_payload()
    return payload

@router.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get results of a completed evaluation job."""
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not isinstance(job, EvaluationJob):
         raise HTTPException(status_code=404, detail="Job is not an evaluation job")
        
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job.results:
        raise HTTPException(status_code=500, detail="Results missing from completed job")
        
    return job.results

@router.get("/jobs")
async def list_jobs():
    """List all recent evaluation jobs."""
    # Filter for EvaluationJob
    all_jobs = job_registry.list().values()
    eval_jobs = [j for j in all_jobs if isinstance(j, EvaluationJob)]
    
    eval_jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return {"jobs": [
        j._status_payload() for j in eval_jobs
    ]}
